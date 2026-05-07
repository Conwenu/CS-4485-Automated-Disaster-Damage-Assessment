"""LangChain Runnables for the RAG pipeline.

Two chains:
  - grader_chain: takes (query, doc) -> bool. Decides if doc is actually
    relevant. Filters retrieval noise before generation.
  - generator_chain: takes (query, docs) -> str. Writes a grounded answer
    that strictly cites the provided context.

Both share one ChatGoogleGenerativeAI instance to avoid duplicate auth setup.
"""
from typing import List, Tuple

from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda

from app.config import settings


def _build_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=settings.GOOGLE_MODEL,
        temperature=0,
        google_api_key=settings.GOOGLE_API_KEY,
    )


class _GraderVerdict(BaseModel):
    is_relevant: bool = Field(
        description=(
            "True if the document genuinely contains information that answers "
            "or directly supports the question. False if it merely mentions "
            "the keywords without addressing them, or is off-topic."
        )
    )
    reasoning: str = Field(
        description="One short sentence explaining the verdict."
    )


_GRADER_SYSTEM = """You are a relevance grader for a RAG system over a small,
trusted corpus of two documents:
  - The xBD research paper (Gupta et al., 2019), describing the xBD dataset.
  - The FEMA Preliminary Damage Assessment Guide (2025).

You receive ONE candidate document chunk and ONE user question. Decide whether
the chunk is RELEVANT to the question.

CRITICAL — SOURCE SELF-REFERENCE:
When a question asks about "xBD" (the dataset) or "FEMA's process," chunks from
the xBD paper or the FEMA guide ARE answering that question, even when they don't
mention "xBD" or "FEMA" by name. The xBD paper IS xBD. The FEMA guide IS FEMA.

For example:
- Question: "What metric does xBD use?"
  Chunk says: "we attained an overall weighted F1 score of 0.2654"
  → RELEVANT. The paper IS xBD; "we" means the xBD authors.
- Question: "How does FEMA classify damage?"
  Chunk says: "Inspectors categorize each structure as Destroyed, Major..."
  → RELEVANT. The guide IS FEMA's process.

RELEVANT means:
- The chunk discusses the topic of the question (a metric, process, definition,
  procedure, criterion, or description related to the subject).
- The chunk contains specific facts, numbers, methods, or descriptions that
  could form part of an answer.
- For questions about xBD or FEMA: a chunk from those documents on the right
  topic is relevant, even without naming xBD or FEMA explicitly.

NOT RELEVANT means:
- The chunk is purely a table of contents, header, citation, or appendix
  listing with no substantive content.
- The chunk is on a clearly different subject than the question.

Lean toward RELEVANT when on-topic. False negatives lose information; false
positives waste a few tokens. When uncertain, return True."""


def build_grader_chain() -> Runnable:
    """Returns a Runnable: dict(question, document) -> _GraderVerdict."""
    llm = _build_llm()
    structured = llm.with_structured_output(_GraderVerdict, method="function_calling")

    prompt = ChatPromptTemplate.from_messages([
        ("system", _GRADER_SYSTEM),
        ("human",
         "Question:\n{question}\n\n"
         "Document chunk:\n{document}\n\n"
         "Is this chunk relevant?"),
    ])

    return prompt | structured


_GENERATOR_SYSTEM = """You are answering a user's question using ONLY the
context passages provided. The context comes from authoritative sources
(xBD research paper and FEMA Preliminary Damage Assessment Guide).

STRICT RULES:
- Use ONLY information present in the context. Never add outside knowledge.
- If the context does not contain enough information to answer, say so plainly:
  "The provided sources don't contain a clear answer to that."
- Keep the answer to 2-4 sentences. Be precise, not flowery.
- Do not begin with phrases like "Based on the context" or "According to the
  sources." Just state the answer.
- Do not list sources inline; the system adds citations automatically.

Write in a neutral, informative tone. Do not speculate."""


def _format_docs(docs: List[Document]) -> str:
    """Render docs as a numbered context block for the prompt."""
    parts = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page")
        cite = f"{src}" + (f", p.{page + 1}" if page is not None else "")
        parts.append(f"[{i}] ({cite})\n{d.page_content.strip()}")
    return "\n\n".join(parts)


def build_generator_chain() -> Runnable:
    """Returns a Runnable: dict(question, documents) -> str."""
    llm = _build_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", _GENERATOR_SYSTEM),
        ("human", "Question: {question}\n\nContext:\n{context}\n\nAnswer:"),
    ])

    return (
        RunnableLambda(
            lambda x: {
                "question": x["question"],
                "context": _format_docs(x["documents"]),
            }
        )
        | prompt
        | llm
        | RunnableLambda(lambda msg: (msg.content or "").strip())
    )



def collect_citations(docs: List[Document]) -> List[str]:
    """Return a deduplicated, ordered list of source citations."""
    seen = set()
    out = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        if src not in seen:
            seen.add(src)
            out.append(src)
    return out