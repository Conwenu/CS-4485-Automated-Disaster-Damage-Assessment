"""LangChain Runnables for the RAG pipeline.

Two chains:
  - grader_chain: takes (query, doc) -> bool. Decides if doc is actually
    relevant. Filters retrieval noise before generation.
  - generator_chain: takes (query, docs) -> str. Writes a grounded answer
    that strictly cites the provided context.

Both share one ChatGoogleGenerativeAI instance to avoid duplicate auth setup.
"""

from typing import List, AsyncGenerator

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
    reasoning: str = Field(description="One short sentence explaining the verdict.")


_GRADER_SYSTEM = """You are a lenient relevance grader for a RAG system over
disaster assessment documents.

You receive ONE document chunk and ONE question. Your job is simple: decide
whether this chunk contains ANY information that could help answer the question,
even partially or indirectly.

Mark RELEVANT (True) when:
- The chunk discusses the same topic, event, or subject as the question
- It contains facts, numbers, names, or descriptions related to what is asked
- The answer can be inferred or synthesized from the chunk's content
- The chunk uses different words but refers to the same concept
  (e.g. "economic loss" is relevant to "how much did it cost?")
  (e.g. "structures destroyed" is relevant to "how many homes burned?")
  (e.g. "suppression costs" is relevant to "what was the financial damage?")
- It provides partial information (the generator can combine multiple chunks)

Mark NOT RELEVANT (False) ONLY when:
- The chunk is clearly about a completely different subject
- It is a navigation element: table of contents, page header, footer,
  citation list, or acknowledgments section with no substantive content
- It contains zero information that could contribute to answering the question

KEY RULE: When in doubt, return True. A false negative (dropping a
relevant chunk) is much more harmful than a false positive (keeping a
marginally relevant chunk). The generator will ignore information it
does not need. It cannot use information it was never shown.

Do NOT require exact keyword matches. Reason about meaning and topic,
not surface form."""


def build_grader_chain() -> Runnable:
    """Returns a Runnable: dict(question, document) -> _GraderVerdict."""
    llm = _build_llm()
    structured = llm.with_structured_output(_GraderVerdict, method="function_calling")

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _GRADER_SYSTEM),
            (
                "human",
                "Question:\n{question}\n\n"
                "Document chunk:\n{document}\n\n"
                "Is this chunk relevant?",
            ),
        ]
    )

    return prompt | structured


_GENERATOR_SYSTEM = """You are answering a user's question using ONLY the
context passages provided. The context comes from authoritative sources
(xBD research paper and FEMA Preliminary Damage Assessment Guide).

STRICT RULES:
- Use ONLY information present in the context. Never add outside knowledge.
- Format the answer in clean Markdown when it improves clarity: use headings,
  bullet lists, bold for key terms. Plain prose is fine for short answers.
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

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _GENERATOR_SYSTEM),
            ("human", "Question: {question}\n\nContext:\n{context}\n\nAnswer:"),
        ]
    )

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


async def stream_answer(
    question: str,
    docs: List[Document],
    llm: ChatGoogleGenerativeAI,
) -> AsyncGenerator[str, None]:
    """Stream the generator response token by token using LangChain astream."""
    from langchain_core.messages import SystemMessage, HumanMessage

    context = _format_docs(docs)

    messages = [
        SystemMessage(content=_GENERATOR_SYSTEM),
        HumanMessage(content=f"Question: {question}\n\nContext:\n{context}\n\nAnswer:"),
    ]

    async for chunk in llm.astream(messages):
        token = chunk.content
        if token:
            yield token


async def stream_polished_answer(
    text: str,
    query: str,
    llm: ChatGoogleGenerativeAI,
) -> AsyncGenerator[str, None]:
    """Stream the polish LLM output token by token.

    Takes a complete base text, sends it to the polish LLM,
    and streams the reformatted markdown response as it arrives.
    """
    from langchain_core.messages import SystemMessage, HumanMessage

    messages = [
        SystemMessage(
            content=(
                "You are rewriting an answer for readability.\n"
                "Reformat the answer in proper Markdown. Choose the best "
                "structure yourself — use headings, bullet lists, bold for "
                "key terms, etc., wherever it improves clarity.\n"
                "STRICT RULES:\n"
                "- Do NOT change any numbers, percentages, or counts.\n"
                "- Do NOT add facts that aren't in the base text.\n"
                "- Keep it concise.\n"
                "Return only the rewritten answer."
            )
        ),
        HumanMessage(
            content=(
                f"User query: {query}\n\nBase answer:\n{text}\n\nRewritten answer:"
            )
        ),
    ]

    async for chunk in llm.astream(messages):
        token = chunk.content
        if token:
            yield token
