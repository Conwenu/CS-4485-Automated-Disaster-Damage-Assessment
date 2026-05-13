"""LangChain-orchestrated RAG retriever.

Pipeline:
  1. Hybrid retrieval: Chroma MMR + BM25, fused via RRF.
  2. Cross-encoder reranker filters candidates.
  3. Grounded generation over surviving documents.
"""

import time
import logging
from pathlib import Path
from typing import List, Optional, AsyncGenerator

from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from app.rag.hybrid import load_bm25, reciprocal_rank_fusion
from app.rag.chains import (
    build_generator_chain,
    collect_citations,
)

from app.config import settings

log = logging.getLogger(__name__)


# ---------- Paths ----------
REPO_ROOT = Path(__file__).resolve().parents[2]
BM25_CHUNKS_PATH = REPO_ROOT / "data" / "bm25_chunks.json"
INDEX_DIR = REPO_ROOT / "data" / "chroma_index"
COLLECTION_NAME = "disaster_damage_kb"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_THRESHOLD = 0.0  # keep everything above this score
# ms-marco scores are unbounded — negative = not relevant
# 0.0 is a reasonable cutoff; tune if needed

# ---------- Retrieval tuning ----------
RETRIEVAL_K = 15  # candidates the grader sees
RETRIEVAL_FETCH_K = 25  # MMR pool size
RETRIEVAL_LAMBDA = 0.5  # 0 = max diversity, 1 = max similarity
MAX_GRADED_DOCS = 5  # cap on docs sent to generator
MIN_GRADED_DOCS_FOR_GEN = 1  # need at least this many to attempt generation


# Hybrid retrieval — top-K from each retriever before RRF fusion
HYBRID_K_VECTOR = 8
HYBRID_K_BM25 = 8
HYBRID_FINAL_K = 6  # how many fused results to send to the grader

REFUSAL_MARKERS = (
    "don't contain",
    "do not contain",
    "no information",
    "cannot answer",
    "isn't enough information",
    "not enough information",
    "do not state",
    "does not state",
    "does not contain",
    "no specific",
    "not mentioned",
)


class KnowledgeRetriever:
    _instance: Optional["KnowledgeRetriever"] = None

    def __init__(self) -> None:
        self._vector_store: Optional[Chroma] = None
        self._generator = None
        self._generator_llm = None

        if not INDEX_DIR.exists():
            log.warning(
                "No vector index at %s. Run `python -m app.rag.build_index` "
                "first. Falling back to seed-facts-only mode.",
                INDEX_DIR,
            )
            return

        log.info("Loading embedding model and Chroma index...")
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self._vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(INDEX_DIR),
        )

        self._bm25 = None
        if BM25_CHUNKS_PATH.exists():
            log.info("Loading BM25 corpus from %s", BM25_CHUNKS_PATH)
            try:
                self._bm25 = load_bm25(BM25_CHUNKS_PATH, k=HYBRID_K_BM25)
            except Exception:
                log.exception("Failed to load BM25; falling back to vector-only.")
        else:
            log.warning(
                "No BM25 corpus at %s. Run `python -m app.rag.build_index` "
                "to enable hybrid retrieval.",
                BM25_CHUNKS_PATH,
            )

        log.info("Loading reranker model (%s)...", RERANKER_MODEL)
        self._reranker = CrossEncoder(RERANKER_MODEL)
        log.info("Reranker loaded.")

        log.info("Building generator chain...")
        self._generator = build_generator_chain()

        self._generator_llm = ChatGoogleGenerativeAI(
            model=settings.GOOGLE_MODEL,
            temperature=0,
            google_api_key=settings.GOOGLE_API_KEY,
        )

        log.info("KnowledgeRetriever ready.")

    @classmethod
    def get(cls) -> "KnowledgeRetriever":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_generator_llm(self):
        return self._generator_llm

    def retrieve(self, query: str) -> Optional[str]:
        if not query or not query.strip():
            return None

        if self._vector_store is None:
            # print("retrieve: vector store is None")
            return None

        candidates = self._search(query)
        # print(f"retrieve: {len(candidates)} candidates for query={query!r}")
        if not candidates:
            return None

        graded = self._grade(query, candidates)
        # print(f"retrieve: {len(graded)}/{len(candidates)} graded")

        if len(graded) < MIN_GRADED_DOCS_FOR_GEN:
            print(f"retrieve: fallback to top-{MAX_GRADED_DOCS} ungraded")
            graded = candidates[:MAX_GRADED_DOCS]

        result = self._generate(query, graded[:MAX_GRADED_DOCS])
        # print(f"retrieve: generate returned {repr(result)[:100] if result else 'None'}")
        return result

    def _search(self, query: str) -> List[Document]:
        # Vector search (semantic)
        vector_hits: List[Document] = []
        try:
            vector_hits = self._vector_store.max_marginal_relevance_search(
                query,
                k=HYBRID_K_VECTOR,
                fetch_k=RETRIEVAL_FETCH_K,
                lambda_mult=RETRIEVAL_LAMBDA,
            )
        except Exception:
            log.exception("Vector retrieval failed for query=%r", query)

        # BM25 search (keyword) — optional, falls back to vector-only if missing
        bm25_hits: List[Document] = []
        if self._bm25 is not None:
            try:
                bm25_hits = self._bm25.invoke(query)
            except Exception:
                log.exception("BM25 retrieval failed for query=%r", query)

        # If only one retriever produced anything, just return that
        if not bm25_hits:
            return vector_hits[:HYBRID_FINAL_K]
        if not vector_hits:
            return bm25_hits[:HYBRID_FINAL_K]

        # Fuse via reciprocal rank fusion
        fused = reciprocal_rank_fusion([vector_hits, bm25_hits], weights=[1.0, 1.3])
        log.info(
            "Hybrid: vector=%d bm25=%d fused=%d (returning top %d)",
            len(vector_hits),
            len(bm25_hits),
            len(fused),
            HYBRID_FINAL_K,
        )
        return fused[:HYBRID_FINAL_K]

    def _grade(self, query: str, docs: List[Document]) -> List[Document]:
        if not docs:
            return []

        pairs = [(query, d.page_content) for d in docs]
        try:
            scores = self._reranker.predict(pairs)
        except Exception:
            log.exception("Reranker failed; returning all docs unranked.")
            return docs

        scored = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)

        kept = [doc for score, doc in scored if score > RERANKER_THRESHOLD]
        return kept

    def _generate(self, query: str, docs: List[Document]) -> Optional[str]:
        # print(f"_generate: called with {len(docs)} docs for query={query!r}")
        for d in docs:
            original = d.metadata.get("original_text")
            if original:
                d.page_content = original

        for attempt in range(3):
            try:
                answer = self._generator.invoke(
                    {
                        "question": query,
                        "documents": docs,
                    }
                )
                break
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    wait = 15 * (attempt + 1)
                    print(f"_generate: rate limited, waiting {wait}s")
                    time.sleep(wait)
                    answer = None
                else:
                    print(f"_generate: exception {e}")
                    return None
        else:
            return None

        if not answer:
            return None

        print(f"_generate: raw answer={repr(answer)[:150]}")

        refusal_markers = (
            "don't contain",
            "do not contain",
            "no information",
            "cannot answer",
            "isn't enough information",
            "not enough information",
            "do not state",
            "does not state",
            "does not contain",
            "no specific",
            "not mentioned",
        )
        if any(m in answer.lower() for m in refusal_markers):
            print(f"_generate: REFUSED — answer contains refusal marker")
            return None

        sources = collect_citations(docs)
        if sources:
            return f"{answer}\n\nSources: {', '.join(sources)}"
        return answer

    async def retrieve_stream(
        self,
        query: str,
    ) -> AsyncGenerator[str, None]:
        """Stream the RAG answer token by token.

        Same pipeline as retrieve() but yields tokens from the generator
        instead of returning a complete string.

        Yields:
            str tokens during generation
            After all tokens: yields a special sentinel with citations
        """
        import json
        from app.rag.chains import stream_answer

        if not query or not query.strip():
            return

        if self._vector_store is None or self._generator_llm is None:
            return

        # Retrieval + reranking (same as retrieve())
        candidates = self._search(query)
        if not candidates:
            return

        graded = self._grade(query, candidates)
        if not graded:
            graded = candidates[:MAX_GRADED_DOCS]

        docs = graded[:MAX_GRADED_DOCS]

        # Use original text for generation
        for d in docs:
            original = d.metadata.get("original_text")
            if original:
                d.page_content = original

        # Stream tokens from Gemini
        full_answer = ""
        async for token in stream_answer(query, docs, self._generator_llm):
            full_answer += token
            yield token

        # After all tokens, yield citations as a special JSON sentinel
        # The caller detects this and handles it separately
        sources = collect_citations(docs)
        if sources and full_answer:
            # Check for refusal before yielding citation sentinel
            if not any(m in full_answer.lower() for m in REFUSAL_MARKERS):
                yield f"\x00SOURCES:{json.dumps(sources)}"
