"""LangChain-orchestrated RAG retriever.

Pipeline:
  1. Hybrid retrieval: Chroma MMR + BM25, deduped per source, fused via RRF.
  2. Cross-encoder reranker filters candidates.
  3. Grounded generation over surviving documents.
"""

import json
import logging
import time
from pathlib import Path
from typing import AsyncGenerator, List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

from app.config import settings
from app.rag.chains import build_generator_chain, collect_citations, stream_answer
from app.rag.hybrid import load_bm25, reciprocal_rank_fusion

log = logging.getLogger(__name__)


# ---------- Paths ----------
REPO_ROOT = Path(__file__).resolve().parents[2]
BM25_CHUNKS_PATH = REPO_ROOT / "data" / "bm25_chunks.json"
INDEX_DIR = REPO_ROOT / "data" / "chroma_index"
COLLECTION_NAME = "disaster_damage_kb"

# ---------- Models ----------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ---------- Retrieval tuning ----------
# Hybrid retrieval — top-K from each retriever before fusion
HYBRID_K_VECTOR = 8
HYBRID_K_BM25 = 8
HYBRID_FINAL_K = 10  # how many fused results to send to the reranker
RETRIEVAL_FETCH_K = 25  # MMR pool size for vector search
RETRIEVAL_LAMBDA = 0.5  # 0 = max diversity, 1 = max similarity
MAX_PER_SOURCE = 2  # cap chunks from any single document (dedupe)

# RRF weighting — vector wins ties; BM25 acts as a tiebreaker.
# Inverted from the default because one long document (FEMA PDA Guide)
# was saturating BM25 and crowding out relevant shorter sources.
RRF_VECTOR_WEIGHT = 1.3
RRF_BM25_WEIGHT = 1.0

# Reranker — ms-marco scores are unbounded; negative = not relevant.
RERANKER_THRESHOLD = 0.0

# Generation
MAX_GRADED_DOCS = 5  # cap on docs sent to generator
MIN_GRADED_DOCS_FOR_GEN = 1  # need at least this many to attempt generation

# Phrases that indicate the generator refused to answer (insufficient context).
# Kept conservative — overly broad markers like "no specific" or "not mentioned"
# trigger false positives on valid answers that use those phrases in passing.
REFUSAL_MARKERS = (
    "sources don't contain",
    "sources do not contain",
    "context does not contain",
    "context doesn't contain",
    "provided sources do not",
    "provided sources don't",
    "cannot answer that",
    "not enough information to answer",
    "isn't enough information to answer",
    "the documents do not",
    "the documents don't",
)


def _is_refusal(text: str) -> bool:
    """True if the generator output looks like a refusal to answer."""
    if not text:
        return True
    lowered = text.lower()
    return any(m in lowered for m in REFUSAL_MARKERS)


def _dedupe_by_source(docs: List[Document], max_per_source: int) -> List[Document]:
    """Keep at most N chunks from any single source, preserving order."""
    counts: dict[str, int] = {}
    out: List[Document] = []
    for d in docs:
        src = d.metadata.get("source", "?")
        if counts.get(src, 0) < max_per_source:
            out.append(d)
            counts[src] = counts.get(src, 0) + 1
    return out


class KnowledgeRetriever:
    _instance: Optional["KnowledgeRetriever"] = None

    def __init__(self) -> None:
        self._vector_store: Optional[Chroma] = None
        self._bm25 = None
        self._reranker: Optional[CrossEncoder] = None
        self._generator = None
        self._generator_llm: Optional[ChatGoogleGenerativeAI] = None

        if not INDEX_DIR.exists():
            log.warning(
                "No vector index at %s. Run `python -m app.rag.build_index` first.",
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

    # ------------------------------------------------------------------
    # Synchronous retrieval
    # ------------------------------------------------------------------
    def retrieve(self, query: str) -> Optional[str]:
        if not query or not query.strip() or self._vector_store is None:
            return None

        candidates = self._search(query)
        if not candidates:
            return None

        graded = self._rerank(query, candidates)
        if len(graded) < MIN_GRADED_DOCS_FOR_GEN:
            log.info(
                "Reranker dropped everything; falling back to top-%d candidates.",
                MAX_GRADED_DOCS,
            )
            graded = candidates[:MAX_GRADED_DOCS]

        return self._generate(query, graded[:MAX_GRADED_DOCS])

    def _search(self, query: str) -> List[Document]:
        """Hybrid retrieval: vector (MMR) + BM25, deduped per source, fused via RRF."""
        if self._vector_store is None:
            return []

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

        bm25_hits: List[Document] = []
        if self._bm25 is not None:
            try:
                bm25_hits = self._bm25.invoke(query)
            except Exception:
                log.exception("BM25 retrieval failed for query=%r", query)

        # Dedupe per source so one long document can't monopolize results
        vector_hits = _dedupe_by_source(vector_hits, max_per_source=MAX_PER_SOURCE)
        bm25_hits = _dedupe_by_source(bm25_hits, max_per_source=MAX_PER_SOURCE)

        # Single-retriever fallback
        if not bm25_hits:
            return vector_hits[:HYBRID_FINAL_K]
        if not vector_hits:
            return bm25_hits[:HYBRID_FINAL_K]

        fused = reciprocal_rank_fusion(
            [vector_hits, bm25_hits],
            weights=[RRF_VECTOR_WEIGHT, RRF_BM25_WEIGHT],
        )
        log.info(
            "Hybrid: vector=%d bm25=%d fused=%d (returning top %d)",
            len(vector_hits),
            len(bm25_hits),
            len(fused),
            HYBRID_FINAL_K,
        )
        return fused[:HYBRID_FINAL_K]

    def _rerank(self, query: str, docs: List[Document]) -> List[Document]:
        """Score (query, doc) pairs with the cross-encoder; keep above threshold."""
        if not docs or self._reranker is None:
            return docs

        pairs = [(query, d.page_content) for d in docs]
        try:
            scores = self._reranker.predict(pairs)
        except Exception:
            log.exception("Reranker failed; returning all docs unranked.")
            return docs

        scored = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored if score > RERANKER_THRESHOLD]

    def _generate(self, query: str, docs: List[Document]) -> Optional[str]:
        """Generate a grounded answer; return None on refusal or failure."""
        if self._generator is None:
            return None

        # Swap in pre-chunking original text where available (better context).
        for d in docs:
            original = d.metadata.get("original_text")
            if original:
                d.page_content = original

        answer = None
        for attempt in range(3):
            try:
                answer = self._generator.invoke({"question": query, "documents": docs})
                break
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    wait = 15 * (attempt + 1)
                    log.warning(
                        "Rate limited on generate, waiting %ds (attempt %d/3)",
                        wait,
                        attempt + 1,
                    )
                    time.sleep(wait)
                else:
                    log.exception("Generator failed for query=%r", query)
                    return None

        if not answer or _is_refusal(answer):
            log.info("Generator refused or returned empty for query=%r", query)
            return None

        sources = collect_citations(docs)
        return f"{answer}\n\nSources: {', '.join(sources)}" if sources else answer

    # ------------------------------------------------------------------
    # Streaming retrieval (unused by current chat_service path but kept
    # available; emits tokens then a sentinel with citations).
    # ------------------------------------------------------------------
    async def retrieve_stream(self, query: str) -> AsyncGenerator[str, None]:
        if not query or not query.strip():
            return
        if self._vector_store is None or self._generator_llm is None:
            return

        candidates = self._search(query)
        if not candidates:
            return

        graded = self._rerank(query, candidates)
        if not graded:
            graded = candidates[:MAX_GRADED_DOCS]

        docs = graded[:MAX_GRADED_DOCS]
        for d in docs:
            original = d.metadata.get("original_text")
            if original:
                d.page_content = original

        full_answer = ""
        async for token in stream_answer(query, docs, self._generator_llm):
            full_answer += token
            yield token

        sources = collect_citations(docs)
        if sources and full_answer and not _is_refusal(full_answer):
            yield f"\x00SOURCES:{json.dumps(sources)}"
