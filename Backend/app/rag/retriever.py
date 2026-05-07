"""LangChain-orchestrated RAG retriever.

Pipeline (in order):
  1. Seed-fact lookup (instant, deterministic, cited).
  2. Vector retrieval via Chroma + MMR.
  3. Per-document relevance grading (LangChain chain).
  4. Grounded generation over surviving documents (LangChain chain).

The retriever is a singleton — load the embedding model and index once.
"""
import logging
from pathlib import Path
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from app.rag.hybrid import load_bm25, reciprocal_rank_fusion
from app.rag.seed_facts import find_seed_fact
from app.rag.chains import (
    build_grader_chain,
    build_generator_chain,
    collect_citations,
)

log = logging.getLogger(__name__)


# ---------- Paths ----------
REPO_ROOT = Path(__file__).resolve().parents[2]
BM25_CHUNKS_PATH = REPO_ROOT / "data" / "bm25_chunks.json"
INDEX_DIR = REPO_ROOT / "data" / "chroma_index"
COLLECTION_NAME = "disaster_damage_kb"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ---------- Retrieval tuning ----------
RETRIEVAL_K = 5             # candidates the grader sees
RETRIEVAL_FETCH_K = 25      # MMR pool size
RETRIEVAL_LAMBDA = 0.6      # 0 = max diversity, 1 = max similarity
MAX_GRADED_DOCS = 4         # cap on docs sent to generator
MIN_GRADED_DOCS_FOR_GEN = 1 # need at least this many to attempt generation


# Hybrid retrieval — top-K from each retriever before RRF fusion
HYBRID_K_VECTOR = 8
HYBRID_K_BM25 = 8
HYBRID_FINAL_K = 6   # how many fused results to send to the grader

class KnowledgeRetriever:
    _instance: Optional["KnowledgeRetriever"] = None

    def __init__(self) -> None:
        self._vector_store: Optional[Chroma] = None
        self._grader = None
        self._generator = None

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
                log.exception(
                    "Failed to load BM25; falling back to vector-only.")
        else:
            log.warning(
                "No BM25 corpus at %s. Run `python -m app.rag.build_index` "
                "to enable hybrid retrieval.", BM25_CHUNKS_PATH)

        log.info("Building grader and generator chains...")
        self._grader = build_grader_chain()
        self._generator = build_generator_chain()

        log.info("KnowledgeRetriever ready.")

    @classmethod
    def get(cls) -> "KnowledgeRetriever":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def retrieve(self, query: str) -> Optional[str]:
        """Return a grounded answer for the query, or None if nothing useful."""
        if not query or not query.strip():
            return None

        # 1. Seed facts (fast path)
        seed = find_seed_fact(query)
        if seed:
            log.info("Seed-fact match for query=%r", query)
            return f"{seed['answer']}\n\nSource: {seed['source']}"

        # 2. Vector retrieval
        if self._vector_store is None:
            return None

        candidates = self._search(query)
        if not candidates:
            log.info("No vector hits for query=%r", query)
            return None

        # 3. Grading — only keep docs that genuinely answer the query
        graded = self._grade(query, candidates)
        if len(graded) < MIN_GRADED_DOCS_FOR_GEN:
            log.info(
                "Grader rejected all %d candidates for query=%r; "
                "falling back to top-%d ungraded.",
                len(candidates), query, MAX_GRADED_DOCS,
            )
            graded = candidates[:MAX_GRADED_DOCS]

        # 4. Generation
        return self._generate(query, graded[:MAX_GRADED_DOCS])

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
        fused = reciprocal_rank_fusion([vector_hits, bm25_hits])
        log.info(
            "Hybrid: vector=%d bm25=%d fused=%d (returning top %d)",
            len(vector_hits), len(bm25_hits), len(fused), HYBRID_FINAL_K,
        )
        return fused[:HYBRID_FINAL_K]

    def _grade(self, query: str, docs: List[Document]) -> List[Document]:
        kept: List[Document] = []
        for d in docs:
            try:
                verdict = self._grader.invoke({
                    "question": query,
                    "document": d.page_content,
                })
                if verdict.is_relevant:
                    kept.append(d)
                else:
                    log.debug("Grader rejected: %s", verdict.reasoning)
            except Exception:
                log.exception("Grader call failed; keeping doc as fallback")
                kept.append(d)
        return kept


    def _generate(self, query: str, docs: List[Document]) -> Optional[str]:
        
        # For generation, prefer the original chunk text (without the prepended context tag) so the generator doesn't echo the tag into its answer.
        for d in docs:
            original = d.metadata.get("original_text")
            if original:
                d.page_content = original
            
        try:
            answer = self._generator.invoke({
                "question": query,
                "documents": docs,
            })
        except Exception:
            log.exception("Generator failed for query=%r", query)
            return None

        if not answer:
            return None

        # If the generator declined to answer, treat that as a clean miss.
        refusal_markers = (
            "don't contain",
            "do not contain",
            "no information",
            "cannot answer",
            "isn't enough information",
            "not enough information",
        )
        if any(m in answer.lower() for m in refusal_markers):
            log.info("Generator declined for query=%r; treating as no-answer", query)
            return None

        sources = collect_citations(docs)
        if sources:
            return f"{answer}\n\nSources: {', '.join(sources)}"
        return answer