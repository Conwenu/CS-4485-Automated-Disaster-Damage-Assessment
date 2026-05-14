"""Retrieval debugging tool.

Shows what each stage of the RAG pipeline produces for a given query:
  1. Raw vector search results (top-K from Chroma MMR)
  2. Raw BM25 results (top-K from keyword search)
  2.5. Per-source dedupe (prevents one long document from dominating)
  3. RRF-fused results (merged ranking)
  4. Reranker scores (cross-encoder scoring on fused candidates)
  5. Final candidates (those above the threshold)

Run:
    python -m app.rag.debug_retrieval "What is major damage?"
    python -m app.rag.debug_retrieval "How much did the Tubbs Fire cost?"
"""

import logging
import sys
from typing import List, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

from app.rag.hybrid import load_bm25, reciprocal_rank_fusion
from app.rag.retriever import (
    BM25_CHUNKS_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    HYBRID_FINAL_K,
    HYBRID_K_BM25,
    HYBRID_K_VECTOR,
    INDEX_DIR,
    RERANKER_MODEL,
    RERANKER_THRESHOLD,
    RETRIEVAL_FETCH_K,
    RETRIEVAL_LAMBDA,
)

# Quiet down the noisy libraries so the output stays readable
logging.basicConfig(level=logging.WARNING)
for noisy in ("httpx", "huggingface_hub", "sentence_transformers", "chromadb"):
    logging.getLogger(noisy).setLevel(logging.ERROR)


# ── Tunable retrieval parameters for this debug session ───────────────
# These override the imported values so you can experiment without
# touching the production retriever. Mirror your production settings here
# once you've found values that work.
MAX_PER_SOURCE = 2  # dedupe cap per document
VECTOR_WEIGHT = 1.3  # RRF weight for vector hits (was 1.0)
BM25_WEIGHT = 1.0  # RRF weight for BM25 hits (was 1.3)

# Output settings
PREVIEW_CHARS = 200
SEP = "=" * 80
SUBSEP = "-" * 80


def color(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m"


def header(title: str) -> None:
    print(f"\n{SEP}")
    print(color(f"  {title}", "1;36"))
    print(SEP)


def format_doc(rank: int, doc: Document, score: float = None) -> str:
    source = doc.metadata.get("source", "?")
    page = doc.metadata.get("page", None)
    page_str = f" p.{page}" if page is not None else ""
    preview = doc.page_content.strip().replace("\n", " ")[:PREVIEW_CHARS]

    score_str = f"  score={score:+.4f}" if score is not None else ""
    rank_str = color(f"[{rank:2d}]", "1;33")
    source_str = color(f"{source}{page_str}", "0;32")

    return f"  {rank_str}{score_str}  {source_str}\n       {preview}..."


def dedupe_by_source(docs: List[Document], max_per_source: int) -> List[Document]:
    """Keep at most N chunks from any single source, preserving order."""
    counts: dict = {}
    out: List[Document] = []
    for d in docs:
        src = d.metadata.get("source", "?")
        if counts.get(src, 0) < max_per_source:
            out.append(d)
            counts[src] = counts.get(src, 0) + 1
    return out


def summarize_sources(docs: List[Document]) -> str:
    """Return a one-line summary of source distribution."""
    from collections import Counter

    counts = Counter(d.metadata.get("source", "?") for d in docs)
    return ", ".join(f"{src}: {n}" for src, n in counts.most_common())


def debug_query(query: str) -> None:
    print(f"\n{color('QUERY:', '1;35')} {query}")
    print(color(f"Settings:", "1;37"))
    print(f"  max_per_source = {MAX_PER_SOURCE}")
    print(f"  rrf weights    = vector={VECTOR_WEIGHT}, bm25={BM25_WEIGHT}")
    print(f"  fused top-k    = {HYBRID_FINAL_K}")
    print(f"  reranker thresh= {RERANKER_THRESHOLD}")

    # ── Load all pipeline components ───────────────────────────────
    print("\nLoading pipeline components (this takes a few seconds)...")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(INDEX_DIR),
    )
    bm25 = (
        load_bm25(BM25_CHUNKS_PATH, k=HYBRID_K_BM25)
        if BM25_CHUNKS_PATH.exists()
        else None
    )
    reranker = CrossEncoder(RERANKER_MODEL)

    # ── Stage 1: Vector search ────────────────────────────────────
    header("STAGE 1: Vector search (Chroma MMR)")
    print(
        f"k={HYBRID_K_VECTOR}, fetch_k={RETRIEVAL_FETCH_K}, lambda={RETRIEVAL_LAMBDA}\n"
    )

    vector_hits: List[Document] = vector_store.max_marginal_relevance_search(
        query,
        k=HYBRID_K_VECTOR,
        fetch_k=RETRIEVAL_FETCH_K,
        lambda_mult=RETRIEVAL_LAMBDA,
    )
    if not vector_hits:
        print("  (nothing returned)")
    for i, doc in enumerate(vector_hits, start=1):
        print(format_doc(i, doc))
    print(f"\n  Source distribution: {summarize_sources(vector_hits)}")

    # ── Stage 2: BM25 search ──────────────────────────────────────
    header("STAGE 2: BM25 keyword search")
    print(f"k={HYBRID_K_BM25}\n")

    bm25_hits: List[Document] = []
    if bm25 is not None:
        bm25_hits = bm25.invoke(query)
    if not bm25_hits:
        print("  (BM25 unavailable or returned nothing)")
    for i, doc in enumerate(bm25_hits, start=1):
        print(format_doc(i, doc))
    if bm25_hits:
        print(f"\n  Source distribution: {summarize_sources(bm25_hits)}")

    # ── Stage 2.5: Per-source dedupe ──────────────────────────────
    header(f"STAGE 2.5: Per-source dedupe (max {MAX_PER_SOURCE} per source)")

    vector_dedup = dedupe_by_source(vector_hits, max_per_source=MAX_PER_SOURCE)
    bm25_dedup = dedupe_by_source(bm25_hits, max_per_source=MAX_PER_SOURCE)

    print(f"  Vector: {len(vector_hits)} → {len(vector_dedup)}")
    print(f"  BM25:   {len(bm25_hits)} → {len(bm25_dedup)}")

    print(color("\n  Deduped vector hits:", "1;37"))
    for i, doc in enumerate(vector_dedup, start=1):
        print(format_doc(i, doc))

    print(color("\n  Deduped BM25 hits:", "1;37"))
    for i, doc in enumerate(bm25_dedup, start=1):
        print(format_doc(i, doc))

    # ── Stage 3: RRF fusion ───────────────────────────────────────
    header("STAGE 3: Reciprocal Rank Fusion")
    print(
        f"weights=[vector={VECTOR_WEIGHT}, bm25={BM25_WEIGHT}], returning top {HYBRID_FINAL_K}\n"
    )

    if vector_dedup and bm25_dedup:
        fused = reciprocal_rank_fusion(
            [vector_dedup, bm25_dedup],
            weights=[VECTOR_WEIGHT, BM25_WEIGHT],
        )
    elif vector_dedup:
        fused = vector_dedup
    else:
        fused = bm25_dedup

    fused = fused[:HYBRID_FINAL_K]
    if not fused:
        print("  (nothing to fuse)")
    for i, doc in enumerate(fused, start=1):
        print(format_doc(i, doc))
    if fused:
        print(f"\n  Source distribution: {summarize_sources(fused)}")

    # ── Stage 4: Cross-encoder reranker ───────────────────────────
    header("STAGE 4: Cross-encoder reranker")
    print(f"model={RERANKER_MODEL}, threshold={RERANKER_THRESHOLD}\n")

    if not fused:
        print("  (nothing to rerank)")
        return

    pairs = [(query, d.page_content) for d in fused]
    scores = reranker.predict(pairs)

    scored: List[Tuple[float, Document]] = sorted(
        zip(scores, fused), key=lambda x: x[0], reverse=True
    )
    for i, (score, doc) in enumerate(scored, start=1):
        marker = (
            color("  KEEP", "1;32")
            if score > RERANKER_THRESHOLD
            else color("  DROP", "1;31")
        )
        print(format_doc(i, doc, score=float(score)) + marker)

    # ── Stage 5: Final candidates sent to generator ──────────────
    header("STAGE 5: Final candidates → generator")

    kept = [(s, d) for s, d in scored if s > RERANKER_THRESHOLD]
    if not kept:
        print("  (everything was dropped — generator would refuse)\n")
        return

    for i, (score, doc) in enumerate(kept, start=1):
        print(format_doc(i, doc, score=float(score)))

    print(f"\n  Total kept: {len(kept)}")
    print(f"  Source distribution: {summarize_sources([d for _, d in kept])}")


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python -m app.rag.debug_retrieval "<query>"')
        print("\nExample queries:")
        print('  python -m app.rag.debug_retrieval "What is major damage?"')
        print('  python -m app.rag.debug_retrieval "How much did the Tubbs Fire cost?"')
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    debug_query(query)


if __name__ == "__main__":
    main()
