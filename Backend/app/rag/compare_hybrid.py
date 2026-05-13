"""Compare vector-only, BM25-only, and hybrid retrieval end-to-end.

Usage:
    python -m app.rag.compare_hybrid
"""

import time
from pathlib import Path
from typing import List, Tuple

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from app.rag.chains import build_generator_chain
from app.rag.hybrid import load_bm25, reciprocal_rank_fusion

REPO_ROOT = Path(__file__).resolve().parents[2]
INDEX_DIR = REPO_ROOT / "data" / "chroma_index"
BM25_PATH = REPO_ROOT / "data" / "bm25_chunks.json"

CALL_DELAY = 15


# Mix of queries: some favor BM25 (specific terms), some favor vector
# (paraphrased), some neutral.
TEST_CASES: List[Tuple[str, List[str]]] = [
    # Specific terminology — BM25 should help
    (
        "What metric does xBD use to score classification performance?",
        ["F1", "weighted f1"],
    ),
    (
        "How many structures did the Tubbs Fire destroy?",
        ["5,643", "5643", "5,636", "5636", "4,658", "4658", "thousand"],
    ),
    (
        "What was happening in Coffey Park during the fire?",
        ["coffey", "leveled", "destroyed", "neighborhood"],
    ),
    ("What is a PDA in FEMA terminology?", ["preliminary damage assessment", "pda"]),
    # Paraphrased — vector should help
    (
        "How does FEMA decide on disaster aid eligibility?",
        ["assistance", "eligibility", "individual", "public"],
    ),
    (
        "What did annotators do to label damage in xBD?",
        ["analyst", "annotation", "joint damage scale", "iteration", "label"],
    ),
    # Neutral
    (
        "What classes does the xBD damage scale use?",
        ["no-damage", "minor", "major", "destroyed"],
    ),
]


def _embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def vector_only(store: Chroma, query: str) -> List[Document]:
    return store.max_marginal_relevance_search(query, k=6, fetch_k=25, lambda_mult=0.6)


def bm25_only(retriever, query: str) -> List[Document]:
    return retriever.invoke(query)


def hybrid(store: Chroma, bm25, query: str) -> List[Document]:
    v = store.max_marginal_relevance_search(query, k=8, fetch_k=25, lambda_mult=0.6)
    b = bm25.invoke(query)
    return reciprocal_rank_fusion([v, b])[:6]


def run_pipeline(docs: List[Document], query: str, reranker, generator) -> str:
    if not docs:
        return ""

    # Rerank instead of LLM-grade
    if reranker is not None:
        pairs = [(query, d.page_content) for d in docs]
        scores = reranker.predict(pairs)
        scored = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        kept = [doc for score, doc in scored if score > 0.0]
        if not kept:
            kept = [doc for _, doc in scored[:3]]
    else:
        kept = docs[:4]

    for d in kept:
        original = d.metadata.get("original_text")
        if original:
            d.page_content = original

    try:
        ans = generator.invoke({"question": query, "documents": kept[:4]})
    except Exception:
        return ""

    return (ans or "").strip()


def evaluate(label: str, retrieve_fn, reranker, generator) -> int:
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
    hits = 0
    for query, keywords in TEST_CASES:
        docs = retrieve_fn(query)
        answer = run_pipeline(docs, query, reranker, generator)
        hit = any(kw.lower() in answer.lower() for kw in keywords)
        marker = "✓" if hit else "✗"
        print(f"\n{marker} {query}")
        print(f"  expected any of: {keywords}")
        print(f"  answer: {answer[:200]}")
        if hit:
            hits += 1
        time.sleep(CALL_DELAY)
    print(f"\n{label} total: {hits}/{len(TEST_CASES)}")
    return hits


def main() -> None:
    from sentence_transformers import CrossEncoder

    store = Chroma(
        collection_name="disaster_damage_kb",
        embedding_function=_embeddings(),
        persist_directory=str(INDEX_DIR),
    )
    bm25 = load_bm25(BM25_PATH, k=8)
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    generator = build_generator_chain()

    v = evaluate("VECTOR ONLY", lambda q: vector_only(store, q), reranker, generator)
    b = evaluate("BM25 ONLY", lambda q: bm25_only(bm25, q), reranker, generator)
    h = evaluate("HYBRID (RRF)", lambda q: hybrid(store, bm25, q), reranker, generator)

    print(f"\n{'=' * 70}\nSummary: vector={v}, bm25={b}, hybrid={h}")


if __name__ == "__main__":
    main()
