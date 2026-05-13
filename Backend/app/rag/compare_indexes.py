"""Compare contextual vs plain retrieval end-to-end.

Runs each query through the full retrieval pipeline (MMR + grader +
generator) using both indexes. Records whether the final answer is correct,
based on substring matching against expected keywords.

Usage:
    python -m app.rag.compare_indexes
"""

import time
from pathlib import Path
from typing import List, Tuple

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from app.rag.chains import build_grader_chain, build_generator_chain


REPO_ROOT = Path(__file__).resolve().parents[2]
PLAIN_DIR = REPO_ROOT / "data" / "chroma_index_plain"
CTX_DIR = REPO_ROOT / "data" / "chroma_index"

# Delay between RAG calls to survive Gemini free-tier (5 RPM).
# Each query uses ~5-7 LLM calls (grader per chunk + generator), so even
# one query per minute is tight. Bump this if you hit rate limits.
CALL_DELAY = 15


# (query, list_of_expected_keywords). A query passes if ANY keyword appears.
# Keywords should be specific to the correct answer, not just the topic.
TEST_CASES: List[Tuple[str, List[str]]] = [
    (
        "What metric does xBD use to score classification performance?",
        ["F1", "weighted f1"],
    ),
    (
        "How are damage labels annotated for xBD?",
        ["analyst", "annotation", "joint damage scale", "iteration"],
    ),
    (
        "What does FEMA look for during a Preliminary Damage Assessment?",
        ["destroyed", "major", "minor", "affected", "damage"],
    ),
    (
        "How many disasters are included in the xBD dataset?",
        ["19", "nineteen"],
    ),
    (
        "What classes does the xBD damage scale use?",
        ["no-damage", "minor", "major", "destroyed"],
    ),
]


def load_store(persist_dir: Path) -> Chroma:
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return Chroma(
        collection_name="disaster_damage_kb",
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )


def run_pipeline(
    store: Chroma, query: str, grader, generator
) -> Tuple[str, List[Document]]:
    """Mirror of KnowledgeRetriever.retrieve but without seed facts."""
    candidates = store.max_marginal_relevance_search(
        query, k=6, fetch_k=25, lambda_mult=0.6
    )
    if not candidates:
        return "", []

    kept: List[Document] = []
    for d in candidates:
        try:
            verdict = grader.invoke({"question": query, "document": d.page_content})
            if verdict.is_relevant:
                kept.append(d)
        except Exception:
            kept.append(d)

    if not kept:
        kept = candidates[:4]

    # Prefer original_text for generation (no prepended tag)
    for d in kept:
        original = d.metadata.get("original_text")
        if original:
            d.page_content = original

    try:
        answer = generator.invoke({"question": query, "documents": kept[:4]})
    except Exception:
        return "", kept

    return (answer or "").strip(), kept[:4]


def evaluate(store: Chroma, label: str) -> None:
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
    grader = build_grader_chain()
    generator = build_generator_chain()

    hits = 0
    for query, keywords in TEST_CASES:
        answer, docs = run_pipeline(store, query, grader, generator)
        hit = any(kw.lower() in answer.lower() for kw in keywords)
        marker = "✓ PASS" if hit else "✗ FAIL"
        source_names = sorted(set(d.metadata.get("source", "?") for d in docs))
        print(f"\n{marker} | {query}")
        print(f"  expected any of: {keywords}")
        print(f"  answer: {answer[:200]}")
        print(f"  sources: {source_names}")
        if hit:
            hits += 1
        time.sleep(CALL_DELAY)

    print(f"\n{label} total: {hits}/{len(TEST_CASES)}")


def main() -> None:
    if PLAIN_DIR.exists():
        evaluate(load_store(PLAIN_DIR), "PLAIN INDEX")
    if CTX_DIR.exists():
        evaluate(load_store(CTX_DIR), "CONTEXTUAL INDEX")


if __name__ == "__main__":
    main()
