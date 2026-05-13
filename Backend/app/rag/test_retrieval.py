"""Quick check after building the index. Run:
python -m app.rag.test_retrieval
"""

from app.rag.retriever import KnowledgeRetriever

QUERIES = [
    # Seed-fact territory — should be instant
    "What is major damage?",
    "What does destroyed mean?",
    "Explain the Joint Damage Scale",
    # RAG territory — should hit Chroma, get graded, then generated
    "How does FEMA decide if a county qualifies for assistance?",
    "What metric does xBD use to score classification performance?",
    "How are damage labels collected for xBD?",
    "How much did the damage cost?",
    # Should miss everything cleanly
    "What's the weather in Tokyo?",
]


def diagnose(query: str) -> None:
    from app.rag.retriever import KnowledgeRetriever

    r = KnowledgeRetriever.get()

    print(f"\n{'=' * 70}\nDIAGNOSE: {query}\n{'=' * 70}")

    candidates = r._search(query)
    print(f"\nVector retrieval returned {len(candidates)} candidates:\n")
    for i, d in enumerate(candidates, 1):
        src = d.metadata.get("source", "?")
        page = d.metadata.get("page", "?")
        preview = d.page_content.strip().replace("\n", " ")[:200]
        print(f"  [{i}] ({src}, p.{page}) {preview}...\n")

    # Reranker scoring (replaces grader verdicts)
    print("Reranker scores:\n")
    if hasattr(r, "_reranker") and r._reranker is not None:
        pairs = [(query, d.page_content) for d in candidates]
        scores = r._reranker.predict(pairs)
        threshold = getattr(r, "RERANKER_THRESHOLD", 0.0)
        # Import the module-level constant
        from app.rag import retriever as ret_module

        threshold = ret_module.RERANKER_THRESHOLD
        for i, (score, d) in enumerate(zip(scores, candidates), 1):
            mark = "✓ KEEP" if score > threshold else "✗ DROP"
            src = d.metadata.get("source", "?")
            print(f"  [{i}] {mark} (score={score:.3f}) — {src}")
    else:
        print("  (no reranker loaded)")


def main() -> None:
    r = KnowledgeRetriever.get()
    for q in QUERIES:
        print("=" * 70)
        print(f"Q: {q}")
        print("-" * 70)
        result = r.retrieve(q)
        print(result if result else "(no answer)")
        print()

    # diagnose("What metric does xBD use to score classification performance?")
    # diagnose("How are damage labels collected for xBD?")

    # diagnose("How much damage did the rosa wildfires cost?")
    # diagnose("How many homes were destroyed in the fire?")

    diagnose("How much did the damage cost?")
    diagnose("What was the cost of the Tubbs Fire?")
    diagnose("Tubbs Fire economic loss billion")


if __name__ == "__main__":
    main()
