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
    # Should miss everything cleanly
    "What's the weather in Tokyo?",
]


def diagnose(query: str) -> None:
    """Print exactly what retrieval+grading is doing for a query."""
    from app.rag.retriever import KnowledgeRetriever
    r = KnowledgeRetriever.get()

    print(f"\n{'=' * 70}\nDIAGNOSE: {query}\n{'=' * 70}")

    # Step 1: raw vector hits
    candidates = r._search(query)
    print(f"\nVector retrieval returned {len(candidates)} candidates:\n")
    for i, d in enumerate(candidates, 1):
        src = d.metadata.get("source", "?")
        page = d.metadata.get("page", "?")
        preview = d.page_content.strip().replace("\n", " ")[:200]
        print(f"  [{i}] ({src}, p.{page}) {preview}...\n")

    # Step 2: what the grader thinks
    print("Grader verdicts:\n")
    for i, d in enumerate(candidates, 1):
        try:
            verdict = r._grader.invoke({
                "question": query,
                "document": d.page_content,
            })
            mark = "✓ KEEP" if verdict.is_relevant else "✗ DROP"
            print(f"  [{i}] {mark} — {verdict.reasoning}")
        except Exception as e:
            print(f"  [{i}] ERROR — {e}")


def main() -> None:
    r = KnowledgeRetriever.get()
    for q in QUERIES:
        print("=" * 70)
        print(f"Q: {q}")
        print("-" * 70)
        result = r.retrieve(q)
        print(result if result else "(no answer)")
        print()
        
    diagnose("What metric does xBD use to score classification performance?")
    diagnose("How are damage labels collected for xBD?")


if __name__ == "__main__":
    main()