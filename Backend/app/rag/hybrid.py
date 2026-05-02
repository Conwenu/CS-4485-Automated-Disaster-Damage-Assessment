"""Hybrid retrieval: BM25 (keyword) + Chroma (semantic), merged via RRF.

Reciprocal Rank Fusion (Cormack et al., 2009) combines rankings from
multiple retrievers without needing comparable scores. Each retriever's
top-N produces ranked positions; each doc gets a fused score of
    score = sum( 1 / (k + rank_i) ) for i in retrievers
where k is a smoothing constant (k=60 is the standard default).

This catches two distinct failure modes:
  - BM25 wins on specific terminology, proper nouns, numbers, acronyms
    (e.g., "Tubbs Fire", "weighted F1", "Coffey Park", "PDA")
  - Vector search wins on paraphrasing and conceptual queries
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

log = logging.getLogger(__name__)


# Standard RRF constant. Higher k = more uniform weighting (later ranks
# matter more); lower k = top-rank bias. 60 is the canonical value.
RRF_K = 60


def load_bm25(chunks_json: Path, k: int = 6) -> BM25Retriever:
    """Build a BM25 retriever from the JSON chunk file produced at index time."""
    if not chunks_json.exists():
        raise FileNotFoundError(
            f"No BM25 corpus at {chunks_json}. "
            f"Run `python -m app.rag.build_index` first."
        )

    payload = json.loads(chunks_json.read_text(encoding="utf-8"))
    docs = [
        Document(page_content=item["page_content"], metadata=item["metadata"])
        for item in payload
    ]

    retriever = BM25Retriever.from_documents(docs)
    retriever.k = k  # number of results to return
    return retriever


def reciprocal_rank_fusion(
    rankings: List[List[Document]],
    k: int = RRF_K,
) -> List[Document]:
    """Merge ranked lists from multiple retrievers via RRF.

    Documents are deduplicated by their (source, page, original_text) tuple
    since LangChain doesn't guarantee stable IDs across retrievers.
    """
    scores: Dict[Tuple, float] = {}
    docs_by_key: Dict[Tuple, Document] = {}

    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            key = _doc_key(doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            # Keep the first seen Document instance for this key
            docs_by_key.setdefault(key, doc)

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [docs_by_key[key] for key, _ in fused]


def _doc_key(doc: Document) -> Tuple:
    """Stable identity for a chunk across retrievers."""
    src = doc.metadata.get("source", "")
    page = doc.metadata.get("page", -1)
    # First 200 chars of content disambiguates same-page chunks
    snippet = (doc.page_content or "")[:200]
    return (src, page, snippet)