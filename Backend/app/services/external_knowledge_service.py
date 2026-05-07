"""External knowledge service.

Thin wrapper around the LangChain RAG retriever. The retriever:
  1. Checks curated seed facts.
  2. Falls back to Chroma vector retrieval over xBD + FEMA PDFs.
  3. Filters retrieved chunks via an LLM grader.
  4. Generates a grounded answer with citations.

Build the index first with:
    python -m app.rag.build_index
"""

import logging
from typing import Optional

from app.rag.retriever import KnowledgeRetriever

log = logging.getLogger(__name__)


class ExternalKnowledgeService:
    def __init__(self) -> None:
        self._retriever = KnowledgeRetriever.get()

    def retrieve(self, query: str) -> Optional[str]:
        if not query or not query.strip():
            return None
        try:
            return self._retriever.retrieve(query)
        except Exception:
            log.exception("External knowledge retrieval failed for query=%r", query)
            return None
