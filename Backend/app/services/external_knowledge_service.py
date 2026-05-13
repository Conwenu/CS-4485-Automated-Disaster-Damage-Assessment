"""External knowledge service.

Thin wrapper around the LangChain RAG retriever. The retriever:
  1. Hybrid retrieval (Chroma MMR + BM25, fused via RRF) over xBD + FEMA
     + Tubbs Fire PDFs.
  2. Cross-encoder reranker filters candidates by relevance.
  3. Generates a grounded answer with citations.

Build the index first with:
    python -m app.rag.build_index
"""

import logging
from typing import Optional, AsyncGenerator

from app.rag.retriever import KnowledgeRetriever

log = logging.getLogger(__name__)


class ExternalKnowledgeService:
    def __init__(self) -> None:
        self._retriever = KnowledgeRetriever.get()

    def get_generator_llm(self):
        return self._retriever.get_generator_llm()

    def retrieve(self, query: str) -> Optional[str]:
        if not query or not query.strip():
            return None
        try:
            result = self._retriever.retrieve(query)
            print(f"retrieve({query}) -> {repr(result)[:100] if result else 'None'}")
            return result
        except Exception:
            log.exception("External knowledge retrieval failed for query=%r", query)
            return None

    async def retrieve_stream(self, query: str) -> AsyncGenerator[str, None]:
        """Stream tokens from the RAG generator."""
        if not query or not query.strip():
            return
        try:
            async for token in self._retriever.retrieve_stream(query):
                yield token
        except Exception:
            log.exception("Streaming external knowledge failed for query=%r", query)
