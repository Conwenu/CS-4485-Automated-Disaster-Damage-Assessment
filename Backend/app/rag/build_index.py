"""One-time ingestion for the RAG knowledge base.

Loads PDFs, markdown, and plain-text files from data/knowledge_base/,
splits them into chunks, embeds them into a Chroma vector store, and
saves the raw chunks to JSON for the BM25 retriever to pick up at runtime.

Run:
    python -m app.rag.build_index
"""

import json
import logging
import re
import shutil
from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


REPO_ROOT = Path(__file__).resolve().parents[2]
KB_DIR = REPO_ROOT / "data" / "knowledge_base"
INDEX_DIR = REPO_ROOT / "data" / "chroma_index"
BM25_CHUNKS_PATH = REPO_ROOT / "data" / "bm25_chunks.json"
COLLECTION_NAME = "disaster_damage_kb"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
MIN_CHUNK_CHARS = 120

# File extensions to include in the corpus
PDF_EXTS = ("*.pdf",)
TEXT_EXTS = ("*.md", "*.txt")


def load_documents(kb_dir: Path) -> List[Document]:
    """Load all supported files from the knowledge base directory."""
    if not kb_dir.exists():
        raise FileNotFoundError(f"Knowledge base directory not found: {kb_dir}")

    documents: List[Document] = []

    # PDFs
    pdf_paths = sorted(p for ext in PDF_EXTS for p in kb_dir.glob(ext))
    for pdf_path in pdf_paths:
        log.info("Loading %s", pdf_path.name)
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        for p in pages:
            p.metadata["source"] = pdf_path.name
        documents.extend(pages)
        log.info("  -> %d pages", len(pages))

    # Markdown and plain text
    text_paths = sorted(p for ext in TEXT_EXTS for p in kb_dir.glob(ext))
    for text_path in text_paths:
        log.info("Loading %s", text_path.name)
        loader = TextLoader(str(text_path), encoding="utf-8")
        docs = loader.load()
        for d in docs:
            d.metadata["source"] = text_path.name
        documents.extend(docs)
        log.info("  -> %d document(s)", len(docs))

    if not documents:
        raise FileNotFoundError(
            f"No supported files found in {kb_dir}. "
            f"Add PDFs, .md, or .txt files first."
        )

    return documents


def chunk_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    filtered = []
    for c in chunks:
        text = c.page_content.strip()
        if len(text) < MIN_CHUNK_CHARS:
            continue
        if _is_citation_noise(text):
            continue
        filtered.append(c)

    return filtered


def _is_citation_noise(text: str) -> bool:
    """True for chunks that are mostly citations, URLs, or references.

    Wikipedia/academic PDFs produce reference-list pages that pollute
    retrieval — they contain the right keywords but no useful prose.
    """
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if not lines:
        return True

    url_pattern = re.compile(r"https?://|www\.", re.IGNORECASE)
    issn_pattern = re.compile(r"ISSN|ISBN|doi:|arxiv:", re.IGNORECASE)
    citation_pattern = re.compile(r"^\d+\.\s+\w+")  # "39. Lorenz, Julie..."

    url_lines = sum(1 for line in lines if url_pattern.search(line))
    issn_lines = sum(1 for line in lines if issn_pattern.search(line))
    citation_lines = sum(1 for line in lines if citation_pattern.match(line))

    noise_ratio = (url_lines + issn_lines + citation_lines) / len(lines)
    return noise_ratio > 0.4


def save_chunks_for_bm25(chunks: List[Document], path: Path) -> None:
    """Persist chunks to JSON so BM25 can rebuild its index at runtime.

    BM25 doesn't need embeddings — only text + metadata. Saving as JSON
    keeps the runtime retriever fast (no re-parsing) and decouples
    BM25 tuning from the vector store.
    """
    payload = [
        {"page_content": d.page_content, "metadata": dict(d.metadata)} for d in chunks
    ]
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    log.info("Saved %d chunks for BM25 to %s", len(chunks), path)


def build_index() -> None:
    if INDEX_DIR.exists():
        log.info("Removing existing Chroma index at %s", INDEX_DIR)
        shutil.rmtree(INDEX_DIR)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    documents = load_documents(KB_DIR)
    log.info("Loaded %d raw documents/pages total", len(documents))

    chunks = chunk_documents(documents)
    log.info("Produced %d chunks after splitting + filtering", len(chunks))

    log.info("Loading embedding model (%s)", EMBEDDING_MODEL)
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    log.info("Indexing %d chunks into Chroma...", len(chunks))
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(INDEX_DIR),
    )
    log.info("Chroma index built at %s", INDEX_DIR)

    save_chunks_for_bm25(chunks, BM25_CHUNKS_PATH)


if __name__ == "__main__":
    build_index()
