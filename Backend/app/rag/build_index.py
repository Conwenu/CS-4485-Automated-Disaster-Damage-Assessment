"""One-time PDF ingestion.

Loads PDFs from data/knowledge_base/, splits them into chunks, embeds them
into a Chroma vector store, and saves the raw chunks to JSON for the BM25
retriever to pick up at runtime.

Run:
    python -m app.rag.build_index
"""
import json
import logging
import shutil
from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


REPO_ROOT = Path(__file__).resolve().parents[2]
PDF_DIR = REPO_ROOT / "data" / "knowledge_base"
INDEX_DIR = REPO_ROOT / "data" / "chroma_index"
BM25_CHUNKS_PATH = REPO_ROOT / "data" / "bm25_chunks.json"
COLLECTION_NAME = "disaster_damage_kb"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
MIN_CHUNK_CHARS = 120


def load_pdfs(pdf_dir: Path) -> List[Document]:
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(
            f"No PDFs found in {pdf_dir}. Place PDFs there first."
        )

    documents: List[Document] = []
    for pdf_path in pdfs:
        log.info("Loading %s", pdf_path.name)
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        for p in pages:
            p.metadata["source"] = pdf_path.name
        documents.extend(pages)
        log.info("  -> %d pages", len(pages))

    return documents


def chunk_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    return [c for c in chunks if len(c.page_content.strip()) >= MIN_CHUNK_CHARS]


def save_chunks_for_bm25(chunks: List[Document], path: Path) -> None:
    """Persist chunks to JSON so BM25 can rebuild its index at runtime.

    BM25 doesn't need embeddings — only text + metadata. Saving as JSON
    keeps the runtime retriever fast (no PDF re-parsing) and decouples
    BM25 tuning from the vector store.
    """
    payload = [
        {"page_content": d.page_content, "metadata": dict(d.metadata)}
        for d in chunks
    ]
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    log.info("Saved %d chunks for BM25 to %s", len(chunks), path)


def build_index() -> None:
    if INDEX_DIR.exists():
        log.info("Removing existing Chroma index at %s", INDEX_DIR)
        shutil.rmtree(INDEX_DIR)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    documents = load_pdfs(PDF_DIR)
    log.info("Loaded %d raw pages total", len(documents))

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