# scratchpad — run interactively or save as test_index.py
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from pathlib import Path

INDEX_DIR = Path("data/chroma_index")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
store = Chroma(
    collection_name="disaster_damage_kb",
    embedding_function=embeddings,
    persist_directory=str(INDEX_DIR),
)

# Pull a handful of docs and inspect
docs = store.similarity_search("xBD evaluation metric", k=3)
for i, d in enumerate(docs, 1):
    print(f"\n--- Doc {i} ({d.metadata.get('source')}, p.{d.metadata.get('page')}) ---")
    print(f"CONTEXT TAG: {d.metadata.get('context_tag', '(none)')}")
    print(f"PAGE CONTENT (first 300 chars):")
    print(d.page_content[:300])