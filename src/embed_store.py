"""
Step 2: Turn text chunks into embeddings (vectors) and store them in
a FAISS index so we can search by meaning later, not just keywords.

This file depends on ingest.py - it expects a list of chunked
Document objects as input.
"""

import sys
import os

_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from langchain_community.vectorstores import FAISS
from ingest import load_data_folder, chunk_documents


# Supported embedding options:
# 1. "sentence-transformers/all-MiniLM-L6-v2" (Fast local)
# 2. "sentence-transformers/all-mpnet-base-v2" (Better local)
# 3. "text-embedding-3-small" (OpenAI)
# 4. "text-embedding-3-large" (OpenAI)

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_SAVE_PATH = "faiss_index"
CHROMA_SAVE_PATH = "chroma_index"
CHROMA_COLLECTION = "smart_research"


def get_index_path(model_name, store_type="FAISS"):
    """
    Returns a unique folder path for the vector index based on the embedding model name.
    This prevents dimension mismatch errors when switching between models.
    """
    safe_name = model_name.replace("/", "_").replace("-", "_").replace(".", "_")
    if store_type == "FAISS":
        return f"faiss_index_{safe_name}"
    else:
        return f"chroma_index_{safe_name}"


def get_embedding_model(model_name=EMBEDDING_MODEL_NAME, provider="Ollama", api_key=None):
    """
    Loads the sentence-transformers model (locally) or OpenAI Embeddings (cloud).
    """
    if provider == "OpenAI" or model_name.startswith("text-embedding-"):
        from langchain_openai import OpenAIEmbeddings
        # Default to small if not valid
        if not model_name.startswith("text-embedding-"):
            model_name = "text-embedding-3-small"
        print(f"Loading OpenAI embedding model: {model_name}")
        return OpenAIEmbeddings(model=model_name, openai_api_key=api_key)
    else:
        print(f"Loading local HuggingFace embedding model: {model_name}")
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=model_name)


def build_vector_store(chunks, embedding_model):
    """
    Takes chunked Documents + an embedding model, embeds every chunk,
    and builds a FAISS index in memory.
    """
    if not chunks:
        raise ValueError("No chunks provided - did ingest.py actually find documents?")

    print(f"Embedding {len(chunks)} chunks... (this can take a bit on CPU)")
    vector_store = FAISS.from_documents(chunks, embedding_model)
    return vector_store


def save_vector_store(vector_store, path=INDEX_SAVE_PATH):
    """
    Persists the FAISS index to disk.
    """
    vector_store.save_local(path)
    print(f"Saved FAISS index to ./{path}")


def load_vector_store(embedding_model, path=INDEX_SAVE_PATH):
    """
    Loads a previously saved FAISS index back from disk.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No saved index found at ./{path} - run build first."
        )
    return FAISS.load_local(
        path, embedding_model, allow_dangerous_deserialization=True
    )


def build_chroma_store(chunks, embedding_model, path=CHROMA_SAVE_PATH):
    """
    Embeds chunks and stores them in a ChromaDB collection on disk.
    """
    if not chunks:
        raise ValueError("No chunks provided - did ingest.py actually find documents?")

    print(f"Embedding {len(chunks)} chunks into ChromaDB...")
    try:
        from langchain_community.vectorstores import Chroma
    except ImportError:
        raise ImportError("ChromaDB not available. Run: pip install chromadb")
    vector_store = Chroma.from_documents(
        chunks,
        embedding_model,
        persist_directory=path,
        collection_name=CHROMA_COLLECTION,
    )
    print(f"Saved ChromaDB index to ./{path}")
    return vector_store


def load_chroma_store(embedding_model, path=CHROMA_SAVE_PATH):
    """
    Loads an existing ChromaDB collection from disk.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No ChromaDB index found at ./{path} — build it first via the Documents tab."
        )
    try:
        from langchain_community.vectorstores import Chroma
    except ImportError:
        raise ImportError("ChromaDB not available. Run: pip install chromadb")
    return Chroma(
        persist_directory=path,
        embedding_function=embedding_model,
        collection_name=CHROMA_COLLECTION,
    )



if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    embedding_model = get_embedding_model()

    if os.path.exists(INDEX_SAVE_PATH):
        print("Existing index found, loading it instead of rebuilding...")
        store = load_vector_store(embedding_model)
    else:
        print("No existing index. Building from scratch...")
        raw_docs = load_data_folder("data")
        chunks = chunk_documents(raw_docs)
        store = build_vector_store(chunks, embedding_model)
        save_vector_store(store)

    # Quick manual sanity test - search for something and see what
    # comes back, just to confirm the whole pipeline actually works.
    test_query = "What vector databases does this project use?"
    print(f"\nTest query: '{test_query}'")
    results = store.similarity_search(test_query, k=2)

    for i, r in enumerate(results, start=1):
        print(f"\n[Result {i}]")
        print(r.page_content[:200], "...")
        print(f"metadata -> {r.metadata}")