# Smart Research Assistant

A fully local, offline RAG (Retrieval-Augmented Generation) pipeline built with LangChain, FAISS, sentence-transformers, and Ollama. Ask questions about your research documents and get grounded, cited answers — no cloud APIs, no cost.

---

## Architecture

```
data/ (PDFs & TXTs)
  └─→ ingest.py   → chunk_documents()
        └─→ embed_store.py  → FAISS index (faiss_index/)
              └─→ retrieve.py   → similarity / MMR search
                    └─→ generate.py  → Ollama llama3.2 answer
                          └─→ evaluate.py → RAGAs metrics
                                └─→ app.py (Streamlit UI)
```

| Component       | Library / Model                        |
|-----------------|----------------------------------------|
| Embeddings      | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector DB       | FAISS (local, no server)               |
| LLM             | Ollama `llama3.2` (local, offline)     |
| Evaluation      | RAGAs (faithfulness, relevancy, precision) |
| UI              | Streamlit                              |

---

## Prerequisites

1. **Python 3.10+**
2. **Ollama** installed and running — [https://ollama.com](https://ollama.com)
3. `llama3.2` model pulled:
   ```bash
   ollama pull llama3.2
   ollama serve   # starts automatically after install on most systems
   ```

---

## Setup

```bash
# 1. Clone / enter the project folder
cd smart-research-assistant

# 2. Create virtual environment (optional but recommended)
python -m venv venvproject
venvproject\Scripts\activate      # Windows
# source venvproject/bin/activate  # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Running the App

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Project Structure

```
smart-research-assistant/
├── app.py                 ← Streamlit UI (entry point)
├── requirements.txt
├── README.md
├── data/                  ← Put your PDFs/TXTs here
│   └── Smart Research Assisstant.pdf
├── faiss_index/           ← Auto-generated vector index
│   ├── index.faiss
│   └── index.pkl
└── src/
    ├── __init__.py
    ├── ingest.py          ← Stage 1: load & chunk documents
    ├── embed_store.py     ← Stage 2: embed chunks → FAISS
    ├── retrieve.py        ← Stage 3: similarity / MMR search
    ├── generate.py        ← Stage 4: Ollama-powered answers
    └── evaluate.py        ← Stage 5: RAGAs evaluation
```

---

## Features

- **💬 Chat tab** — multi-turn Q&A with source page citations
- **📁 Documents tab** — upload new PDFs/TXTs and auto-rebuild the FAISS index
- **📊 Evaluation tab** — run RAGAs metrics (faithfulness, answer relevancy, context precision) on demand
- **⚙️ Sidebar** — tune retrieval `k` and toggle MMR (Maximal Marginal Relevance)
- **100% local** — no API keys, no internet required after setup

---

## Running Individual Stages (CLI)

```bash
# From the src/ directory
cd src

python ingest.py          # test chunking
python embed_store.py     # build / load FAISS index
python retrieve.py        # interactive retrieval test
python generate.py        # interactive Q&A loop
python evaluate.py        # full RAGAs evaluation run
```

---

## Configuration

Edit constants at the top of each module:

| File            | Constant              | Default                      |
|-----------------|-----------------------|------------------------------|
| `ingest.py`     | `CHUNK_SIZE`          | `500`                        |
| `ingest.py`     | `CHUNK_OVERLAP`       | `50`                         |
| `embed_store.py`| `EMBEDDING_MODEL_NAME`| `all-MiniLM-L6-v2`          |
| `embed_store.py`| `INDEX_SAVE_PATH`     | `faiss_index`                |
| `generate.py`   | `OLLAMA_MODEL_NAME`   | `llama3.2`                   |
| `retrieve.py`   | `DEFAULT_K`           | `4`                          |
