"""
Step 1: Load raw docs from the data folder and break them into chunks.
This is the first stage of my RAG pipeline - nothing about embeddings
or FAISS happens here, just getting clean text chunks ready for the
next step.
"""

import os
import sys
import urllib.request
import hashlib
from urllib.parse import urlparse
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# Keeping these as constants up top so I can tweak them later without
# digging through function bodies. Started with 500/50, may experiment.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
SUPPORTED_EXTENSIONS = (".pdf", ".txt")


def load_single_file(filepath):
    """
    Reads one file (pdf or txt) and returns LangChain Document objects.
    Raises an error for anything else so I notice unsupported files
    instead of silently skipping them.
    """
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".pdf":
        loader = PyPDFLoader(filepath)
    elif ext == ".txt":
        loader = TextLoader(filepath, encoding="utf-8")
    else:
        raise ValueError(f"Don't know how to load this file type: {filepath}")

    return loader.load()


def scrape_url_to_file(url, data_dir="data"):
    """
    Scrapes text content from a web URL using BeautifulSoup and saves it as
    a text file in the data folder so it gets chunked and indexed.
    """
    try:
        req = urllib.request.Request(
            url,
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        )
        with urllib.request.urlopen(req, timeout=15) as response:
            html = response.read().decode('utf-8', errors='ignore')

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")

        # Decompose elements we don't want
        for s in soup(["script", "style", "nav", "footer", "header"]):
            s.decompose()

        title = soup.title.string.strip() if soup.title else "Web Page"
        # Get clean text
        lines = [line.strip() for line in soup.get_text(separator="\n").splitlines()]
        text_blocks = [l for l in lines if l]
        body_text = "\n".join(text_blocks)

        final_content = f"Title: {title}\nURL: {url}\n\n{body_text}"

        parsed_url = urlparse(url)
        domain = parsed_url.netloc.replace(":", "_").replace(".", "_")
        url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()[:8]
        filename = f"web_{domain}_{url_hash}.txt"

        os.makedirs(data_dir, exist_ok=True)
        filepath = os.path.join(data_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(final_content)

        print(f"Scraped and saved: {filename}")
        return filepath, filename
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        raise e


def load_data_folder(folder_path="data"):
    """
    Goes through every file in the data folder and loads the supported
    ones. Skips anything unsupported with a warning instead of crashing,
    since a stray .docx or .jpg in there shouldn't kill the whole run.
    """
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Can't find folder: {folder_path}")

    loaded_docs = []
    skipped_files = []

    for fname in sorted(os.listdir(folder_path)):
        full_path = os.path.join(folder_path, fname)

        if not fname.lower().endswith(SUPPORTED_EXTENSIONS):
            skipped_files.append(fname)
            continue

        print(f"  -> loading {fname}")
        try:
            docs = load_single_file(full_path)
            loaded_docs.extend(docs)
        except Exception as e:
            print(f"     failed to load {fname}: {e}")

    if skipped_files:
        print(f"\nSkipped (unsupported type): {skipped_files}")

    return loaded_docs


def chunk_documents(documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """
    Splits documents into overlapping chunks. Using recursive splitting
    so it prefers breaking at paragraph/sentence boundaries first, and
    only falls back to raw character cuts if it has no other choice -
    keeps chunks from ending mid-sentence as much as possible.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def summarize_chunks(chunks):
    """
    Just a sanity-check helper - prints stats about the chunks so I can
    eyeball whether my chunk_size choice is reasonable (e.g. are chunks
    consistently hitting the max, or mostly much smaller?).
    """
    if not chunks:
        print("No chunks to summarize.")
        return

    lengths = [len(c.page_content) for c in chunks]
    print(f"Total chunks: {len(chunks)}")
    print(f"Avg chunk length: {sum(lengths) / len(lengths):.0f} chars")
    print(f"Shortest chunk: {min(lengths)} chars")
    print(f"Longest chunk: {max(lengths)} chars")


if __name__ == "__main__":
    print("Loading documents from ./data ...")
    raw_docs = load_data_folder("data")
    print(f"\nLoaded {len(raw_docs)} pages/sections total.\n")

    print("Chunking...")
    chunks = chunk_documents(raw_docs)
    print()
    summarize_chunks(chunks)

    print("\nPreview of first 2 chunks:\n")
    for idx, chunk in enumerate(chunks[:2], start=1):
        print(f"[Chunk {idx}] ({len(chunk.page_content)} chars)")
        print(chunk.page_content)
        print(f"metadata -> {chunk.metadata}")
        print("-" * 50)