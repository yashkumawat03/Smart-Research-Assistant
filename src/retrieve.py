"""
Step 3: Given a user's question, fetch the most relevant chunks from
the FAISS index. This is the "R" in RAG - pure retrieval, no LLM
involved yet, that's the next file.
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

from embed_store import get_embedding_model, load_vector_store

DEFAULT_K = 4



def get_relevant_chunks(query, vector_store, k=DEFAULT_K, use_mmr=False):
    """
    Fetches the top-k chunks most relevant to the query.

    use_mmr=False -> plain similarity search, just ranks by closeness
    use_mmr=True  -> Maximal Marginal Relevance, favors diverse results
                     over near-duplicate ones (useful when source docs
                     repeat similar info in multiple places)
    """
    if use_mmr:
        return vector_store.max_marginal_relevance_search(query, k=k)
    return vector_store.similarity_search(query, k=k)


def get_relevant_chunks_with_scores(query, vector_store, k=DEFAULT_K):
    """
    Same as above but also returns the similarity score per chunk.
    Handy for debugging retrieval quality - lets you actually see
    HOW confident the match was, not just what came back.
    Lower score = more similar (FAISS returns L2 distance by default).
    """
    return vector_store.similarity_search_with_score(query, k=k)


def format_context(chunks):
    """
    Combines retrieved chunks into a single text block, tagged with
    their source, ready to be dropped into an LLM prompt in the next
    stage. Keeping source tags in here so the LLM (and we) can trace
    which chunk backed which part of an answer.
    """
    pieces = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page_label", "?")
        pieces.append(
            f"[Source {i} - {source}, page {page}]\n{chunk.page_content}"
        )
    return "\n\n".join(pieces)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    embedding_model = get_embedding_model()
    store = load_vector_store(embedding_model)

    test_query = input("Ask a question about your document: ")

    print("\n--- Plain similarity search ---")
    results = get_relevant_chunks(test_query, store, k=3, use_mmr=False)
    for i, r in enumerate(results, start=1):
        print(f"\n[{i}] {r.page_content[:150]}...")

    print("\n--- With similarity scores ---")
    scored_results = get_relevant_chunks_with_scores(test_query, store, k=3)
    for chunk, score in scored_results:
        print(f"score={score:.4f} | {chunk.page_content[:100]}...")

    print("\n--- Formatted context (what the LLM will actually see) ---")
    context = format_context(results)
    print(context)