"""
Step 4: Take retrieved context + the user's question, build a grounded
prompt, and get an answer from a locally-running Ollama model.

This is where retrieval (retrieve.py) finally meets generation.
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

from langchain_ollama import OllamaLLM
from embed_store import get_embedding_model, load_vector_store
from retrieve import get_relevant_chunks, format_context

OLLAMA_MODEL_NAME = "llama3.2"


# Keeping the prompt as a template string I can tweak easily - this is
# the single most important piece of this file. The instructions here
# are what keep the model grounded instead of making things up.
PROMPT_TEMPLATE = """You are a helpful research assistant. Answer the user's question using ONLY the context provided below.

Guidelines:
- Base your answer strictly on the provided context chunks.
- If asked to "summarize", provide a clear, structured summary of the key points found across all context chunks.
- If the context does not contain enough information, say: "I don't have enough information in the provided documents to answer this."
- Do not use outside knowledge beyond what is in the context.
- Cite the source(s) where relevant.

{history_block}Relevant Document Context:
{context}

Current Question: {question}

Answer:"""


def get_llm(provider="Ollama", model_name=OLLAMA_MODEL_NAME, api_key=None):
    """
    Connects to either local Ollama LLM or cloud OpenAI Chat model.
    """
    if provider == "OpenAI":
        from langchain_openai import ChatOpenAI
        if not model_name or "llama" in model_name.lower():
            model_name = "gpt-4o-mini"
        print(f"Connecting to OpenAI Chat model: {model_name}")
        return ChatOpenAI(model=model_name, openai_api_key=api_key)
    else:
        # Default to Ollama
        print(f"Connecting to local Ollama LLM: {model_name}")
        return OllamaLLM(model=model_name)


def build_prompt(question, context, chat_history=None):
    """
    Fills in the prompt template with the retrieved context, the user's
    question, and optionally the last few conversation turns so the model
    can answer follow-up questions correctly (multi-turn memory).
    """
    history_block = ""
    if chat_history:
        lines = ["Conversation History (for reference — answer only from documents below):"]
        for msg in chat_history[-4:]:  # keep last 2 exchanges (4 messages)
            role = "User" if msg["role"] == "user" else "Assistant"
            content = msg["content"][:400] + "..." if len(msg["content"]) > 400 else msg["content"]
            lines.append(f"{role}: {content}")
        history_block = "\n".join(lines) + "\n\n"
    return PROMPT_TEMPLATE.format(context=context, question=question, history_block=history_block)


def answer_question(question, vector_store, llm, k=4, use_mmr=False, verbose=False, chat_history=None):
    """
    The full pipeline for one question:
    retrieve chunks → build grounded prompt (with optional history) → call the LLM → return answer.
    Returns both the answer and the chunks used so the UI can show sources.
    chat_history is optional — when provided the model can handle follow-up questions.
    """
    chunks = get_relevant_chunks(question, vector_store, k=k, use_mmr=use_mmr)
    context = format_context(chunks)
    prompt = build_prompt(question, context, chat_history=chat_history)

    if verbose:
        print("\n----- PROMPT SENT TO LLM -----")
        print(prompt)
        print("-------------------------------\n")

    answer = llm.invoke(prompt)
    if hasattr(answer, "content"):
        answer = answer.content
    return answer, chunks


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    embedding_model = get_embedding_model()
    store = load_vector_store(embedding_model)
    llm = get_llm()

    while True:
        question = input("\nAsk a question (or type 'exit'): ")
        if question.strip().lower() == "exit":
            break

        answer, sources = answer_question(question, store, llm, verbose=True)

        print("\n=== ANSWER ===")
        print(answer)

        print("\n=== SOURCES USED ===")
        for i, chunk in enumerate(sources, start=1):
            page = chunk.metadata.get("page_label", "?")
            print(f"[{i}] page {page}")