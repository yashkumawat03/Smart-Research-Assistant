"""
Step 4: Take retrieved context + the user's question, build a grounded
prompt, and get an answer from a locally-running Ollama model or cloud Groq/OpenAI APIs.

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

from embed_store import get_embedding_model, load_vector_store
from retrieve import get_relevant_chunks, format_context

OLLAMA_MODEL_NAME = "llama3.2"
GROQ_MODEL_NAME = "llama-3.3-70b-versatile"


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


def get_groq_api_key(api_key=None):
    # Highest priority: explicitly passed key
    if api_key:
        return api_key

    # Streamlit Secrets (Deployment)
    try:
        import streamlit as st
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass

    # Local .env
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    return os.getenv("GROQ_API_KEY")


def get_llm(provider="Ollama", model_name=OLLAMA_MODEL_NAME, api_key=None):
    """
    Connects to either local Ollama LLM, cloud Groq Chat model, or cloud OpenAI Chat model.
    """
    provider_clean = str(provider).lower() if provider else "ollama"

    if "groq" in provider_clean:
        from langchain_groq import ChatGroq
        key = get_groq_api_key(api_key)
        if not key:
            raise ValueError(
                "Groq API key not found. Please enter it in the sidebar, set st.secrets['GROQ_API_KEY'], or add GROQ_API_KEY to your .env file."
            )
        target_model = model_name if model_name and ("llama-3" in model_name.lower() or "mixtral" in model_name.lower() or "gemma" in model_name.lower()) else GROQ_MODEL_NAME
        print(f"Connecting to Groq Cloud LLM: {target_model}")
        return ChatGroq(model=target_model, groq_api_key=key)

    elif "openai" in provider_clean:
        from langchain_openai import ChatOpenAI
        target_model = model_name if model_name and not "llama" in model_name.lower() else "gpt-4o-mini"
        print(f"Connecting to OpenAI Chat model: {target_model}")
        return ChatOpenAI(model=target_model, openai_api_key=api_key)

    else:
        # Default to local Ollama
        from langchain_ollama import OllamaLLM
        target_model = model_name if model_name else OLLAMA_MODEL_NAME
        print(f"Connecting to local Ollama LLM: {target_model}")
        return OllamaLLM(model=target_model)


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