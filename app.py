"""
Smart Research Assistant — Streamlit App
Full RAG pipeline: ingest → embed → retrieve → generate → evaluate
Supports local Ollama + Cloud OpenAI, FAISS + ChromaDB, URL Scraping,
Multiple Embedding Models, Match Highlighting, Voice Synthesis, and full RAGAs Evaluation.
"""

import os
import sys
import time
import re
import json

# Setup system path for src imports
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import streamlit as st
from src.embed_store import (
    get_embedding_model,
    build_vector_store,
    save_vector_store,
    load_vector_store,
    build_chroma_store,
    load_chroma_store,
    get_index_path,
)
from src.retrieve import get_relevant_chunks, format_context
from src.generate import get_llm, answer_question, OLLAMA_MODEL_NAME
from src.ingest import load_data_folder, chunk_documents, scrape_url_to_file
from src.evaluate import TEST_QUESTIONS

# Page configuration
st.set_page_config(
    page_title="Smart Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS theme styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Outfit', sans-serif;
    font-weight: 600;
}

/* Base Body Style */
.stApp {
    background-color: #0d1117;
    color: #c9d1d9;
}

/* Header gradient */
.grad-text {
    background: linear-gradient(135deg, #4f46e5 0%, #a855f7 50%, #ec4899 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 2.8rem;
    font-weight: 800;
    line-height: 1.2;
    margin-bottom: 0.2rem;
}
.subtitle {
    color: #8b949e;
    font-size: 1.05rem;
    margin-bottom: 2rem;
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #30363d;
}
[data-testid="stSidebar"] h3 {
    color: #58a6ff;
}

/* Card layout for info */
.stat-card {
    background: rgba(22, 27, 34, 0.7);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 0.85rem 1.1rem;
    margin-bottom: 0.75rem;
    backdrop-filter: blur(8px);
}
.stat-label {
    color: #8b949e;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: .08em;
    font-weight: 600;
}
.stat-value {
    color: #f0f6fc;
    font-size: 0.95rem;
    font-weight: 600;
    margin-top: 0.2rem;
}

/* Chat bubble aesthetics */
.msg-row {
    display: flex;
    gap: 0.95rem;
    margin-bottom: 1.3rem;
    animation: slideIn 0.35s cubic-bezier(0.16, 1, 0.3, 1) forwards;
}
@keyframes slideIn {
    from { opacity: 0; transform: translateY(12px); }
    to { opacity: 1; transform: translateY(0); }
}
.msg-row.user {
    flex-direction: row-reverse;
}
.av {
    width: 38px;
    height: 38px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.05rem;
    flex-shrink: 0;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}
.av-u {
    background: linear-gradient(135deg, #4f46e5, #a855f7);
    border: 1px solid rgba(255,255,255,0.1);
}
.av-a {
    background: linear-gradient(135deg, #059669, #10b981);
    border: 1px solid rgba(255,255,255,0.1);
}
.bubble {
    max-width: 80%;
    padding: 0.9rem 1.2rem;
    border-radius: 16px;
    font-size: 0.95rem;
    line-height: 1.6;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}
.bubble-u {
    background: rgba(79, 70, 229, 0.12);
    border: 1px solid rgba(79, 70, 229, 0.35);
    color: #f0f6fc;
    border-top-right-radius: 3px;
}
.bubble-a {
    background: #161b22;
    border: 1px solid #30363d;
    color: #e6edf3;
    border-top-left-radius: 3px;
}

/* Source tags */
.chips {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    margin-top: 0.75rem;
}
.chip {
    background: #0d1117;
    border: 1px solid rgba(16, 185, 129, 0.25);
    color: #10b981;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 0.2rem 0.65rem;
    border-radius: 99px;
}

/* Styled text input & textarea */
.stTextInput>div>div>input, .stTextArea>div>div>textarea {
    background-color: #0d1117 !important;
    border: 1px solid #30363d !important;
    border-radius: 10px !important;
    color: #c9d1d9 !important;
}
.stTextInput>div>div>input:focus, .stTextArea>div>div>textarea:focus {
    border-color: #58a6ff !important;
    box-shadow: 0 0 0 1px #58a6ff !important;
}

/* Tabs customized styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 1.5rem;
    border-bottom: 1px solid #30363d;
}
.stTabs [data-baseweb="tab"] {
    height: 3rem;
    background-color: transparent !important;
    color: #8b949e !important;
    font-weight: 500;
    border: none !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #c9d1d9 !important;
}
.stTabs [aria-selected="true"] {
    color: #58a6ff !important;
    border-bottom: 2px solid #58a6ff !important;
    font-weight: 600;
}

/* Custom Highlight styles */
mark.matched {
    background-color: rgba(245, 158, 11, 0.22);
    border-bottom: 2px solid #f59e0b;
    color: #f0f6fc;
    padding: 0.05rem 0.2rem;
    border-radius: 3px;
}

/* Remove default footer and header */
#MainMenu, footer, header[data-testid="stHeader"] {
    visibility: hidden;
    height: 0;
}
</style>
""", unsafe_allow_html=True)


# Helper functions
def highlight_text(text, query):
    """
    Finds keyword matches from query in text and wraps them in HTML mark tags.
    Filters out common stopwords to avoid highlighting everything.
    """
    if not query:
        return text

    # Extract terms (words) from the user query
    terms = re.findall(r'\w+', query.lower())
    # Exclude typical small grammatical terms
    stopwords = {
        "what", "is", "the", "a", "an", "and", "or", "but", "in", "on", "at",
        "for", "with", "this", "that", "these", "those", "does", "doesnt",
        "how", "why", "where", "you", "me", "i", "we", "he", "she", "it"
    }
    terms = [t for t in terms if len(t) > 2 and t not in stopwords]

    if not terms:
        return text

    # Escape terms and build query pattern
    terms = sorted(list(set(terms)), key=len, reverse=True)
    highlighted = text

    for term in terms:
        # Match word boundaries strictly to avoid partial matches
        pattern = re.compile(rf'\b({re.escape(term)})\b', re.IGNORECASE)
        highlighted = pattern.sub(r'<mark class="matched">\1</mark>', highlighted)

    return highlighted


def make_tts_html(text):
    """
    Embeds a browser SpeechSynthesis client-side script for text-to-speech.
    Works instantly on click and runs 100% locally in user browser.
    """
    clean_text = re.sub(r'[*_`#\-]', ' ', text)
    safe_text_js = json.dumps(clean_text)

    html_code = f"""
    <style>
        body {{
            margin: 0;
            padding: 0;
            overflow: hidden;
            background-color: transparent;
        }}
    </style>
    <script>
        var ttsText = {safe_text_js};
        function speakText() {{
            var synth = window.parent.speechSynthesis || window.speechSynthesis;
            if (synth) {{
                var msg = new SpeechSynthesisUtterance(ttsText);
                msg.rate = 1.05;
                synth.cancel();
                synth.speak(msg);
            }} else {{
                console.error('Speech synthesis not supported');
            }}
        }}
    </script>
    <div style="display:flex; justify-content: flex-end; align-items: center; height: 38px;">
        <button onclick="speakText()" style="
            background: rgba(22, 27, 34, 0.6);
            border: 1px solid #30363d;
            color: #8b949e;
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 0.72rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s ease;
        " onmouseover="this.style.borderColor='#58a6ff';this.style.color='#58a6ff';this.style.background='rgba(88,166,255,0.05)';"
           onmouseout="this.style.borderColor='#30363d';this.style.color='#8b949e';this.style.background='rgba(22, 27, 34, 0.6)';">
             🔊 Read Aloud
        </button>
    </div>
    """
    return html_code


def make_stt_html():
    """
    Renders a microphone button that triggers browser SpeechRecognition
    and writes the transcribed text directly into Streamlit's input box.
    """
    html_code = """
    <style>
        body {
            margin: 0;
            padding: 0;
            overflow: hidden;
            background-color: transparent;
        }
    </style>
    <div style="display:flex; justify-content: center; align-items:center; height: 44px;">
        <button id="mic-btn" onclick="
            var btn = document.getElementById('mic-btn');
            try {
                var parentWin = window.parent;
                
                // Define the speech recognition function in the parent window context
                // so it bypasses the iframe's strict microphone permission blocks
                if (!parentWin.runSpeechRecognition) {
                    parentWin.runSpeechRecognition = function() {
                        var doc = parentWin.document;
                        var SpeechRecognition = parentWin.SpeechRecognition || parentWin.webkitSpeechRecognition;
                        if (!SpeechRecognition) {
                            alert('Speech recognition is not supported in this browser. Please use Chrome or Edge.');
                            return;
                        }
                        
                        var recognition = new SpeechRecognition();
                        recognition.continuous = false;
                        recognition.interimResults = false;
                        recognition.lang = 'en-US';
                        
                        // Helper to find our button inside the parent's iframe list
                        var findMicButton = function() {
                            var iframes = doc.querySelectorAll('iframe');
                            for (var i = 0; i < iframes.length; i++) {
                                try {
                                    var iframeDoc = iframes[i].contentDocument || iframes[i].contentWindow.document;
                                    var b = iframeDoc.getElementById('mic-btn');
                                    if (b) return b;
                                } catch (e) {}
                            }
                            return null;
                        };
                        
                        recognition.onstart = function() {
                            var b = findMicButton();
                            if (b) {
                                b.style.background = '#ef4444';
                                b.style.borderColor = '#ef4444';
                                b.style.color = '#ffffff';
                                b.innerHTML = '🔴 Listening...';
                            }
                        };
                        
                        recognition.onerror = function(event) {
                            console.error('Speech recognition error', event.error);
                            var b = findMicButton();
                            if (b) {
                                b.style.background = 'rgba(22, 27, 34, 0.6)';
                                b.style.borderColor = '#30363d';
                                b.style.color = '#e6edf3';
                                b.innerHTML = '🎙️ Error: ' + event.error;
                            }
                        };
                        
                        recognition.onend = function() {
                            var b = findMicButton();
                            if (b) {
                                b.style.background = 'rgba(22, 27, 34, 0.6)';
                                b.style.borderColor = '#30363d';
                                b.style.color = '#e6edf3';
                                b.innerHTML = '🎙️ Speak';
                            }
                        };
                        
                        recognition.onresult = function(event) {
                            var transcript = event.results[0][0].transcript;
                            var parentInput = doc.querySelector('input[placeholder*=\\'Ask anything\\']');
                            
                            if (parentInput) {
                                var setter = Object.getOwnPropertyDescriptor(parentWin.HTMLInputElement.prototype, 'value').set;
                                setter.call(parentInput, transcript);
                                
                                // Dispatch standard events to notify React of the change
                                parentInput.dispatchEvent(new Event('input', { bubbles: true }));
                                parentInput.dispatchEvent(new Event('change', { bubbles: true }));
                                parentInput.dispatchEvent(new Event('blur', { bubbles: true }));
                                parentInput.dispatchEvent(new Event('focusout', { bubbles: true }));
                                
                                // Wait 250ms for React state to sync and send WebSocket data, then programmatically trigger the Send button
                                setTimeout(function() {
                                    var buttons = doc.querySelectorAll('button');
                                    for (var i = 0; i < buttons.length; i++) {
                                        if (buttons[i].innerText && buttons[i].innerText.indexOf('Send') !== -1) {
                                            buttons[i].click();
                                            break;
                                        }
                                    }
                                }, 250);
                            }
                        };
                        
                        recognition.start();
                    };
                }
                
                // Call the parent window speech recognition function
                parentWin.runSpeechRecognition();
                
            } catch (err) {
                btn.innerHTML = '⚠️ Error: ' + err.message.substring(0, 10);
                console.error(err);
            }
        " style="
            background: rgba(22, 27, 34, 0.6);
            border: 1px solid #30363d;
            color: #e6edf3;
            padding: 5px 12px;
            border-radius: 8px;
            font-size: 0.82rem;
            font-weight: 600;
            cursor: pointer;
            height: 38px;
            width: 100%;
            transition: all 0.2s ease;
            box-sizing: border-box;
            display: flex;
            align-items: center;
            justify-content: center;
        " onmouseover="this.style.borderColor='#58a6ff';this.style.color='#58a6ff';this.style.background='rgba(88,166,255,0.05)';"
           onmouseout="this.style.borderColor='#30363d';this.style.color='#e6edf3';this.style.background='rgba(22, 27, 34, 0.6)';">
             🎙️ Speak
        </button>
    </div>
    """
    return html_code


# Session state initialization
def _init():
    for k, v in {
        "messages": [],
        "vector_store": None,
        "eval_results": None,
        "store_type": "FAISS",
        "provider": "Ollama",
        "openai_api_key": "",
        "embed_model": "sentence-transformers/all-MiniLM-L6-v2",
        "loaded_provider": None,
        "loaded_embed_model": None,
        "loaded_store_type": None,
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()


# Sidebar config panel
def _sidebar():
    with st.sidebar:
        st.markdown("### 🛠️ Configuration")

        # Provider Selector
        provider = st.radio(
            "API Provider",
            ["Ollama", "OpenAI"],
            help="Ollama runs locally for free. OpenAI runs in the cloud (requires key)."
        )
        st.session_state.provider = provider

        # API Key input if OpenAI
        api_key = ""
        if provider == "OpenAI":
            api_key = st.text_input(
                "OpenAI API Key",
                value=st.session_state.openai_api_key,
                type="password",
                placeholder="sk-..."
            )
            st.session_state.openai_api_key = api_key
            if not api_key:
                st.warning("🔑 Please enter an API key to use OpenAI models.")

        # Embedding model configuration
        st.markdown("##### Embedding Model")
        if provider == "Ollama":
            embed_options = [
                "sentence-transformers/all-MiniLM-L6-v2",
                "sentence-transformers/all-mpnet-base-v2"
            ]
        else:
            embed_options = [
                "text-embedding-3-small",
                "text-embedding-3-large"
            ]

        embed_model = st.selectbox(
            "Select Embedding Model",
            options=embed_options,
            help="Switching embedding models will require rebuilding/reloading the vector database index."
        )
        st.session_state.embed_model = embed_model

        # LLM Selection
        st.markdown("##### LLM Model")
        if provider == "Ollama":
            llm_model = st.text_input(
                "Ollama Model Name",
                value=OLLAMA_MODEL_NAME,
                placeholder="e.g. llama3.2"
            )
        else:
            llm_model = st.selectbox(
                "OpenAI Model Name",
                options=["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]
            )

        st.divider()

        # Database Settings
        st.markdown("### 🗄️ Vector Database")
        store_type = st.radio(
            "Active Vector DB",
            ["FAISS", "ChromaDB"],
            horizontal=True,
            help="FAISS: Lightweight local index. ChromaDB: Persistent local database."
        )
        st.session_state.store_type = store_type

        # Advanced retrieval parameters
        st.markdown("##### Retrieval Parameters")
        k = st.slider("Context chunks (k)", 2, 10, 4, help="Number of document chunks to retrieve.")
        use_mmr = st.toggle("MMR Search", False, help="Maximal Marginal Relevance reduces redundancy.")

        st.divider()

        # Render active parameters card
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Active Provider</div>
            <div class="stat-value">{provider}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Active LLM</div>
            <div class="stat-value">{llm_model}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Vector Database</div>
            <div class="stat-value">{store_type}</div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    return llm_model, api_key, embed_model, k, use_mmr, store_type


# Chat rendering helper
def _render_chat(query_text_for_highlight):
    if not st.session_state.messages:
        st.markdown("""
        <div style="text-align:center; padding:4rem 0; color:#8b949e;">
          <div style="font-size:3.5rem;">💬</div>
          <h4 style="margin-top:1rem; color:#c9d1d9;">Ask your first question</h4>
          <div style="font-size:0.88rem; color:#8b949e; max-width:400px; margin:0 auto; line-height:1.5;">
            Upload files or paste a web URL in the <strong>Documents</strong> tab, select your models, and ask questions here.
          </div>
        </div>""", unsafe_allow_html=True)
        return

    # Display conversational message list
    for idx, msg in enumerate(st.session_state.messages):
        role = msg["role"]
        is_user = role == "user"
        av_cls = "av-u" if is_user else "av-a"
        av_icon = "👤" if is_user else "🤖"
        bub_cls = "bubble-u" if is_user else "bubble-a"
        row_cls = "msg-row user" if is_user else "msg-row"

        # Embed source chips
        chips_html = ""
        if not is_user and msg.get("sources"):
            chips = "".join(f'<span class="chip">📄 {s}</span>' for s in msg["sources"])
            chips_html = f'<div class="chips">{chips}</div>'

        # Main message structure
        st.markdown(f"""
        <div class="{row_cls}">
          <div class="av {av_cls}">{av_icon}</div>
          <div class="bubble {bub_cls}">
            <div>{msg["content"]}</div>
            {chips_html}
          </div>
        </div>""", unsafe_allow_html=True)

        # Extra UI widgets for AI responses
        if not is_user:
            # Voice Speech Synthesis
            st.components.v1.html(make_tts_html(msg["content"]), height=38)

            # Grounded chunk details with word matches highlighted
            if msg.get("raw_chunks"):
                with st.expander("🔍 Grounded Context & Highlighted Matches", expanded=False):
                    for c_idx, c in enumerate(msg["raw_chunks"]):
                        src_name = os.path.basename(c.metadata.get("source", "unknown"))
                        page_num = c.metadata.get("page", c.metadata.get("page_label", "?"))
                        st.markdown(f"**Chunk {c_idx+1} — Source:** `{src_name}` | **Page:** `{page_num}`")

                        # Perform matching highlight in html
                        highlighted_body = highlight_text(c.page_content, msg.get("query", ""))
                        st.markdown(f"""<div style="background-color:#161b22; border-left:3px solid #10b981; 
                                        padding:0.75rem 1rem; margin-bottom:1rem; font-size:0.85rem; 
                                        color:#c9d1d9; border-radius: 0 8px 8px 0; overflow-x: auto;">
                                        {highlighted_body}</div>""", unsafe_allow_html=True)


# Main app entrypoint
def main():
    # Header
    st.markdown('<div class="grad-text">🔬 Smart Research Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Natural language retrieval assistant — local Ollama or cloud OpenAI</div>', unsafe_allow_html=True)

    llm_model, api_key, embed_model, k, use_mmr, store_type = _sidebar()

    # Determine dynamic database storage path
    idx_path = get_index_path(embed_model, store_type)

    # Determine if configuration changed, forcing reload
    config_changed = (
        st.session_state.loaded_provider != st.session_state.provider or
        st.session_state.loaded_embed_model != embed_model or
        st.session_state.loaded_store_type != store_type
    )

    if config_changed:
        st.session_state.vector_store = None

    # Load embedding model
    with st.spinner("⚙️ Loading embedding model..."):
        try:
            emb = get_embedding_model(model_name=embed_model, provider=st.session_state.provider, api_key=api_key)
        except Exception as e:
            st.error(f"Failed to load embedding model: {e}")
            return

    # Load vector store from disk
    if st.session_state.vector_store is None:
        try:
            if store_type == "FAISS":
                st.session_state.vector_store = load_vector_store(emb, path=idx_path)
            else:
                st.session_state.vector_store = load_chroma_store(emb, path=idx_path)
            st.session_state.loaded_provider = st.session_state.provider
            st.session_state.loaded_embed_model = embed_model
            st.session_state.loaded_store_type = store_type
        except FileNotFoundError:
            pass  # Index not built yet, will prompt user

    # Load LLM
    llm = None
    if st.session_state.provider == "OpenAI" and not api_key:
        # LLM disabled until API key provided
        pass
    else:
        with st.spinner("🧠 Initializing Language Model..."):
            try:
                llm = get_llm(provider=st.session_state.provider, model_name=llm_model, api_key=api_key)
            except Exception as e:
                st.error(f"Failed to initialize LLM: {e}")

    # Tabs
    tab_chat, tab_docs, tab_eval = st.tabs(["💬 Chat", "📁 Documents & URLs", "📊 Evaluation"])

    # Chat tab interface
    with tab_chat:
        if st.session_state.vector_store is None:
            st.info("📁 To begin, please upload documents or scrape a URL in the **Documents & URLs** tab.")
        elif llm is None and st.session_state.provider == "OpenAI":
            st.warning("🔑 Please enter your OpenAI API key in the sidebar configuration.")
        else:
            # Active input highlight tracking
            last_query = st.session_state.messages[-2]["content"] if len(st.session_state.messages) >= 2 and st.session_state.messages[-2]["role"] == "user" else ""
            _render_chat(last_query)

            st.divider()

            # Chat inputs
            col_q, col_mic, col_btn = st.columns([4, 1, 1])
            with col_q:
                question = st.text_input(
                    "User Input", label_visibility="collapsed",
                    placeholder="Ask anything about your loaded documents...",
                    key="chat_input",
                )
            with col_mic:
                st.components.v1.html(make_stt_html(), height=44)
            with col_btn:
                send = st.button("Send ➤", use_container_width=True)

            if send and question.strip():
                chat_history = list(st.session_state.messages)
                st.session_state.messages.append({"role": "user", "content": question.strip()})

                with st.spinner("⚡ Retrieving relevant segments and generating answer..."):
                    try:
                        ans, chunks = answer_question(
                            question.strip(), st.session_state.vector_store, llm,
                            k=k, use_mmr=use_mmr, chat_history=chat_history
                        )

                        # Track unique sources names/pages for bubble chips
                        source_labels = []
                        seen_labels = set()
                        for c in chunks:
                            src_fn = os.path.basename(c.metadata.get("source", "unknown"))
                            page_no = c.metadata.get("page", c.metadata.get("page_label", "?"))
                            label = f"{src_fn} (p.{page_no})"
                            if label not in seen_labels:
                                source_labels.append(label)
                                seen_labels.add(label)

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": ans,
                            "query": question.strip(),
                            "sources": source_labels,
                            "raw_chunks": chunks
                        })
                    except Exception as e:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"❌ Generation Error: {e}",
                            "sources": [],
                            "raw_chunks": []
                        })
                st.rerun()

    # Document ingestion tab interface
    with tab_docs:
        st.subheader("📁 Knowledge Base & Ingestion")
        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown("#### 📂 Current Ingested Files")
            data_dir = "data"
            if os.path.isdir(data_dir):
                files = [f for f in os.listdir(data_dir) if f.lower().endswith((".pdf", ".txt"))]
                if files:
                    for f in files:
                        col_file, col_del = st.columns([5, 1])
                        with col_file:
                            kb = os.path.getsize(os.path.join(data_dir, f)) / 1024
                            st.markdown(f"📄 **{f}** &nbsp; `{kb:.1f} KB`")
                        with col_del:
                            if st.button("🗑️", key=f"del_{f}", help=f"Delete {f}"):
                                try:
                                    os.remove(os.path.join(data_dir, f))
                                    with st.spinner("Updating vector database..."):
                                        raw = load_data_folder(data_dir)
                                        if raw:
                                            chunks = chunk_documents(raw)
                                            if store_type == "FAISS":
                                                store = build_vector_store(chunks, emb)
                                                save_vector_store(store, path=idx_path)
                                            else:
                                                store = build_chroma_store(chunks, emb, path=idx_path)
                                        else:
                                            store = None
                                            import shutil
                                            if os.path.exists(idx_path):
                                                if os.path.isdir(idx_path):
                                                    shutil.rmtree(idx_path, ignore_errors=True)
                                                else:
                                                    os.remove(idx_path)
                                        st.session_state.vector_store = store
                                    st.success(f"Deleted {f} and updated index!")
                                    st.rerun()
                                except Exception as err:
                                    st.error(f"Failed to delete: {err}")
                else:
                    st.info("No documents uploaded yet.")

            st.markdown("#### 🔍 Selected Index Status")
            idx_exists = os.path.exists(idx_path)
            if idx_exists:
                st.success(f"✅ Vector index built and active (`{idx_path}`)")
            else:
                st.warning(f"❌ Index not found (`{idx_path}`). Please upload docs or input a URL to build it.")

        with col_r:
            st.markdown("#### 📥 Ingest Documents")
            uploads = st.file_uploader(
                "Upload PDF or TXT files", type=["pdf", "txt"],
                accept_multiple_files=True, label_visibility="collapsed"
            )

            if uploads and st.button("⚡ Save & Reindex Files", use_container_width=True):
                os.makedirs(data_dir, exist_ok=True)
                for f in uploads:
                    dest = os.path.join(data_dir, f.name)
                    with open(dest, "wb") as fp:
                        fp.write(f.read())
                    st.success(f"Saved local file: {f.name}")

                with st.spinner("🔄 Building vector embeddings..."):
                    raw = load_data_folder(data_dir)
                    chunks = chunk_documents(raw)
                    if store_type == "FAISS":
                        store = build_vector_store(chunks, emb)
                        save_vector_store(store, path=idx_path)
                    else:
                        store = build_chroma_store(chunks, emb, path=idx_path)
                    st.session_state.vector_store = store
                    st.session_state.loaded_provider = st.session_state.provider
                    st.session_state.loaded_embed_model = embed_model
                    st.session_state.loaded_store_type = store_type
                st.success(f"Success! Vector database populated with {len(chunks)} text chunks.")
                st.rerun()

            st.divider()

            st.markdown("#### 🔗 Ingest Web Content")
            url_to_scrape = st.text_input("Enter Web URL (Article, Documentation, Blog post)", placeholder="https://example.com/topic")
            if url_to_scrape and st.button("🚀 Fetch & Index URL", use_container_width=True):
                with st.spinner(f"Scraping web page content from {url_to_scrape}..."):
                    try:
                        filepath, filename = scrape_url_to_file(url_to_scrape, data_dir=data_dir)
                        st.success(f"Saved web text content to `{filename}`")

                        # Rebuild database index
                        with st.spinner("Updating vector database..."):
                            raw = load_data_folder(data_dir)
                            chunks = chunk_documents(raw)
                            if store_type == "FAISS":
                                store = build_vector_store(chunks, emb)
                                save_vector_store(store, path=idx_path)
                            else:
                                store = build_chroma_store(chunks, emb, path=idx_path)
                            st.session_state.vector_store = store
                            st.session_state.loaded_provider = st.session_state.provider
                            st.session_state.loaded_embed_model = embed_model
                            st.session_state.loaded_store_type = store_type
                        st.success("Web page successfully chunked, embedded, and added to search index!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Ingestion failed: {e}")

    # Evaluation pipeline tab interface
    with tab_eval:
        st.subheader("📊 RAG Pipeline Quality Evaluation")
        st.markdown("""
        Automated scoring judges the answers generated by your RAG pipeline against four key metrics:
        1. **Faithfulness**: Is the answer fully grounded in the retrieved documents? (Prevents hallucinations)
        2. **Answer Relevancy**: Does the generated answer address the question?
        3. **Context Precision**: Does the database retrieve relevant context chunks at the top ranks?
        4. **Answer Correctness**: How closely does the generated answer match the reference ground truth?
        """)

        st.info("⏱️ Runs evaluation utilizing the active LLM as a judge. Can take several minutes depending on CPU performance.")

        # Default questions list formatting
        default_qs = ""
        for record in TEST_QUESTIONS:
            default_qs += f"{record['question']} | {record['ground_truth']}\n"

        st.markdown("**Test Questions & Optional Ground Truth Answers**")
        st.caption("Format: `Question | Ground Truth`. If no ground truth is provided, the system automatically uses the LLM to generate it.")
        raw_qs = st.text_area(
            "Enter evaluation items (one per line)",
            value=default_qs.strip(),
            height=140
        )

        if st.button("🚀 Run Evaluation Pipeline", use_container_width=True):
            if st.session_state.vector_store is None:
                st.error("No search index loaded. Please upload documents or scrape a URL in the Documents tab first.")
            elif llm is None:
                st.error("Model engine not initialized. Ensure Ollama is running or input an OpenAI API key.")
            else:
                lines = [line.strip() for line in raw_qs.strip().splitlines() if line.strip()]
                eval_records = []
                for line in lines:
                    if "|" in line:
                        parts = line.split("|", 1)
                        eval_records.append({
                            "question": parts[0].strip(),
                            "ground_truth": parts[1].strip()
                        })
                    else:
                        eval_records.append({
                            "question": line.strip(),
                            "ground_truth": ""
                        })

                try:
                    from src.evaluate import run_pipeline_for_eval, run_evaluation
                    with st.spinner("🤖 Generation Phase: Fetching contexts and generating answers..."):
                        records = run_pipeline_for_eval(eval_records, st.session_state.vector_store, llm)
                    with st.spinner("⚖️ Judging Phase: Scoring faithfulness, relevancy, precision, and correctness..."):
                        results = run_evaluation(records, llm, emb)
                    st.session_state.eval_results = results.to_pandas()
                    st.success("Evaluation complete!")
                except Exception as e:
                    st.error(f"Evaluation failed: {e}")
                    st.caption("Verify that dependencies `ragas` and `datasets` are installed correctly.")

        # Display results
        if st.session_state.eval_results is not None:
            df = st.session_state.eval_results
            metrics = ["faithfulness", "answer_relevancy", "context_precision", "answer_correctness"]
            avail = [m for m in metrics if m in df.columns]

            # Renders averages
            st.markdown("#### 📈 Average Scores Summary")
            cols = st.columns(len(avail))
            for col, m in zip(cols, avail):
                name = m.replace("_", " ").title()
                score = df[m].mean()
                col.metric(name, f"{score:.3f}")

            # Renders tabular breakdown
            st.markdown("#### 📋 Question-by-Question Score Detail")
            display_cols = ["question", "answer", "ground_truth"] + avail
            st.dataframe(df[display_cols], use_container_width=True)

            st.download_button(
                "⬇️ Export Results as CSV",
                data=df.to_csv(index=False),
                file_name="rag_eval_results.csv",
                mime="text/csv",
                use_container_width=True
            )


if __name__ == "__main__":
    main()
