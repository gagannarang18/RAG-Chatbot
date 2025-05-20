import streamlit as st
from rag_chatbot import RAGChatbot
import os

# Page config
st.set_page_config(
    page_title="AI Knowledge Assistant",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Improved CSS for light and dark mode support and better layout
st.markdown("""
<style>
    /* Avoid fixed position for input, use sticky instead */
    .stChatInput {
        position: sticky;
        bottom: 1.5rem;
        z-index: 1000;
    }

    .stChatMessage {
        padding: 1rem;
    }

    /* Sidebar gradient */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #4a6fa5 0%, #3a5a8a 100%);
    }

    .sidebar-title {
        color: white !important;
        font-size: 24px !important;
        margin-bottom: 20px !important;
    }

    .sidebar-text {
        color: #f0f2f6 !important;
    }

    /* Chat message boxes - light mode */
    .assistant-message {
        background-color: #f8f9fa;
        color: #000000;  /* black text for visibility */
        border-radius: 15px;
        padding: 12px;
        margin: 5px 0;
    }

    .user-message {
        background-color: #e3f2fd;
        color: #000000;  /* black text for visibility */
        border-radius: 15px;
        padding: 12px;
        margin: 5px 0;
    }

    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
        .assistant-message {
            background-color: #2a2a2a;
            color: #f0f0f0;  /* light text for dark background */
        }
        .user-message {
            background-color: #3a3a3a;
            color: #f0f0f0;  /* light text */
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1e3a5f 0%, #122a45 100%);
        }
        .sidebar-text {
            color: #cccccc !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# Sidebar content
with st.sidebar:
    st.markdown('<p class="sidebar-title">🔍 RAG Chatbot</p>', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-text">Ask questions about AI using Retrieval-Augmented Generation.</p>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    <p class="sidebar-text">
    <b>Technologies:</b><br>
    • Groq (LLaMA3)<br>
    • Amazon Titan Embeddings<br>
    • LangChain<br>
    • FAISS Vector DB
    </p>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<p class="sidebar-text">Made by Gagan Narang</p>', unsafe_allow_html=True)

# Main app interface
st.title("💬 AI Knowledge Assistant")
st.caption("Powered by Groq + Amazon Titan Embeddings")

# Initialize chatbot
if "chatbot" not in st.session_state:
    st.session_state.chatbot = RAGChatbot()
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I'm your AI assistant. Ask me anything about AI concepts!"}
    ]

# Show messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        role_class = "assistant-message" if msg["role"] == "assistant" else "user-message"
        st.markdown(f'<div class="{role_class}">{msg["content"]}</div>', unsafe_allow_html=True)

# User input
if prompt := st.chat_input("Type your question here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(f'<div class="user-message">{prompt}</div>', unsafe_allow_html=True)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching..."):
            response = st.session_state.chatbot.ask(prompt)
        st.markdown(f'<div class="assistant-message">{response}</div>', unsafe_allow_html=True)
        st.session_state.messages.append({"role": "assistant", "content": response})
