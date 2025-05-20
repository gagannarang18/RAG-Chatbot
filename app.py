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

# Custom CSS
st.markdown("""
<style>
    .stChatInput {position: fixed; bottom: 2rem;}
    .stChatMessage {padding: 1rem;}
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
    .assistant-message {
        background-color: #f8f9fa;
        border-radius: 15px;
        padding: 12px;
        margin: 5px 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-radius: 15px;
        padding: 12px;
        margin: 5px 0;
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
