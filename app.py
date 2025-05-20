import streamlit as st
from rag_chatbot import RAGChatbot
import os

# Fallback to Streamlit secrets if environment variables are missing
groq_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
aws_id = os.getenv("AWS_ACCESS_KEY_ID") or st.secrets.get("AWS_ACCESS_KEY_ID")
aws_secret = os.getenv("AWS_SECRET_ACCESS_KEY") or st.secrets.get("AWS_SECRET_ACCESS_KEY")

# Page config - must be first Streamlit command
st.set_page_config(
    page_title="AI Knowledge Assistant",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
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

# Sidebar with info
with st.sidebar:
    st.markdown('<p class="sidebar-title">🔍 RAG Chatbot</p>', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-text">Ask questions about AI concepts using Retrieval-Augmented Generation</p>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    <p class="sidebar-text">
    <b>Technologies:</b><br>
    • Groq (Llama3-8B)<br>
    • Amazon Titan Embeddings<br>
    • LangChain RAG Pipeline<br>
    • FAISS Vector Store
    </p>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<p class="sidebar-text">Created by Gagan Narang</p>', unsafe_allow_html=True)

# Main chat interface
st.title("💬 AI Knowledge Assistant")
st.caption("Powered by Groq's ultra-fast LLM inference")

# Initialize chatbot
if "chatbot" not in st.session_state:
    st.session_state.chatbot = RAGChatbot()
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I'm your AI assistant. Ask me anything about AI concepts!"}
    ]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            st.markdown(f'<div class="assistant-message">{message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="user-message">{message["content"]}</div>', unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Type your question here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(f'<div class="user-message">{prompt}</div>', unsafe_allow_html=True)
    
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching knowledge base..."):
            response = st.session_state.chatbot.ask(prompt)
        st.markdown(f'<div class="assistant-message">{response}</div>', unsafe_allow_html=True)
        st.session_state.messages.append({"role": "assistant", "content": response})