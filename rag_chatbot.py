# rag_chatbot.py

import os
from dotenv import load_dotenv
import pandas as pd

from langchain_community.document_loaders import (
    DataFrameLoader, PyPDFLoader, TextLoader, JSONLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load environment variables
load_dotenv()

class RAGChatbot:
    def __init__(self, dataset_path="data/ai_faq.csv",
                 embedding_model="sentence-transformers/all-mpnet-base-v2",
                 llm_model="llama3-8b-8192", temperature=0.7):
        self.dataset_path = dataset_path
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.temperature = temperature
        self.vector_store = None
        self.retriever = None
        self.chain = None

    def load_data(self):
        """Load and split documents from supported formats."""
        print("[INFO] Loading data...")
        if self.dataset_path.endswith('.csv'):
            df = pd.read_csv(self.dataset_path)
            loader = DataFrameLoader(df, page_content_column="content")
        elif self.dataset_path.endswith('.json'):
            loader = JSONLoader(file_path=self.dataset_path, jq_schema='.[]', text_content=False)
        elif self.dataset_path.endswith('.txt'):
            loader = TextLoader(self.dataset_path)
        elif self.dataset_path.endswith('.pdf'):
            loader = PyPDFLoader(self.dataset_path)
        else:
            raise ValueError(f"Unsupported file type: {self.dataset_path}")

        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return splitter.split_documents(documents)

    def rag_pipeline(self):
        """Build the RAG pipeline: Embed, retrieve, query."""
        docs = self.load_data()

        print("[INFO] Using HuggingFace Inference API for embeddings...")
        embeddings = HuggingFaceInferenceAPIEmbeddings(
            api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
            model_name=self.embedding_model
        )

        self.vector_store = FAISS.from_documents(docs, embeddings)
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})

        llm = ChatGroq(
            temperature=self.temperature,
            model=self.llm_model,
            streaming=False
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant. Use the context to answer user questions."),
            ("user", "{input}")
        ])

        self.chain = prompt | self.retriever | llm | StrOutputParser() | RunnablePassthrough()

    def ask(self, query):
        """Answer a user query using the RAG pipeline."""
        if not self.chain:
            self.rag_pipeline()
        return self.chain.invoke({"input": query})
