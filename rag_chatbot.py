# Author: Gagan Narang
#importing neccessary libraries
import os
from dotenv import load_dotenv
import pandas as pd
from getpass import getpass
from langchain_community.document_loaders import (
    DataFrameLoader, PyPDFLoader, TextLoader, JSONLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load environment variables
load_dotenv()

#defining the RAGChatbot class
class RAGChatbot:
    def __init__(self, dataset_path="data/ai_faq.csv",
                 embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                 llm_model="llama3-8b-8192", temperature=0.7):
        self.dataset_path = dataset_path
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.temperature = temperature
        self.vector_store = None
        self.retriever = None
        self.chain = None
        self._initialized = False
        
    def load_data(self):    
        """Load and split documents from supported formats."""
        print("[INFO] Loading data...")
        try:
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
        
        except Exception as e:
            raise RuntimeError(f"Failed to load data: {str(e)}")

    def initialize(self):
        """Initialize the RAG pipeline (lazy initialization)"""
        if not self._initialized:
            self.rag_pipeline()
            self._initialized = True

    def rag_pipeline(self):
        """Build the complete RAG pipeline using Groq LLM and Hugging Face Inference API for embeddings."""
        try:
            # Step 1: Load and split documents
            docs = self.load_data()

            # Step 2: Load Hugging Face API token
            print("[INFO] Using HuggingFace Inference API for embeddings...")
            hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
            if not hf_token:
                from getpass import getpass
                print("[WARNING] HuggingFace API token not found in environment. Prompting manually...")
                hf_token = getpass("Enter your HF Inference API Token: ")

            # Step 3: Ensure correct model casing (Hugging Face model names are case-sensitive)
            if self.embedding_model.lower() == "sentence-transformers/all-minilm-l6-v2":
                self.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"

            # Step 4: Create embedding model using HF Inference API
            embeddings = HuggingFaceInferenceAPIEmbeddings(
                api_key=hf_token,
                model_name=self.embedding_model
            )

            # Step 5: Debug: test embedding API call
            try:
                _ = embeddings.embed_query("test query")
            except Exception as embed_err:
                raise RuntimeError(f"[ERROR] Embedding model failed: {embed_err}")

            # Step 6: Create FAISS vector store
            self.vector_store = FAISS.from_documents(docs, embeddings)
            self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})

            # Step 7: Initialize Groq LLM
            groq_key = os.getenv("GROQ_API_KEY")
            if not groq_key:
                raise RuntimeError("GROQ_API_KEY not found in environment variables.")
                
            llm = ChatGroq(
                temperature=self.temperature,
                model_name=self.llm_model,
                groq_api_key=groq_key
            )

            # Step 8: Prompt template for retrieval
            prompt_template = """Answer the question based only on the following context:
            {context}
            
            Question: {input}
            
            Provide a concise, accurate response. If unsure, say you don't know."""

            prompt = ChatPromptTemplate.from_template(prompt_template)

            # Step 9: Assemble RAG chain
            self.chain = (
                {"context": self.retriever, "input": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )

            print("[INFO] RAG pipeline successfully initialized.")

        except Exception as e:
            raise RuntimeError(f"Failed to build RAG pipeline: {str(e)}")


    def ask(self, query):
        """Answer a user query using the RAG pipeline."""
        try:
            if not self._initialized:
                self.initialize()
            return self.chain.invoke(query)
        except Exception as e:
            return f"Error processing your query: {str(e)}"