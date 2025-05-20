import os
from dotenv import load_dotenv
import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load environment variables from .env file
load_dotenv()

class RAGChatbot:
    
     def __init__(self, dataset_path="ai_faq.csv", embedding_model="sentence-transformers/all-mpnet-base-v2", llm_model="llama-3.1-8b-instant", temperature=0.7):
         
        #  initialize the chatbot with the dataset path, embedding model, LLM model, and temperature
         self.dataset_path = dataset_path
         self.embedding_model = embedding_model
         self.llm_model = llm_model
         self.temperature = temperature
         self.vector_store = None
         self.retriever = None
         self.chain = None
    