# Author: Gagan Narang
import os
import pandas as pd
from dotenv import load_dotenv
from langchain_community.document_loaders import DataFrameLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_aws import BedrockEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load environment variables
load_dotenv()

class RAGChatbot:
    def __init__(self, dataset_path="data/ai_faq.csv",
                 embedding_model="amazon.titan-embed-text-v2:0",
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
        """Load and split documents from a CSV."""
        try:
            if not os.path.exists(self.dataset_path):
                raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")
            df = pd.read_csv(self.dataset_path)
            loader = DataFrameLoader(df, page_content_column="content")
            documents = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            return splitter.split_documents(documents)
        except Exception as e:
            raise RuntimeError(f"Error loading data: {str(e)}")

    def rag_pipeline(self):
        """Build the complete RAG pipeline."""
        try:
            docs = self.load_data()

            embeddings = BedrockEmbeddings(
                model_id=self.embedding_model,
                region_name=os.getenv("AWS_REGION", "us-east-1"),
                credentials_profile_name=None  # Optional if running on Streamlit Cloud
            )

            self.vector_store = FAISS.from_documents(docs, embeddings)
            self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})

            llm = ChatGroq(
                temperature=self.temperature,
                model_name=self.llm_model,
                groq_api_key=os.getenv("GROQ_API_KEY")
            )

            prompt = ChatPromptTemplate.from_template("""
                Answer the question based only on the following context:
                {context}

                Question: {input}

                Provide a concise, accurate response. If unsure, say you don't know.
            """)

            self.chain = (
                {"context": self.retriever, "input": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
        except Exception as e:
            raise RuntimeError(f"Error initializing RAG pipeline: {str(e)}")

    def ask(self, query):
        """Answer user query using the RAG pipeline."""
        try:
            if not self._initialized:
                self.rag_pipeline()
                self._initialized = True
            return self.chain.invoke(query)
        except Exception as e:
            return f"Error: {str(e)}"
