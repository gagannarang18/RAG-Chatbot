# Author: Gagan Narang
#importing essesntial libraries
import os
from dotenv import load_dotenv
import pandas as pd
from langchain_community.document_loaders import (
    DataFrameLoader, PyPDFLoader, TextLoader, JSONLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_aws import BedrockEmbeddings  # Changed from HuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load environment variables
load_dotenv()

class RAGChatbot:
    def __init__(self, dataset_path="data/ai_faq.csv",
                 embedding_model="amazon.titan-embed-text-v2:0",  # Leveraged the Amazon Titan Embedding model
                 llm_model="llama3-8b-8192", temperature=0.7):   #Groq cloud free tier llama model for text generation
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

    def rag_pipeline(self):
        """Build the complete RAG pipeline using Amazon Titan Embeddings."""
        try:
            docs = self.load_data()

            print("[INFO] Using Amazon Titan Embeddings...")
            embeddings = BedrockEmbeddings(
                model_id=self.embedding_model,
                region_name=os.getenv("AWS_REGION", "us-east-1")
            )

            self.vector_store = FAISS.from_documents(docs, embeddings)
            self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})

            llm = ChatGroq(
                temperature=self.temperature,
                model_name=self.llm_model,
                groq_api_key=os.getenv("GROQ_API_KEY")
            )

            prompt_template = """Answer the question based only on the following context:
            {context}
            
            Question: {input}
            
            Provide a concise, accurate response. If unsure, say you don't know."""
            
            prompt = ChatPromptTemplate.from_template(prompt_template)

            self.chain = (
                {"context": self.retriever, "input": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
            
        except Exception as e:
            raise RuntimeError(f"Failed to build RAG pipeline: {str(e)}")

    def ask(self, query):
        """Answer a user query using the RAG pipeline."""
        try:
            if not self._initialized:
                self.rag_pipeline()
                self._initialized = True
            return self.chain.invoke(query)
        except Exception as e:
            return f"Error processing your query: {str(e)}"
        
# Example usage in terminal
# chatbot = RAGChatbot(embedding_model="amazon.titan-embed-text-v2:0")
# response = chatbot.ask("What is RAG?")
# print(response)
        
        