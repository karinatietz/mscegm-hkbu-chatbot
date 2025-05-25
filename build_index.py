# build_index.py
import os
import logging
from dotenv import load_dotenv

# --- RAG Libraries ---
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

from config import full_knowledge_text

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

# --- Secrets & Settings ---
openai_api_key = os.getenv("OPENAI_API_KEY") 
if not openai_api_key:
    logging.error("OPENAI_API_KEY not found in .env file. Please ensure it's set.")
    exit(1)

# OpenAI embedding model name
OPENAI_EMBEDDING_MODEL_NAME = "text-embedding-3-small" 

FAISS_INDEX_PATH = "faiss_index" 

def build_and_save_index():
    """Builds the FAISS index from the knowledge base using OpenAI embeddings and saves it locally."""
    logging.info("Initializing OpenAI Embeddings...")
    try:
        embeddings = OpenAIEmbeddings(
            openai_api_key=openai_api_key,
            model=OPENAI_EMBEDDING_MODEL_NAME
        )
        logging.info(f"OpenAI Embeddings client initialized successfully with model: {OPENAI_EMBEDDING_MODEL_NAME}.")
    except Exception as e:
        logging.error(f"Failed to initialize OpenAI Embeddings client. Check your API key and network: {e}", exc_info=True)
        return

    if not full_knowledge_text:
        logging.error("Source text data (full_knowledge_text) is empty. Cannot build index.")
        return

    logging.info("Preparing documents...")
    docs = [Document(page_content=full_knowledge_text)]
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  
        chunk_overlap=100, 
        length_function=len,
    )
    chunks = text_splitter.split_documents(docs)
    logging.info(f"Split data into {len(chunks)} chunks.")

    if not chunks:
        logging.error("No chunks were created. Cannot build index.")
        return

    try:
        logging.info("Creating FAISS index from documents... This may take a while as embeddings are generated.")
        # This step will call the OpenAI embedding API for all chunks
        db = FAISS.from_documents(chunks, embeddings)
        logging.info("FAISS index created successfully.")

        os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
        
        logging.info(f"Saving FAISS index to: {FAISS_INDEX_PATH}")
        db.save_local(FAISS_INDEX_PATH)
        logging.info("FAISS index saved successfully.")
        logging.info(f"Index files saved: {FAISS_INDEX_PATH}/index.faiss, {FAISS_INDEX_PATH}/index.pkl")

    except Exception as e:
        logging.error(f"Error creating or saving FAISS index: {e}", exc_info=True)

if __name__ == "__main__":
    build_and_save_index()