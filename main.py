# main.py
import sys
import streamlit as st
import requests
import json
import os
import logging
import time
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document


# --- Load Data from Config ---
from config import full_knowledge_text

# --- Logging Configuration ---
log_level = logging.DEBUG
log_format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'

logging.basicConfig(level=log_level, format=log_format, stream=sys.stdout)
logger = logging.getLogger(__name__)


# --- Configuration ---
# Use OpenAI API Key from Streamlit secrets
openai_api_key = st.secrets.get("OPENAI_API_KEY")
openai_chat_model_name = "gpt-4o-mini"  
openai_embedding_model_name = "text-embedding-3-small" 

FAISS_INDEX_PATH = "faiss_index" 

# --- Helper function to initialize Vector Store ---
@st.cache_resource(show_spinner="Loading Knowledge Base...")
def initialize_vector_store(_embedding_function):
    logger.info(f"Attempting to load FAISS index from path: {FAISS_INDEX_PATH}")
    vectorstore = None

    if not os.path.exists(FAISS_INDEX_PATH) or \
       not os.path.exists(os.path.join(FAISS_INDEX_PATH, "index.faiss")) or \
       not os.path.exists(os.path.join(FAISS_INDEX_PATH, "index.pkl")):
        error_msg = f"FAISS index files not found in directory '{FAISS_INDEX_PATH}'. " \
                    "Please ensure 'index.faiss' and 'index.pkl' were created and committed to the repository."
        logger.error(error_msg)
        st.error(error_msg)
        st.stop()
        return None

    try:
        vectorstore = FAISS.load_local(
            folder_path=FAISS_INDEX_PATH,
            embeddings=_embedding_function,
            allow_dangerous_deserialization=True
        )
        logger.info(f"Successfully loaded FAISS index from {FAISS_INDEX_PATH}")
    except Exception as e:
        logger.exception(f"Error loading FAISS index from {FAISS_INDEX_PATH}: {e}")
        st.error(f"Failed to load the Knowledge Base from local files: {e}")
        st.stop()
        return None

    return vectorstore

# --- Streamlit App UI ---
st.set_page_config(page_title="HKBU MScEGM Chatbot", page_icon="🎓")
st.title("🎓 HKBU Master of Science in Entrepreneurship and Global Marketing FAQ")
st.markdown("Ask me anything about the **HKBU MSc in Entrepreneurship and Global Marketing** program! (Powered by OpenAI API & FAISS)")

def clear_chat_history():
    logger.info("Clearing chat history.")
    st.session_state.messages = [{"role": "assistant", "content": "Hi! Ask me about the HKBU MSc in Entrepreneurship and Global Marketing program."}]

st.button('Clear Chat History', on_click=clear_chat_history)

# --- Chatbot Logic ---
if not openai_api_key:
    st.info("Please add your **OpenAI API key** to Streamlit secrets (key: `OPENAI_API_KEY`) to continue.", icon="🗝️")
    logger.warning("OpenAI API key not found in secrets.")
    st.stop()
else:
    # *** Initialize OpenAI Embeddings ***
    try:
        embeddings = OpenAIEmbeddings(
            openai_api_key=openai_api_key,
            model=openai_embedding_model_name
        )
        logger.info(f"OpenAI Embeddings client initialized successfully with model: {openai_embedding_model_name}.")
    except Exception as e:
        logger.exception("Failed to initialize OpenAI Embeddings client.")
        st.error(f"Failed to initialize OpenAI Embeddings service. Please check your API key and network connection: {e}")
        st.stop()

    # Initialize OpenAI Chat Model
    try:
        llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model_name=openai_chat_model_name,
            temperature=0.2 
        )
        logger.info(f"OpenAI Chat Model initialized successfully with model: {openai_chat_model_name}.")
    except Exception as e:
        logger.exception("Failed to initialize OpenAI Chat Model.")
        st.error(f"Failed to initialize OpenAI Chat service. Please check your API key and network connection: {e}")
        st.stop()


    # --- Load FAISS Vector Store from local files ---
    vectorstore = initialize_vector_store(embeddings) # Pass embeddings object

    if vectorstore is None:
        logger.error("FAISS vector store loading failed. Stopping execution.")
        st.stop()

    # --- Initialize chat history ---
    if "messages" not in st.session_state:
        logger.info("Initializing chat history session state.")
        st.session_state.messages = [{"role": "assistant", "content": "Hi! Ask me anything about the **HKBU Master of Science in Entrepreneurship and Global Marketing** program."}]

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- Handle user input ---
    if prompt := st.chat_input("e.g., What are the admission requirements?"):
        logger.info(f"User input received: '{prompt}'")
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            thinking_message = st.empty()
            thinking_message.markdown("Thinking... *(Accessing knowledge base & generating response)*")
            with st.spinner("Processing your request..."):
                # --- RAG Step: Retrieve ---
                context = "Error during context retrieval."
                retrieved_docs_content = []
                try:
                    logger.info(f"Retrieving relevant documents for query: '{prompt}' from FAISS index.")
                    if vectorstore:
                        retriever = vectorstore.as_retriever(
                            search_type="similarity",
                            search_kwargs={"k": 10} 
                        )
                        retrieved_docs = retriever.invoke(prompt)
                        retrieved_docs_content = [doc.page_content for doc in retrieved_docs]

                        if not retrieved_docs_content:
                            logger.warning("No relevant documents found in the FAISS knowledge base for the query.")
                            context = "No specific information found in the knowledge base for this query."
                        else:
                            context = "\n\n---\n\n".join(retrieved_docs_content)
                            logger.info(f"Retrieved {len(retrieved_docs)} documents from FAISS index.")
                            logger.debug(f"Retrieved context:\n{context[:500]}...")
                    else:
                         logger.error("Vectorstore object (FAISS) is invalid/None. Cannot retrieve documents.")
                         context = "Error: Failed to access the knowledge base (invalid vectorstore)."

                except Exception as e:
                    logger.exception("Error retrieving documents from FAISS vector store.")
                    st.error(f"Error retrieving information from knowledge base files: {e}")
                    context = "Error: Failed to access the knowledge base files."

                # --- RAG Step: Augment Prompt ---
                system_message_content = """You are an AI assistant for the **Master of Science in Entrepreneurship and Global Marketing (MSc EGM)** program at **Hong Kong Baptist University (HKBU)**.
                - Answer the user's question **ONLY** based on the provided context below.
                - Be concise, direct, and factual.
                - If the context doesn't contain the answer, state clearly and politely that the information is not available in the provided documents, or suggest where they might find more information (e.g., "Please refer to the official HKBU MSc EGM program website for the most up-to-date details.").
                - Do not make up information or use external knowledge beyond what is explicitly in the context.
                - Quote specific program details, admission requirements, curriculum points, or faculty information from the context when relevant.
                - Maintain a helpful and informative tone.
                - If the context indicates an error occurred during retrieval, inform the user politely that you couldn't access the necessary information from the knowledge base.
                """
                messages_for_api = [
                    {"role": "system", "content": system_message_content},
                    {"role": "user", "content": f"Based on the following information:\n\nContext:\n---\n{context}\n---\n\nQuestion: {prompt}"}
                ]

                # --- RAG Step: Generate using LangChain ChatOpenAI ---
                response_content = "Sorry, I encountered an error processing your request."
                try:
                    logger.info(f"Sending request to OpenAI Chat API for response generation.")
                    # Use llm.invoke() from LangChain's ChatOpenAI
                    ai_message = llm.invoke(messages_for_api)
                    response_content = ai_message.content
                    logger.info("Successfully received response from OpenAI Chat API.")

                except Exception as e:
                    logger.exception(f"Error calling OpenAI Chat API: {e}")
                    st.error(f"An error occurred while generating a response from OpenAI: {e}. Please check your API key or try again.")
                    response_content = "Sorry, I couldn't generate a response due to an issue with the AI service."


            thinking_message.empty()
            st.markdown(response_content)
            logger.info(f"Assistant final response displayed.")

        st.session_state.messages.append({"role": "assistant", "content": response_content})

# --- End of App Logic ---