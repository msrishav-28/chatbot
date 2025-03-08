import streamlit as st
from groq import Groq
from pinecone import Pinecone, ServerlessSpec
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
import PyPDF2
import pickle
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("chatbot.log"),  # Save logs to file
        logging.StreamHandler()  # Print logs to console
    ]
)
logger = logging.getLogger(__name__)

# Set Streamlit page config
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-west-2")  # Default to 'us-west-2' for India

# Validate API keys
if not GROQ_API_KEY or not PINECONE_API_KEY:
    st.error("❌ API keys missing! Set GROQ_API_KEY and PINECONE_API_KEY in your .env file.")
    st.stop()

# Initialize Groq client
groq_client = Groq(api_key=GROQ_API_KEY)

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "chatbot-index"
if index_name not in pc.list_indexes().names():
    logger.info(f"Creating new Pinecone index: {index_name}")
    pc.create_index(
        name=index_name,
        dimension=384,  # Ensure this matches your embedding model
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
    )
index = pc.Index(index_name)

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Functions for handling chat sessions
def save_chat_sessions(sessions):
    with open("chat_sessions.pkl", "wb") as f:
        pickle.dump(sessions, f)

def load_chat_sessions():
    try:
        with open("chat_sessions.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return {}

# File processing function
def process_file(file):
    logger.info(f"Processing file: {file.name}")
    if file.name.endswith(".pdf"):
        pdf_reader = PyPDF2.PdfReader(file)
        text = "".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
    else:
        text = file.read().decode("utf-8")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(text)
    
    for i, chunk in enumerate(chunks):
        embedding = embeddings.embed_query(chunk)
        index.upsert([(f"{file.name}_chunk_{i}", embedding, {"text": chunk, "file_name": file.name})])
    logger.info(f"Upserted {len(chunks)} chunks to Pinecone for {file.name}")
    return len(chunks)

# Query Pinecone
def retrieve_chunks(query, top_k=3):
    query_embedding = embeddings.embed_query(query)
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    retrieved_chunks = [(match["metadata"]["text"], match["metadata"].get("file_name", "Unknown")) 
                        for match in results["matches"]]
    return retrieved_chunks

# Generate response using Groq
def generate_response(query, context, model):
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    response_stream = groq_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        stream=True
    )
    return response_stream

# Streamlit UI
st.title("🤖 RAG Chatbot with Groq & Pinecone")

if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = load_chat_sessions()

st.session_state.selected_model = st.selectbox("Select Model", [
    "mixtral-8x7b-32768", "llama2-70b-4096", "llama3-8b-8192", "llama3-70b-8192",
    "gemma-7b-it", "deepseek-r1-distill-llama-70b"
])

prompt = st.chat_input("Ask me anything!")

if prompt:
    st.session_state.chat_sessions.setdefault("latest_chat", []).append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
    
    retrieved_chunks = retrieve_chunks(prompt)
    context = "\n".join(chunk for chunk, _ in retrieved_chunks)
    
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking..."):
            if not context:
                response_text = "I don’t have enough context. Please upload a document!"
            else:
                response_stream = generate_response(prompt, context, st.session_state.selected_model)
                response_placeholder = st.empty()
                response_text = ""
                for chunk in response_stream:
                    content = chunk.choices[0].delta.content
                    if content:
                        response_text += content
                        response_placeholder.markdown(response_text + "▌")
                response_placeholder.markdown(response_text)
            st.session_state.chat_sessions["latest_chat"].append({"role": "assistant", "content": response_text})
    save_chat_sessions(st.session_state.chat_sessions)

with st.sidebar:
    st.header("📂 Document Upload")
    uploaded_file = st.file_uploader("Upload PDF/TXT", type=["pdf", "txt"])
    if uploaded_file:
        with st.spinner("Processing..."):
            process_file(uploaded_file)
            st.success(f"{uploaded_file.name} processed successfully!")
