import os
import faiss
import numpy as np
import json
import tempfile
from tqdm import tqdm
from huggingface_hub import InferenceClient
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import streamlit as st
from datetime import datetime

st.set_page_config(page_title="Financial Assistant", page_icon="💰", layout="wide")

# HuggingFace Token
os.environ["HF_TOKEN"] = "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"

# Initialize models
embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
llm_client = InferenceClient(model=repo_id, timeout=120)

def emb_text(text):
    return embedding_model.encode([text], normalize_embeddings=True).tolist()[0]

# Initialize session states
if 'index' not in st.session_state:
    st.session_state.index = None
if 'data' not in st.session_state:
    st.session_state.data = []
if 'text_lines' not in st.session_state:
    st.session_state.text_lines = []
if 'doc_chat_history' not in st.session_state:
    st.session_state.doc_chat_history = []
if 'finance_chat_history' not in st.session_state:
    st.session_state.finance_chat_history = []

# Chat bubbles and UI
st.markdown("""
<style>
.user-bubble {
    background-color: #e6f3ff;
    padding: 15px;
    border-radius: 15px;
    margin: 5px 0;
    max-width: 80%;
    margin-left: auto;
    margin-right: 10px;
}
.assistant-bubble {
    background-color: #f0f0f0;
    padding: 15px;
    border-radius: 15px;
    margin: 5px 0;
    max-width: 80%;
    margin-left: 10px;
}
.timestamp {
    font-size: 0.8em;
    color: #888;
    margin-top: 5px;
}
.disclaimer {
    font-size: 0.8em;
    color: #ff4b4b;
    padding: 10px;
    border: 1px solid #ff4b4b;
    border-radius: 5px;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

def process_document(file, file_type):
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_type}") as tmp_file:
        tmp_file.write(file.read())
        temp_file_path = tmp_file.name

    loader = PyPDFLoader(temp_file_path) if file_type == "pdf" else Docx2txtLoader(temp_file_path)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    st.session_state.text_lines = [chunk.page_content for chunk in chunks]

    embedding_dim = len(emb_text("This is a test"))
    index = faiss.IndexFlatL2(embedding_dim)

    embeddings = []
    for i, line in enumerate(tqdm(st.session_state.text_lines, desc="Creating embeddings")):
        embedding = emb_text(line)
        embeddings.append(embedding)
        embedding_np = np.array(embedding).astype('float32')
        index.add(np.array([embedding_np]))
        st.session_state.data.append({"id": i, "vector": embedding, "text": line})

    np.save("embeddings.npy", np.array(embeddings))
    st.session_state.index = index
    return True

def get_rag_response(question):
    question_embedding = emb_text(question)
    question_embedding_np = np.array([question_embedding]).astype('float32')
    D, I = st.session_state.index.search(question_embedding_np, 3)

    retrieved_lines = [st.session_state.data[idx]["text"] for idx in I[0]]
    context = "\n".join(retrieved_lines)

    prompt = f"""
    Use the following pieces of information to provide an answer to the question.
    Context:
    {context}

    Question:
    {question}
    """

    return llm_client.text_generation(prompt, max_new_tokens=1000).strip()

def get_finance_response(question):
    prompt = f"""You are a knowledgeable finance AI assistant. Your role is to provide helpful, accurate information about finance-related queries while maintaining appropriate financial ethics. Remember to:

1. Don't include a disclaimer about not being a replacement for professional financial advice.
2. Focus on general finance information, tips, and guidance.
3. Encourage consulting finance professionals for specific financial concerns.
4. Be clear, accurate, and factual in your responses.
5. Avoid making specific financial decisions or directly endorsing particular investments.

Question: {question}

Please provide a helpful response while maintaining these guidelines."""

    response = llm_client.text_generation(prompt, max_new_tokens=1000).strip()
    disclaimer = "\n\nDisclaimer: Please consult a licensed financial advisor or professional for personalized advice if you have specific financial concerns or complex investment decision"
    
    return response + disclaimer

def display_chat_history(chat_history):
    for message in chat_history:
        if message["role"] == "user":
            st.markdown(f"""
                <div class="user-bubble">
                    {message["content"]}
                    <div class="timestamp">{message["timestamp"]}</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="assistant-bubble">
                    {message["content"]}
                    <div class="timestamp">{message["timestamp"]}</div>
                </div>
            """, unsafe_allow_html=True)

# Main interface
st.title("Financial Assistant")

# Tabs for different modes
tab1, tab2 = st.tabs(["💬 Document Chat(RAG Mode)", "Financial Assistant Chat Mode"])

# Document Chat Tab
with tab1:
    st.header("Document Chat")
    
    # Sidebar for document upload
    with st.sidebar:
        st.header("📄 Upload Document")
        uploaded_file = st.file_uploader(
            "Upload PDF or DOCX file",
            type=["pdf", "docx"],
            help="Supported formats: PDF, DOCX"
        )

        if uploaded_file:
            file_type = uploaded_file.name.split('.')[-1].lower()
            with st.spinner("Processing document..."):
                success = process_document(uploaded_file, file_type)
                if success:
                    st.success(f"✅ {uploaded_file.name} processed successfully!")

    # Document chat interface
    chat_container = st.container()
    with chat_container:
        display_chat_history(st.session_state.doc_chat_history)

    if st.session_state.index is not None:
        with st.form(key="doc_chat_form"):
            doc_question = st.text_input("Ask about your document:", key="doc_question_input")
            submit_button = st.form_submit_button("Send")

            if submit_button and doc_question:
                st.session_state.doc_chat_history.append({
                    "role": "user",
                    "content": doc_question,
                    "timestamp": datetime.now().strftime("%H:%M")
                })

                with st.spinner("Thinking..."):
                    answer = get_rag_response(doc_question)

                st.session_state.doc_chat_history.append({
                    "role": "assistant",
                    "content": answer,
                    "timestamp": datetime.now().strftime("%H:%M")
                })
                st.rerun()
    else:
        st.info("👈 Please upload a document to start chatting!")

# Health Assistant Tab
with tab2:
    st.header("Financial Assistant")
    
    st.markdown("""
    <div class="disclaimer"> 
        Please consult financial professionals for specific finance concerns for clarification.
    </div>
    """, unsafe_allow_html=True)
    
    chat_container = st.container()
    with chat_container:
        display_chat_history(st.session_state.finance_chat_history)

    with st.form(key="finance_chat_form"):
        health_question = st.text_input("Ask a finance-related question:", key="finance_question_input")
        submit_button = st.form_submit_button("Send")

        if submit_button and health_question:
            st.session_state.finance_chat_history.append({
                "role": "user",
                "content": health_question,
                "timestamp": datetime.now().strftime("%H:%M")
            })

            with st.spinner("Thinking..."):
                answer = get_finance_response(health_question)

            st.session_state.finance_chat_history.append({
                "role": "assistant",
                "content": answer,
                "timestamp": datetime.now().strftime("%H:%M")
            })
            st.rerun()