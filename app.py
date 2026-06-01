import streamlit as st
import os
from dotenv import load_dotenv

from utils.pdf_loader import extract_text_from_pdf
from utils.chunking import get_text_chunks
from utils.embeddings import get_vector_store
from utils.retrieval import generate_answer

# Load environment variables
load_dotenv()

st.set_page_config(page_title="DocuMind AI", page_icon="📄", layout="wide")

st.title("DocuMind AI – Context-Aware Document Q&A Bot")
st.markdown("Upload a PDF or paste text, and ask questions based on the provided content.")

# Sidebar for inputs
with st.sidebar:
    st.header("Document Input")
    
    upload_option = st.radio("Choose input method:", ("PDF Upload", "Paste Text"))
    
    raw_text = ""
    
    if upload_option == "PDF Upload":
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True, type=["pdf"])
        if st.button("Process PDF"):
            with st.spinner("Extracting text..."):
                for pdf in pdf_docs:
                    raw_text += extract_text_from_pdf(pdf)
                if raw_text:
                    st.session_state["raw_text"] = raw_text
                    st.success("PDF processed successfully!")
                else:
                    st.error("No text could be extracted.")
    else:
        pasted_text = st.text_area("Paste your text here", height=200)
        if st.button("Process Text"):
            if pasted_text.strip():
                st.session_state["raw_text"] = pasted_text
                st.success("Text processed successfully!")
            else:
                st.error("Please paste some text.")
                
    if "raw_text" in st.session_state and st.session_state["raw_text"]:
        if st.button("Build Knowledge Base"):
            with st.spinner("Processing chunks and generating embeddings..."):
                text_chunks = get_text_chunks(st.session_state["raw_text"])
                vector_store = get_vector_store(text_chunks)
                st.session_state["vector_store"] = vector_store
                st.success("Knowledge base built successfully! You can now ask questions.")

# Main area for Q&A
st.header("Ask a Question")
user_question = st.text_input("Enter your question based on the document:")

if st.button("Ask"):
    if not user_question:
        st.warning("Please enter a question.")
    elif "vector_store" not in st.session_state:
        st.warning("Please process a document and build the knowledge base first.")
    else:
        with st.spinner("Searching for answer..."):
            answer = generate_answer(user_question, st.session_state["vector_store"])
            st.subheader("Answer:")
            st.write(answer)
