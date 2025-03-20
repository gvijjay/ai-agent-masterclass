import json
import streamlit as st
from typing import TypedDict
from langgraph.graph import StateGraph
import openai
import os
from dotenv import load_dotenv
from io import BytesIO
from docx import Document
from pptx import Presentation
import pdfplumber
import pandas as pd
from bs4 import BeautifulSoup
import numpy as np
import faiss
import pypandoc
import mimetypes
import tempfile

# Load environment variables
load_dotenv()
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Define a state schema
class DocumentState(TypedDict):
    uploaded_file: BytesIO
    file_type: str
    summary: str
    chunks: list
    message: str

# Initialize LangGraph with a state schema
graph = StateGraph(state_schema=DocumentState)

# File extraction functions
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(BytesIO(file.read())) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text.strip()

def extract_text_from_doc(file, file_type):
    if file_type == "docx":
        doc = Document(BytesIO(file.read()))
        return "\n".join([para.text for para in doc.paragraphs])
    elif file_type == "doc":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".doc") as temp_doc:
            temp_doc.write(file.read())
            temp_doc_path = temp_doc.name
        temp_docx_path = temp_doc_path + "x"
        try:
            pypandoc.convert_file(temp_doc_path, "docx", outputfile=temp_docx_path)
            doc = Document(temp_docx_path)
            return "\n".join([para.text for para in doc.paragraphs])
        finally:
            os.remove(temp_doc_path)
            if os.path.exists(temp_docx_path):
                os.remove(temp_docx_path)
    else:
        return ""

def extract_text_from_pptx(file):
    prs = Presentation(BytesIO(file.read()))
    return "\n".join([
        shape.text for slide in prs.slides for shape in slide.shapes if hasattr(shape, "text")
    ]).strip()

def extract_text_from_txt(file):
    return file.read().decode("utf-8").strip()

def extract_text_from_html(file):
    soup = BeautifulSoup(file, "html.parser")
    return soup.get_text().strip()

def extract_text_from_any_file(uploaded_file):
    file_type, _ = mimetypes.guess_type(uploaded_file.name)
    
    extractors = {
        "application/pdf": extract_text_from_pdf,
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": extract_text_from_doc,
        "application/msword": extract_text_from_doc,
        "text/plain": extract_text_from_txt,
        "text/html": extract_text_from_html,
        "application/vnd.ms-powerpoint": extract_text_from_pptx,
        "application/vnd.openxmlformats-officedocument.presentationml.presentation": extract_text_from_pptx,
        "application/json": lambda file: json.load(file),
        "text/csv": lambda file: pd.read_csv(file).to_string(),
    }
    
    if file_type in extractors:
        return extractors[file_type](uploaded_file, uploaded_file.name.split('.')[-1].lower())
    return "Unsupported file type. Please upload a document or readable file."

def chunk_text(text, chunk_size=500):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def encode_text(text):
    response = openai.Embedding.create(
        model="text-embedding-ada-002",  # Using OpenAI embeddings
        input=text
    )
    return np.array(response['data'][0]['embedding'], dtype=np.float32)


def query_document(state: DocumentState, query: str):
    chunks = state["chunks"]
    
    if not chunks:
        return {"answer": "No document chunks available for querying.", "message": "Error in document processing."}

    # Encode text chunks
    embeddings = [encode_text(chunk) for chunk in chunks if chunk]
    
    if not embeddings:
        return {"answer": "Unable to create valid embeddings.", "message": "Error in embeddings creation."}

    embeddings = np.array(embeddings, dtype=np.float32)
    dimension = embeddings.shape[1]
    
    # Create FAISS index
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Search for the most relevant chunk
    query_embedding = encode_text(query)
    if query_embedding is None or query_embedding.size == 0:
        return {"answer": "Failed to generate query embedding.", "message": "Error in query embedding."}

    D, I = index.search(np.array([query_embedding]), 3)
    relevant_chunks = [chunks[i] for i in I[0]]
    context = " ".join(relevant_chunks)
    
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Answer this question based on the document:\n\n{context}\n\nQuestion: {query}"}
        ],
        max_tokens=1000
    )
    
    return {"answer": response['choices'][0]['message']['content'].strip(), "message": "Query answered successfully!"}

def summarize_large_text(text, chunk_size=2000):
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    summaries = []
    for chunk in chunks:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Summarize the following document:\n\n{chunk}"}
            ],
            max_tokens=500
        )
        summaries.append(response['choices'][0]['message']['content'].strip())
    return " ".join(summaries)

def process_document(state: DocumentState):
    uploaded_file = state["uploaded_file"]
    text = extract_text_from_any_file(uploaded_file)
    if not text:
        return {"summary": "", "chunks": [], "message": "No text extracted from the document."}
    chunks = chunk_text(text)
    summary = summarize_large_text(text)
    return {"summary": summary, "chunks": chunks, "message": "Document processed successfully!"}

def upload_document(state: DocumentState, uploaded_file: BytesIO):
    state.update({"uploaded_file": uploaded_file})
    return process_document(state)

def query_document_function(state: DocumentState, query: str):
    return query_document(state, query)

def main():
    st.title("Document Upload and Query System with LangGraph")
    if "chunks" not in st.session_state:
        st.session_state["chunks"] = []
    
    uploaded_file = st.file_uploader("Upload a Word Document", type=None)
    if uploaded_file:
        result = upload_document({}, uploaded_file)
        st.subheader("Document Summary")
        st.write(result["summary"])
        st.session_state["chunks"] = result["chunks"]

    query = st.text_input("Ask a question about the document:")
    if query:
        query_result = query_document_function({"chunks": st.session_state["chunks"]}, query)
        st.subheader("Answer")
        st.write(query_result["answer"])

if __name__ == "__main__":
    main()
