# streamlit_app.py
import os
import pymupdf as fitz  
import streamlit as st
import tempfile
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load the model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text.strip()

# Function to compute similarity
def compute_similarity(resume_text, job_desc):
    embeddings = model.encode([resume_text, job_desc])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])
    return round(similarity[0][0] * 100, 2)

# Streamlit UI
st.set_page_config(page_title="Resume & JD Matcher", layout="centered")
st.title("📄 Resume vs Job Description Prediction Model:")

st.write("Upload your **Resume (PDF)** and enter the **Job Description** to see how well they match.")

uploaded_file = st.file_uploader("Upload Resume (PDF only)", type=["pdf"])

job_desc = st.text_area("Enter Job Description", height=200)

if st.button("Check Similarity"):
    if uploaded_file is not None and job_desc.strip() != "":
        with st.spinner("Analyzing..."):
            resume_text = extract_text_from_pdf(uploaded_file)
            score = compute_similarity(resume_text, job_desc)
        st.success(f"✅ Similarity Score: **{score}%**")
    else:
        st.error("❌ Please upload a PDF and enter a job description.") 
        
st.markdown(
    """
    <style>
    .stApp{
        background-color: #6B8E23;
    }
    </style>
    """,
    unsafe_allow_html=True
) 
   

