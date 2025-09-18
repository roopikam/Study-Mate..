import streamlit as st
from studymate_backend import StudyMateBackend
import tempfile

backend = StudyMateBackend()

st.set_page_config(page_title="📚 StudyMate - AI Academic Assistant", layout="wide")

st.title("📚 StudyMate - AI Academic Assistant")
st.write("Upload your study PDFs and ask questions based on their content.")

uploaded_files = st.file_uploader("Upload PDF documents", accept_multiple_files=True, type=["pdf"])

if uploaded_files:
    all_text = ""

    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        text = backend.extract_text_from_pdf(tmp_path)
        all_text += text + "\n\n"

    chunks = backend.chunk_text(all_text)
    backend.build_faiss_index(chunks)

    st.success(f"✅ Extracted and indexed {len(chunks)} text chunks from uploaded PDFs.")
