import tempfile

import streamlit as st
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter

from config import TMP_DIR


def save_uploaded_files(uploaded_files):
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(
            delete=False, dir=TMP_DIR.as_posix(), suffix=".pdf"
        ) as tmp_file:
            tmp_file.write(uploaded_file.read())


def load_documents():
    loader = DirectoryLoader(TMP_DIR.as_posix(), glob="**/*.pdf", loader_cls=PyPDFLoader)
    try:
        return loader.load()
    except Exception as e:
        st.error(f"Error loading documents: {str(e)}")
        return []


def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separator="\n"
    )
    return text_splitter.split_documents(documents)


def cleanup_temp_files():
    try:
        for file in TMP_DIR.glob("*"):
            file.unlink()
    except Exception as e:
        st.error(f"Error cleaning up files: {str(e)}")
