import os
import time

import streamlit as st
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import ServerlessSpec, Pinecone as PineconeClient


def get_pinecone_client():
    return PineconeClient(api_key=os.getenv("PINECONE_API_KEY"))


def get_embeddings():
    return OpenAIEmbeddings(
        openai_api_key=os.getenv("OPEN_ROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        model="text-embedding-3-small",
    )


def create_vectorstore(texts):
    try:
        embeddings = get_embeddings()
        pc = get_pinecone_client()
        index_name = os.getenv("PINECONE_INDEX")

        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            time.sleep(20)

        store = PineconeVectorStore(
            index=pc.Index(index_name), embedding=embeddings
        )
        vs = store.from_documents(texts, embeddings, index_name=index_name)
        return vs.as_retriever()
    except Exception as e:
        st.error(f"Error creating vectorstore: {str(e)}")
        return None


def clear_vector_index():
    try:
        pc = get_pinecone_client()
        index_name = os.getenv("PINECONE_INDEX")
        pc.Index(index_name).delete(delete_all=True)
        st.success("Vector index cleared successfully.")
    except Exception as e:
        st.error(f"Error clearing index: {str(e)}")
