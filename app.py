import os
from pathlib import Path
from dotenv import load_dotenv
import tempfile
import time

# Vector store and embedding imports
from langchain_pinecone import PineconeVectorStore
from langchain_openai.embeddings import OpenAIEmbeddings

# Document processing imports
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter

# LLM and chain imports
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI

from pinecone import ServerlessSpec, Pinecone as PineconeClient

import streamlit as st

# Load env variables
load_dotenv()

# Set up directory structure
TMP_DIR = Path(__file__).resolve().parent.joinpath('data', 'tmp')
TMP_DIR.mkdir(parents=True, exist_ok=True)

# Streamlit page configuration
st.set_page_config(page_title="RAG System")
st.title("📚 Document AI Assistant")
st.write("Upload your documents or just chat with friendly bot.")

def validate_api_keys() -> bool:
    if not os.getenv("OPEN_ROUTER_API_KEY"):
        st.error("OPEN_ROUTER_API_KEY not found in .env")
        return False

    if not os.getenv("PINECONE_API_KEY"):
        st.error("PINECONE_API_KEY not found in .env")
        return False

    if not os.getenv("PINECONE_INDEX"):
        st.error("PINECONE_INDEX not found in .env")
        return False

    return True

def cleanup_temp_files():
    """Clean up temporary files"""
    try:
        for file in TMP_DIR.glob('*'):
            file.unlink()
    except Exception as e:
        st.error(f"Error cleaning up files: {str(e)}")

def clear_vector_index():
    try:
        pc = PineconeClient(api_key=os.getenv("PINECONE_API_KEY"))
        index_name = os.getenv("PINECONE_INDEX")

        index = pc.Index(index_name)

        # Удаляем все векторы
        index.delete(delete_all=True)

        st.success("Vector index cleared successfully.")

    except Exception as e:
        st.error(f"Error clearing index: {str(e)}")

def load_documents():
    """Load documents from temp directory"""
    loader = DirectoryLoader(TMP_DIR.as_posix(), glob='**/*.pdf', loader_cls=PyPDFLoader)
    try:
        documents = loader.load()
        return documents
    except Exception as e:
        st.error(f"Error loading documents: {str(e)}")
        return []

def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """Split documents into chunks with overlap"""
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator="\n"
    )
    return text_splitter.split_documents(documents)

def create_vectorstore(texts):
    """Create Pinecone vectorstore"""
    try:
        embeddings = OpenAIEmbeddings(
            openai_api_key=os.getenv("OPEN_ROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            model="text-embedding-3-small",
        )

        pc = PineconeClient(api_key=os.getenv("PINECONE_API_KEY"))

        # Create index if it doesn't exist
        if os.getenv("PINECONE_INDEX") not in pc.list_indexes().names():
            pc.create_index(
                name=os.getenv("PINECONE_INDEX"),
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            # Wait for index to be ready
            time.sleep(20)

        vector_store = PineconeVectorStore(
            index=pc.Index(os.getenv("PINECONE_INDEX")),
            embedding=embeddings
        )

        vs = vector_store.from_documents(texts, embeddings, index_name=os.getenv("PINECONE_INDEX"))
        return vs.as_retriever()

    except Exception as e:
        st.error(f"Error creating vectorstore: {str(e)}")
        return None

def run_simple_chat():
    if query := st.chat_input("Ask something..."):
        st.chat_message("human").write(query)

        llm = ChatOpenAI(
            openai_api_key=os.getenv("OPEN_ROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            temperature=0.7,
            model="gpt-3.5-turbo"
        )

        messages = []
        for human, ai in st.session_state.messages:
            messages.append(("human", human))
            messages.append(("ai", ai))

        response = llm.invoke(messages + [("human", query)])
        answer = response.content

        st.session_state.messages.append((query, answer))
        st.chat_message("assistant").write(answer)

def run_rag_chat():
    if not st.session_state.documents_ready:

        uploaded_files = st.file_uploader(
            "Upload Documents",
            type=["pdf"],
            accept_multiple_files=True
        )

        if st.button("Process Documents"):
            if not uploaded_files:
                st.warning("Please upload documents first.")
                return

            with st.status("Processing documents...", expanded=True) as status:
                for uploaded_file in uploaded_files:
                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        dir=TMP_DIR.as_posix(),
                        suffix=".pdf"
                    ) as tmp_file:
                        tmp_file.write(uploaded_file.read())

                status.write("Loading documents...")
                documents = load_documents()

                status.write("Splitting documents...")
                texts = split_documents(documents)

                status.write("Creating embeddings...")
                st.session_state.retriever = create_vectorstore(texts)

                cleanup_temp_files()

                status.update(label="Documents ready!", state="complete")

            st.session_state.documents_ready = True
            st.rerun()

    if st.session_state.documents_ready:

        if query := st.chat_input("Ask about your documents..."):
            st.chat_message("human").write(query)
            answer = query_llm(st.session_state.retriever, query)
            st.chat_message("assistant").write(answer)

def query_llm(retriever, query):
    """Process queries using the retrieval chain"""
    try:
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(
                openai_api_key=os.getenv("OPEN_ROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1",
                temperature=0.7,
                model="gpt-3.5-turbo"
            ),
            retriever=retriever,
            return_source_documents=True
        )

        result = qa_chain({
            'question': query,
            'chat_history': st.session_state.messages
        })

        st.session_state.messages.append((query, result['answer']))
        return result['answer']
    except Exception as e:
        st.error(f"Error processing query: {str(e)}")
        return "Error processing your question. Please try again."

def main():
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    if "mode" not in st.session_state:
        st.session_state.mode = "Simple Chat"
    if "documents_ready" not in st.session_state:
        st.session_state.documents_ready = False


    # --- Sidebar ---
    with st.sidebar:
        new_mode = st.radio(
            "Select Mode",
            ["Simple Chat", "RAG Chat"],
            index=0
        )

        if new_mode != st.session_state.mode:
            clear_vector_index()
            st.session_state.mode = new_mode
            st.session_state.messages = []
            st.session_state.retriever = None
            st.session_state.documents_ready = False
            st.rerun()

        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()

        if st.button("Clear Vector Index"):
            clear_vector_index()
            st.session_state.retriever = None
            st.session_state.documents_ready = False
            st.session_state.messages = []
            st.rerun()

    # Validate keys before processing
    if not validate_api_keys():
        return

    for human, ai in st.session_state.messages:
        st.chat_message("human").write(human)
        st.chat_message("assistant").write(ai)

    # --- Mode switch ---
    if st.session_state.mode == "Simple Chat":
        run_simple_chat()
    else:
        run_rag_chat()

if __name__ == '__main__':
    main()
