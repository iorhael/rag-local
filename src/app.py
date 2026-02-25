import streamlit as st

from config import validate_api_keys
from vectorstore import clear_vector_index
from chat import run_simple_chat, run_rag_chat

st.set_page_config(page_title="RAG System")
st.title("👾 AI Assistant with RAG options")
st.write("Upload your documents or just chat with friendly bot.")


def init_session_state():
    defaults = {
        "messages": [],
        "retriever": None,
        "mode": "Simple Chat",
        "documents_ready": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_sidebar():
    with st.sidebar:
        new_mode = st.radio("Select Mode", ["Simple Chat", "RAG Chat"], index=0)

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


def render_chat_history():
    for human, ai in st.session_state.messages:
        st.chat_message("human").write(human)
        st.chat_message("assistant").write(ai)


def main():
    init_session_state()
    render_sidebar()

    if not validate_api_keys():
        return

    render_chat_history()

    if st.session_state.mode == "Simple Chat":
        run_simple_chat()
    else:
        run_rag_chat()


if __name__ == "__main__":
    main()
