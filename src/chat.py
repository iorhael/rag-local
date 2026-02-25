import streamlit as st

from documents import (
    save_uploaded_files,
    load_documents,
    split_documents,
    cleanup_temp_files,
)
from vectorstore import create_vectorstore
from config import SYSTEM_PROMPT
from llm import create_llm, query_llm


def run_simple_chat():
    if query := st.chat_input("Ask something..."):
        st.chat_message("human").write(query)

        llm = create_llm()

        messages = [("system", SYSTEM_PROMPT)]
        for human, ai in st.session_state.messages:
            messages.append(("human", human))
            messages.append(("ai", ai))

        response = llm.invoke(messages + [("human", query)])
        answer = response.content

        st.session_state.messages.append((query, answer))
        st.chat_message("assistant").write(answer)


def _process_documents(uploaded_files):
    if not uploaded_files:
        st.warning("Please upload documents first.")
        return

    with st.status("Processing documents...", expanded=True) as status:
        save_uploaded_files(uploaded_files)

        status.write("Loading documents...")
        documents = load_documents()

        status.write("Splitting documents...")
        texts = split_documents(documents)

        status.write("Creating embeddings...")
        st.session_state.retriever = create_vectorstore(texts)
        if st.session_state.retriever is None:
            status.update(label="Failed to create vectorstore.", state="error")
            return

        cleanup_temp_files()
        status.update(label="Documents ready!", state="complete")

    st.session_state.documents_ready = True
    st.session_state.index_nonempty = True
    st.rerun()


def _index_nonempty_flag():
    return bool(st.session_state.get("index_nonempty", False))


def _render_upload_panel(index_nonempty):
    if "hide_upload_menu" not in st.session_state:
        st.session_state.hide_upload_menu = False

    # If the index is empty, the upload panel can't be hidden.
    if not index_nonempty:
        st.session_state.hide_upload_menu = False

    if index_nonempty and st.session_state.hide_upload_menu:
        if st.button("Show upload menu", key="show_upload_menu"):
            st.session_state.hide_upload_menu = False
            st.rerun()
        return

    header_cols = st.columns([0.92, 0.08])
    header_cols[0].markdown("**Upload Documents**")
    if index_nonempty:
        if header_cols[1].button(
            "✕",
            key="hide_upload_menu_btn",
            help="Hide upload menu",
            use_container_width=True,
        ):
            st.session_state.hide_upload_menu = True
            st.rerun()

    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=["pdf"],
        accept_multiple_files=True,
        key="rag_upload_documents",
    )

    if st.button("Process Documents", key="process_documents"):
        _process_documents(uploaded_files)


def render_rag_upload_panel():
    _render_upload_panel(index_nonempty=_index_nonempty_flag())


def render_rag_chat_box():
    if st.session_state.documents_ready and st.session_state.retriever:
        if query := st.chat_input("Ask about your documents..."):
            st.chat_message("human").write(query)
            answer = query_llm(st.session_state.retriever, query)
            st.chat_message("assistant").write(answer)
    elif st.session_state.documents_ready and not st.session_state.retriever:
        st.error("Retriever is not available. Please process documents again.")
