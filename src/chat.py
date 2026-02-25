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


def run_rag_chat():
    if not st.session_state.documents_ready:
        uploaded_files = st.file_uploader(
            "Upload Documents", type=["pdf"], accept_multiple_files=True
        )

        if st.button("Process Documents"):
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

                cleanup_temp_files()
                status.update(label="Documents ready!", state="complete")

            st.session_state.documents_ready = True
            st.rerun()

    if st.session_state.documents_ready:
        if query := st.chat_input("Ask about your documents..."):
            st.chat_message("human").write(query)
            answer = query_llm(st.session_state.retriever, query)
            st.chat_message("assistant").write(answer)
