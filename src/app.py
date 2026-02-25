import streamlit as st

from config import validate_api_keys
from vectorstore import clear_vector_index
from chat import run_simple_chat, render_rag_upload_panel, render_rag_chat_box

st.set_page_config(page_title="RAG System")
st.title("👾 AI Assistant with RAG options")
st.write("Upload your documents or just chat with friendly bot.")

MODES = ("Simple Chat", "RAG Chat")


def _messages_key_for_mode(mode):
    return "messages_simple" if mode == "Simple Chat" else "messages_rag"


def _sync_active_messages_pointer():
    st.session_state.messages = st.session_state[_messages_key_for_mode(st.session_state.mode)]


def _switch_mode(new_mode):
    st.session_state[_messages_key_for_mode(st.session_state.mode)] = st.session_state.messages
    st.session_state.mode = new_mode
    _sync_active_messages_pointer()


def _clear_active_chat():
    st.session_state[_messages_key_for_mode(st.session_state.mode)] = []
    _sync_active_messages_pointer()


def init_session_state():
    defaults = {
        "messages": [],
        "messages_simple": [],
        "messages_rag": [],
        "retriever": None,
        "mode": "Simple Chat",
        "documents_ready": False,
        "hide_upload_menu": False,
        "index_nonempty": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Backward-compat: if legacy `messages` has content, bind it to current mode once.
    active_key = _messages_key_for_mode(st.session_state.mode)
    if st.session_state.messages and not st.session_state[active_key]:
        st.session_state[active_key] = st.session_state.messages
    _sync_active_messages_pointer()


def render_sidebar():
    with st.sidebar:
        new_mode = st.radio(
            "Select Mode",
            MODES,
            index=MODES.index(st.session_state.mode),
        )

        if new_mode != st.session_state.mode:
            _switch_mode(new_mode)
            st.rerun()

        if st.button("Clear Chat"):
            _clear_active_chat()
            st.rerun()

        if st.button("Clear Vector Index"):
            clear_vector_index()
            st.session_state.retriever = None
            st.session_state.documents_ready = False
            st.session_state.messages_rag = []
            if st.session_state.mode == "RAG Chat":
                _sync_active_messages_pointer()
            st.session_state.hide_upload_menu = False
            st.session_state.index_nonempty = False
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

    if st.session_state.mode == "Simple Chat":
        render_chat_history()
        run_simple_chat()
    else:
        # Upload menu must be above the first chat message.
        render_rag_upload_panel()
        render_chat_history()
        render_rag_chat_box()


if __name__ == "__main__":
    main()
