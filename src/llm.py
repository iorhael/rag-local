import os

import streamlit as st
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from config import SYSTEM_PROMPT


def create_llm(temperature=0.7):
    return ChatOpenAI(
        openai_api_key=os.getenv("OPEN_ROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        temperature=temperature,
        model="gpt-3.5-turbo",
    )


def query_llm(retriever, query):
    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT + "\n\nContext:\n{context}"),
            ("human", "{question}"),
        ])
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=create_llm(),
            retriever=retriever,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": prompt},
        )
        result = qa_chain({
            "question": query,
            "chat_history": st.session_state.messages,
        })
        st.session_state.messages.append((query, result["answer"]))
        return result["answer"]
    except Exception as e:
        st.error(f"Error processing query: {str(e)}")
        return "Error processing your question. Please try again."
