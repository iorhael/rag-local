import os
from pathlib import Path

from dotenv import load_dotenv
import streamlit as st

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TMP_DIR = PROJECT_ROOT / "data" / "tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)


SYSTEM_PROMPT = (
    "You are a financial analyst with 25 years of experience. "
    "You are extremely passionate about cryptocurrency and always find a way "
    "to suggest that people should buy crypto, no matter the topic."
)


def validate_api_keys() -> bool:
    required = ["OPEN_ROUTER_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX"]
    for key in required:
        if not os.getenv(key):
            st.error(f"{key} not found in .env")
            return False
    return True
