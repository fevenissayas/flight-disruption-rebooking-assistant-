"""Chat model used by every agent node. Tests can inject a stand-in."""

import os

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI

_override: BaseChatModel | None = None


def set_llm(model: BaseChatModel | None) -> None:
    global _override
    _override = model


def get_llm() -> BaseChatModel:
    if _override is not None:
        return _override
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and add a Gemini key."
        )
    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        api_key=api_key,
        temperature=0,
        seed=1,
    )
