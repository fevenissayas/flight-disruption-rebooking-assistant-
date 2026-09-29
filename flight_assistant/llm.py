"""Chat model used by every agent node. Tests can inject a stand-in."""

import os

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

_override: BaseChatModel | None = None


def set_llm(model: BaseChatModel | None) -> None:
    global _override
    _override = model


def get_llm() -> BaseChatModel:
    if _override is not None:
        return _override
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add a key."
        )
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,
        seed=1,
    )
