"""Shared chat model for novel agents (model id from NOVEL_LLM)."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel

load_dotenv()

DEFAULT_NOVEL_MODEL = "google_genai:gemini-2.5-flash"
_DEFAULT_MODEL = DEFAULT_NOVEL_MODEL


def get_novel_llm() -> BaseChatModel:
    model_id = os.getenv("NOVEL_LLM", _DEFAULT_MODEL).strip() or _DEFAULT_MODEL
    return init_chat_model(model_id)
