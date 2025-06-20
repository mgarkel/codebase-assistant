"""Utility module for agents."""

import os

from langchain.chat_models import ChatOpenAI
from langgraph_flow.models.assistant_state import AssistantState
from utils.constants import (
    ENV_OPENAIAPI_KEY,
    KEY_CHUNK,
    KEY_EMBEDDING_MODEL,
    KEY_INFERENCE_MODEL,
    KEY_OPENAI,
    KEY_SOURCE,
    KEY_UNKNOWN,
    MODEL_EMBEDDING_OPEN_AI,
    MODEL_INFERENCE_OPEN_AI,
)


def get_question_and_config_from_state(state: AssistantState) -> tuple:
    return state.question, state.cfg


def get_combined_text_from_docs(docs: list) -> str:
    """Unpacks the documents, join the data and returns as a single str."""
    combined = []
    for doc in docs:
        meta = doc.metadata or {}
        src = meta.get(KEY_SOURCE, KEY_UNKNOWN)
        idx = meta.get(KEY_CHUNK, "?")
        snippet = doc.page_content.strip()
        combined.append(f"# {KEY_SOURCE}: {src} ({KEY_CHUNK} {idx})\n{snippet}")

    combined_code_context = "\n\n".join(combined)
    return combined_code_context


class OpenAIModel:
    def __init__(self, cfg):
        self.__openai_api_key = os.getenv(ENV_OPENAIAPI_KEY)
        self.__cfg = cfg
        self._inference_model = None
        self._embedding_model = None

    @property
    def inference_model(self):
        if not self._inference_model:
            model_name = self.__cfg.get(KEY_OPENAI, {}).get(
                KEY_INFERENCE_MODEL, MODEL_INFERENCE_OPEN_AI
            )
            self._inference_model = ChatOpenAI(
                model=model_name,
                temperature=0,
                openai_api_key=self.__openai_api_key,
            )
        return self._inference_model

    @property
    def embedding_model(self):
        if not self._embedding_model:
            model_name = self.__cfg.get(KEY_OPENAI, {}).get(
                KEY_EMBEDDING_MODEL, MODEL_EMBEDDING_OPEN_AI
            )
            self._embedding_model = ChatOpenAI(
                model=model_name,
                temperature=0,
                openai_api_key=self.__openai_api_key,
            )
        return self._embedding_model
