"""Utility module for agents."""

import logging
import os
from typing import List

from langchain.chat_models import ChatOpenAI
from langchain.schema import Document

from langgraph_flow.models.assistant_state import AssistantState
from utils.constants import (
    ENV_OPENAIAPI_KEY,
    KEY_CHUNK,
    KEY_CONFIG_TOP_K,
    KEY_EMBEDDING_MODEL,
    KEY_INFERENCE_MODEL,
    KEY_OPENAI,
    KEY_RESPONSE,
    KEY_SOURCE,
    KEY_UNKNOWN,
    MODEL_EMBEDDING_OPEN_AI,
    MODEL_INFERENCE_OPEN_AI,
    VALUES_UTF_8,
)

logger = logging.getLogger(__name__)


def get_question_and_config_from_state(state: AssistantState) -> tuple:
    return state.question, state.cfg


def get_agent_prompt_template(prompt_template_file: str):
    # Load prompt template
    tmpl_path = os.path.join(
        os.path.dirname(__file__), "..", "prompts", prompt_template_file
    )
    with open(tmpl_path, "r", encoding=VALUES_UTF_8) as f:
        template = f.read()
    return template


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


def load_vectorstore(cfg: dict):
    # Load vectorstore
    try:
        store = load_vectorstore(cfg)
    except Exception as e:
        logger.error(
            "Failed to load vectorstore for explanation: %s", e, exc_info=True
        )
        raise Exception
    return store


def get_relevant_code_context_chunks_from_vectorstore(
    cfg: dict, question: str, agent_name: str, default_top_k
):
    # Determine how many snippets to explain
    top_k = cfg.get(agent_name, {}).get(KEY_CONFIG_TOP_K, default_top_k)

    # Load vectorstore
    store = load_vectorstore(cfg)

    # Perform similarity search
    try:
        logger.info(
            "Retrieving top %d snippets for explanation of: %s", top_k, question
        )
        docs: List[Document] = store.similarity_search(question, k=top_k)
    except Exception as e:
        logger.error(
            "Similarity search failed in explainer_agent: %s", e, exc_info=True
        )
        raise Exception

    if not docs:
        logger.error("No snippets found for explanation query: %s", question)
        raise Exception

    # Combine docs
    code_context = get_combined_text_from_docs(docs)
    return code_context


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
