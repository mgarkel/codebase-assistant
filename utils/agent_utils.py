"""Utility module for agents."""

import logging
import os
from typing import List

from langchain.schema import Document

from ingestion.load_vectorstore import load_vectorstore
from langgraph_flow.models.assistant_state import AssistantState
from utils.constants import (
    KEY_CHUNK_INDEX,
    KEY_CODE_LANGUAGE,
    KEY_CONFIG_TOP_K,
    KEY_RELATIVE_PATH,
    KEY_UNKNOWN,
    VALUES_UTF_8,
)

logger = logging.getLogger(__name__)


def get_question_and_config_from_state(state: AssistantState) -> tuple:
    return state.question, state.cfg


def run_llm(runnable, input_params: dict) -> str:
    """Invoke LLM prompt."""
    return runnable.invoke(input_params).content


def get_agent_prompt_template(prompt_template_file: str):
    # Load prompt template
    tmpl_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "langgraph_flow",
        "prompts",
        prompt_template_file,
    )
    with open(tmpl_path, "r", encoding=VALUES_UTF_8) as f:
        template = f.read()
    return template.strip()


def get_combined_text_from_docs(docs: list) -> str:
    """Unpacks the documents, join the data and returns as a single str."""
    combined = []
    for doc in docs:
        meta = doc.metadata or {}
        path = meta.get(KEY_RELATIVE_PATH, KEY_UNKNOWN)
        idx = meta.get(KEY_CHUNK_INDEX, "?")
        lang = meta.get(KEY_CODE_LANGUAGE)
        snippet = doc.page_content.strip()
        combined.append(
            f"''' {KEY_CODE_LANGUAGE}: {lang}, {KEY_RELATIVE_PATH}: {path}, ({KEY_CHUNK_INDEX} {idx})\n{snippet} '''\n"
        )

    combined_code_context = "\n\n".join(combined)
    return combined_code_context


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
            "Retrieving top %d snippets for %s with question: %s",
            top_k,
            agent_name,
            question,
        )
        docs: List[Document] = store.similarity_search(question, k=top_k)
    except Exception as e:
        logger.error(
            "Similarity search failed in %s: %s", agent_name, e, exc_info=True
        )
        raise Exception

    if not docs:
        logger.error("No snippets found for %s query: %s", agent_name, question)
        raise Exception

    # Combine docs
    code_context = get_combined_text_from_docs(docs)
    return code_context
