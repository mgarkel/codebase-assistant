import logging
import os
from typing import Dict, List

from langchain.chat_models import ChatOpenAI
from langchain.schema import Document

from ingestion.load_vectorstore import load_vectorstore
from langgraph_flow.models.assistant_state import AssistantState
from utils.agent_utils import (
    OpenAIModel,
    get_agent_prompt_template,
    get_combined_text_from_docs,
    get_question_and_config_from_state,
    get_relevant_code_context_chunks_from_vectorstore,
)
from utils.constants import (
    DEFAULT_TOP_K_NAVIGATOR,
    ENV_OPENAIAPI_KEY,
    KEY_CHUNK,
    KEY_CONFIG,
    KEY_CONFIG_NAVIGATOR,
    KEY_CONFIG_TOP_K,
    KEY_INFERENCE_MODEL,
    KEY_OPENAI,
    KEY_QUESTION,
    KEY_RESPONSE,
    KEY_SOURCE,
    KEY_UNKNOWN,
    MODEL_INFERENCE_OPEN_AI,
    VALUES_UTF_8,
)

logger = logging.getLogger(__name__)
NAVIGATOR_PROMPT_TEMPLATE = "navigation_prompt.txt"


def navigate_code(state: AssistantState) -> Dict:
    """
    Trace symbol usage or workflow through the codebase.

    Expects in state:
      - "question": str   # typically a navigation request, e.g., "Where is X implemented?"
      - "cfg": dict

    Returns updated state with:
      - "response": str   # navigation summary or trace steps
    """
    question, cfg = get_question_and_config_from_state(state)
    code_context = get_relevant_code_context_chunks_from_vectorstore(
        cfg, question, KEY_CONFIG_NAVIGATOR, DEFAULT_TOP_K_NAVIGATOR
    )
    template = get_agent_prompt_template(NAVIGATOR_PROMPT_TEMPLATE)
    prompt = template.format(question=question, code=code_context)
    llm = OpenAIModel(cfg).inference_model

    # Generate navigation summary
    try:
        logger.debug("Sending navigation prompt to LLM")
        navigation = llm.predict(prompt).strip()
    except Exception as e:
        logger.error("LLM navigation generation failed: %s", e, exc_info=True)
        return {
            **state,
            KEY_RESPONSE: "Error: failed to generate navigation summary.",
        }

    logger.info("Generated navigation summary successfully")
    return {**state, KEY_RESPONSE: navigation}
