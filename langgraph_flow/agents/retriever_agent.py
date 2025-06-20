import logging
from typing import Dict

from langgraph_flow.models.assistant_state import AssistantState
from utils.agent_utils import (
    get_question_and_config_from_state,
    get_relevant_code_context_chunks_from_vectorstore,
)
from utils.constants import (
    DEFAULT_TOK_K_RETRIEVER,
    KEY_CONFIG_RETRIEVER,
    KEY_RESPONSE,
)

logger = logging.getLogger(__name__)


def retrieve_code(state: AssistantState) -> Dict:
    """
    Query the vectorstore for the top-k code chunks relevant to the user's question.

    Expects in state:
      - "question": str
      - "cfg": dict (the full config)

    Returns updated state with:
      - "response": str   # formatted top results
    """
    question, cfg = get_question_and_config_from_state(state)
    try:
        code_context = get_relevant_code_context_chunks_from_vectorstore(
            cfg, question, KEY_CONFIG_RETRIEVER, DEFAULT_TOK_K_RETRIEVER
        )
        response = f"Here are the relevant code snippets:\n\n" + code_context
        return {**state, KEY_RESPONSE: response}

    except Exception as e:
        logger.error("Error during code retrieval: %s", e, exc_info=True)
        return {
            **state,
            KEY_RESPONSE: "Sorry, I ran into an error while searching the codebase.",
        }
