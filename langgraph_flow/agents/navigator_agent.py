import os
import logging
from typing import List, Dict

from langchain.chat_models import ChatOpenAI
from langchain.schema import Document

from ingestion.load_vectorstore import load_vectorstore
from langgraph_flow.models.assistant_state import AssistantState
from utils.agent_utils import (
    OpenAIModel,
    get_combined_text_from_docs,
    get_question_and_config_from_state,
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

    # Load vectorstore
    try:
        store = load_vectorstore(cfg)
    except Exception as e:
        logger.error(
            "Failed to load vectorstore for navigation: %s", e, exc_info=True
        )
        return {**state, "response": "Error: could not access the code index."}

    # Determine how many snippets to consider for navigation
    top_k = cfg.get(KEY_CONFIG_NAVIGATOR, {}).get(
        KEY_CONFIG_TOP_K, DEFAULT_TOP_K_NAVIGATOR
    )

    # Perform similarity search to find relevant code contexts
    try:
        logger.info(
            "Retrieving top %d snippets for navigation of: %s", top_k, question
        )
        docs: List[Document] = store.similarity_search(question, k=top_k)
    except Exception as e:
        logger.error(
            "Similarity search failed in navigator_agent: %s", e, exc_info=True
        )
        return {
            **state,
            "response": "Error: failed to retrieve code snippets for navigation.",
        }

    if not docs:
        logger.warning("No snippets found for navigation query: %s", question)
        return {
            **state,
            "response": "I couldn't find any relevant code to navigate.",
        }

    code_context = get_combined_text_from_docs(docs)

    # Load navigation prompt template
    tmpl_path = os.path.join(
        os.path.dirname(__file__), "..", "prompts", "navigation_prompt.txt"
    )
    with open(tmpl_path, "r", encoding=VALUES_UTF_8) as f:
        template = f.read()

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
