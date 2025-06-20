import os
import logging
from typing import List, Dict

from langchain.chat_models import ChatOpenAI
from langchain.schema import Document

from ingestion.load_vectorstore import load_vectorstore
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


def navigate_code(state: Dict) -> Dict:
    """
    Trace symbol usage or workflow through the codebase.

    Expects in state:
      - "question": str   # typically a navigation request, e.g., "Where is X implemented?"
      - "cfg": dict

    Returns updated state with:
      - "response": str   # navigation summary or trace steps
    """
    question = state.get(KEY_QUESTION, "").strip()
    cfg = state.get(KEY_CONFIG, {})

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

    combined = []
    for doc in docs:
        meta = doc.metadata or {}
        src = meta.get(KEY_SOURCE, KEY_UNKNOWN)
        idx = meta.get(KEY_CHUNK, "?")
        snippet = doc.page_content.strip()
        combined.append(f"# {KEY_SOURCE}: {src} ({KEY_CHUNK} {idx})\n{snippet}")

    code_context = "\n\n".join(combined)

    # Load navigation prompt template
    tmpl_path = os.path.join(
        os.path.dirname(__file__), "..", "prompts", "navigation_prompt.txt"
    )
    with open(tmpl_path, "r", encoding=VALUES_UTF_8) as f:
        template = f.read()

    prompt = template.format(question=question, code=code_context)

    # Initialize LLM
    model_name = cfg.get(KEY_OPENAI, {}).get(
        KEY_INFERENCE_MODEL, MODEL_INFERENCE_OPEN_AI
    )
    openai_api_key = os.getenv(ENV_OPENAIAPI_KEY)
    llm = ChatOpenAI(
        model=model_name, temperature=0, openai_api_key=openai_api_key
    )

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
