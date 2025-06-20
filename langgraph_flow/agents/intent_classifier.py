import logging
import os

from langchain.chat_models import ChatOpenAI

from utils.constants import (
    ENV_OPENAIAPI_KEY,
    KEY_CONFIG,
    KEY_INFERENCE_MODEL,
    KEY_INTENT,
    KEY_OPENAI,
    KEY_QUESTION,
    MODEL_INFERENCE_OPEN_AI,
    VALUES_UTF_8,
)

logger = logging.getLogger(__name__)


def classify_intent(state: dict) -> dict:
    """
    Analyze the user’s question and classify it into one of:
      - 'retrieve' (fetch relevant code snippets)
      - 'explain'  (describe what a snippet does)
      - 'navigate' (trace symbol usage across the repo)

    Adds 'intent' to the state for downstream routing.
    """
    question = state.get(KEY_QUESTION, "").strip()
    cfg = state.get(KEY_CONFIG, {})
    model_name = cfg.get(KEY_OPENAI, {}).get(
        KEY_INFERENCE_MODEL, MODEL_INFERENCE_OPEN_AI
    )
    openai_api_key = os.getenv(ENV_OPENAIAPI_KEY)
    # TODO - Create a class/util function for this model that can be re-used in each agent.
    llm = ChatOpenAI(
        model=model_name, temperature=0, openai_api_key=openai_api_key
    )

    # Load navigation prompt template
    tmpl_path = os.path.join(
        os.path.dirname(__file__), "..", "prompts", "intent_prompt.txt"
    )
    with open(tmpl_path, "r", encoding=VALUES_UTF_8) as f:
        template = f.read()
    prompt = template.format(question=question)

    logger.debug("Dispatching intent-classification prompt to LLM")
    try:
        raw = llm.predict(prompt).strip().lower()
        # TODO - change these options to enums
        if raw not in {"retrieve", "explain", "navigate"}:
            logger.warning(
                "Unexpected intent '%s'; defaulting to 'retrieve'", raw
            )
            raw = "retrieve"
        intent = raw
    except Exception as e:
        logger.error("LLM intent classification failed: %s", e, exc_info=True)
        intent = "retrieve"

    logger.info("Intent classified as '%s'", intent)
    return {**state, KEY_INTENT: intent}
