import logging
import os

from langchain.chat_models import ChatOpenAI

logger = logging.getLogger(__name__)


def classify_intent(state: dict) -> dict:
    """
    Analyze the user’s question and classify it into one of:
      - 'retrieve' (fetch relevant code snippets)
      - 'explain'  (describe what a snippet does)
      - 'navigate' (trace symbol usage across the repo)

    Adds 'intent' to the state for downstream routing.
    """
    question = state.get("question", "").strip()
    cfg = state.get("cfg", {})
    model_name = cfg.get("openai", {}).get("inference_model", "gpt-4")
    openai_api_key = os.getenv("OPENAPI_KEY")

    llm = ChatOpenAI(
        model=model_name, temperature=0, openai_api_key=openai_api_key
    )

    # Load navigation prompt template
    tmpl_path = os.path.join(
        os.path.dirname(__file__), "..", "prompts", "intent_prompt.txt"
    )
    with open(tmpl_path, "r", encoding="utf-8") as f:
        template = f.read()
    prompt = template.format(question=question)

    logger.debug("Dispatching intent-classification prompt to LLM")
    try:
        raw = llm.predict(prompt).strip().lower()
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
    return {**state, "intent": intent}
