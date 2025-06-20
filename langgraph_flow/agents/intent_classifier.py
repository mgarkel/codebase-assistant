import logging

from langgraph_flow.models.assistant_state import AssistantState
from langgraph_flow.models.openai_model import OpenAIModel
from utils.agent_utils import (
    get_agent_prompt_template,
    get_question_and_config_from_state,
)
from utils.constants import KEY_INTENT

logger = logging.getLogger(__name__)
INTENT_PROMPT_TEMPLATE = "intent_prompt.txt"


def classify_intent(state: AssistantState) -> dict:
    """
    Analyze the user’s question and classify it into one of:
      - 'retrieve' (fetch relevant code snippets)
      - 'explain'  (describe what a snippet does)
      - 'navigate' (trace symbol usage across the repo)

    Adds 'intent' to the state for downstream routing.
    """
    question, cfg = get_question_and_config_from_state(state)
    llm = OpenAIModel(cfg).inference_model
    template = get_agent_prompt_template(INTENT_PROMPT_TEMPLATE)
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
    return {**state.dict(), KEY_INTENT: intent}
