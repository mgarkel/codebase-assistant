import logging

from langchain import PromptTemplate

from langgraph_flow.models.assistant_state import AssistantState
from langgraph_flow.models.openai_model import OpenAIModel
from utils.agent_utils import (
    get_agent_prompt_template,
    get_question_and_config_from_state,
)
from utils.constants import ALLOWED_INTENTS, KEY_INTENT, KEY_QUESTION

logger = logging.getLogger(__name__)
INTENT_PROMPT_TEMPLATE_TEXT = "intent_prompt.txt"
template_str = get_agent_prompt_template(INTENT_PROMPT_TEMPLATE_TEXT)
INTENT_PROMPT = PromptTemplate(
    input_variables=[KEY_QUESTION],
    template=template_str,
)


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
    runnable = INTENT_PROMPT | llm

    logger.debug("Dispatching intent-classification prompt to LLM")
    try:
        raw = runnable.invoke({KEY_QUESTION: question}).content
        while raw not in ALLOWED_INTENTS:
            logger.warning(
                "Sorry - intent of question was unclear. Please rephrase question"
            )
            question = input("\n❓ Ask your codebase: ").strip()
            raw = runnable.invoke({KEY_QUESTION: question}).content
            state.question = question
        intent = raw
    except Exception as e:
        logger.error("LLM intent classification failed: %s", e, exc_info=True)
        raise e

    logger.info("Intent classified as '%s'", intent)
    return {**state.dict(), KEY_INTENT: intent}
