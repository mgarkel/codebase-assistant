import logging
from typing import Dict

from langchain import PromptTemplate

from langgraph_flow.models.assistant_state import AssistantState
from langgraph_flow.models.openai_model import OpenAIModel
from utils.agent_utils import (
    get_agent_prompt_template,
    get_question_and_config_from_state,
    get_relevant_code_context_chunks_from_vectorstore,
)
from utils.constants import (
    DEFAULT_TOP_K_NAVIGATOR,
    KEY_CODE,
    KEY_CONFIG_NAVIGATOR,
    KEY_QUESTION,
    KEY_RESPONSE,
)

logger = logging.getLogger(__name__)
NAVIGATOR_PROMPT_TEMPLATE_TEXT = "navigation_prompt.txt"
template_str = get_agent_prompt_template(NAVIGATOR_PROMPT_TEMPLATE_TEXT)
NAVIGATOR_PROMPT = PromptTemplate(
    input_variables=[KEY_QUESTION, KEY_CODE],
    template=template_str,
)


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
    llm = OpenAIModel(cfg).inference_model
    runnable = NAVIGATOR_PROMPT | llm
    # Generate navigation summary
    try:
        logger.debug("Sending navigation prompt to LLM")
        navigation = runnable.invoke(
            {KEY_QUESTION: question, KEY_CODE: code_context}
        )
    except Exception as e:
        logger.error("LLM navigation generation failed: %s", e, exc_info=True)
        return {
            **state,
            KEY_RESPONSE: "Error: failed to generate navigation summary.",
        }

    logger.info("Generated navigation summary successfully")
    return {**state.dict(), KEY_RESPONSE: navigation}
