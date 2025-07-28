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
    DEFAULT_TOP_K_EXPLAINER,
    KEY_CODE,
    KEY_CONFIG_EXPLAINER,
    KEY_RESPONSE,
)

logger = logging.getLogger(__name__)
EXPLAINER_PROMPT_TEMPLATE_TEXT = "explanation_prompt.txt"
template_str = get_agent_prompt_template(EXPLAINER_PROMPT_TEMPLATE_TEXT)
EXPLAINER_PROMPT = PromptTemplate(
    input_variables=[KEY_CODE],
    template=template_str,
)


def explain_code(state: AssistantState) -> Dict:
    """
    Retrieve top-k relevant code snippets and generate a natural-language explanation.
    Expects in state:
      - "question": str
      - "cfg": dict

    Returns updated state with:
      - "response": str  # explanation text
    """
    question, cfg = get_question_and_config_from_state(state)
    code_context = get_relevant_code_context_chunks_from_vectorstore(
        cfg, question, KEY_CONFIG_EXPLAINER, DEFAULT_TOP_K_EXPLAINER
    )
    llm = OpenAIModel(cfg).inference_model
    runnable = EXPLAINER_PROMPT | llm

    try:
        logger.debug("Sending explanation prompt to LLM")
        explanation = runnable.invoke({KEY_CODE: code_context}).content
    except Exception as e:
        logger.error("LLM explanation generation failed: %s", e, exc_info=True)
        return {**state, KEY_RESPONSE: "Error: failed to generate explanation."}

    logger.info("Generated explanation successfully")
    return {**state.dict(), KEY_RESPONSE: explanation}
