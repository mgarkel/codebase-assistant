import logging
from typing import Dict

from langgraph_flow.models.assistant_state import AssistantState
from utils.agent_utils import (
    OpenAIModel,
    get_agent_prompt_template,
    get_question_and_config_from_state,
    get_relevant_code_context_chunks_from_vectorstore,
)
from utils.constants import (
    DEFAULT_TOP_K_EXPLAINER,
    KEY_CONFIG_EXPLAINER,
    KEY_RESPONSE,
)

logger = logging.getLogger(__name__)
EXPLAINER_PROMPT_TEMPLATE = "explanation_prompt.txt"


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
    template = get_agent_prompt_template(EXPLAINER_PROMPT_TEMPLATE)
    prompt = template.format(code=code_context)

    # Initialize LLM
    llm = OpenAIModel(cfg).inference_model

    # Generate explanation
    try:
        logger.debug("Sending explanation prompt to LLM")
        explanation = llm.predict(prompt).strip()
    except Exception as e:
        logger.error("LLM explanation generation failed: %s", e, exc_info=True)
        return {**state, KEY_RESPONSE: "Error: failed to generate explanation."}

    logger.info("Generated explanation successfully")
    return {**state, KEY_RESPONSE: explanation}
