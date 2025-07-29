from typing import Dict

from langgraph_flow.agents.agent import Agent
from langgraph_flow.agents.enums import Intent
from langgraph_flow.models.assistant_state import AssistantState
from utils.constants import DEFAULT_TOP_K_EXPLAINER

EXPLAINER_PROMPT_TEMPLATE_TEXT = "explanation_prompt.txt"


def explain_code(state: AssistantState) -> Dict:
    agent = Agent(
        agent_type=Intent.EXPLAIN.value,
        prompt_file=EXPLAINER_PROMPT_TEMPLATE_TEXT,
        default_top_k=DEFAULT_TOP_K_EXPLAINER,
        is_input_code=True,
        is_input_question=False,
    )
    return agent.infer(state)
