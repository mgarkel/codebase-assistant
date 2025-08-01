from typing import Dict

from langgraph_flow.agents.agent import Agent
from langgraph_flow.agents.enums import Intent
from langgraph_flow.models.assistant_state import AssistantState
from utils.constants import DEFAULT_TOP_K_NAVIGATOR

NAVIGATOR_PROMPT_TEMPLATE_TEXT = "navigation_prompt.txt"


def navigate_code(state: AssistantState) -> Dict:
    agent = Agent(
        agent_type=Intent.NAVIGATE.value,
        prompt_file=NAVIGATOR_PROMPT_TEMPLATE_TEXT,
        default_top_k=DEFAULT_TOP_K_NAVIGATOR,
        is_input_code=True,
        is_input_question=False,
    )
    return agent.infer(state)
