from typing import Dict

from langgraph_flow.agents.agent import Agent
from langgraph_flow.models.assistant_state import AssistantState
from utils.constants import DEFAULT_TOP_K_NAVIGATOR, KEY_CONFIG_NAVIGATOR

NAVIGATOR_PROMPT_TEMPLATE_TEXT = "navigation_prompt.txt"


def navigate_code(state: AssistantState) -> Dict:
    agent = Agent(
        agent_type=KEY_CONFIG_NAVIGATOR,
        prompt_file=NAVIGATOR_PROMPT_TEMPLATE_TEXT,
        default_top_k=DEFAULT_TOP_K_NAVIGATOR,
        is_input_code=True,
        is_input_question=True,
    )
    return agent.infer(state)
