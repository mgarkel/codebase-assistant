import logging
from typing import Dict

from langgraph_flow.agents.agent import Agent
from langgraph_flow.models.assistant_state import AssistantState
from utils.constants import DEFAULT_TOK_K_RETRIEVER, KEY_RETRIEVER

logger = logging.getLogger(__name__)


def retrieve_code(state: AssistantState) -> Dict:
    agent = Agent(
        agent_type=KEY_RETRIEVER,
        prompt_file=None,
        default_top_k=DEFAULT_TOK_K_RETRIEVER,
        is_input_code=False,
        is_input_question=False,
    )
    return agent.infer(state)
