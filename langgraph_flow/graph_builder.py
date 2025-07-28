import logging

from langgraph.graph import END, StateGraph

from utils.constants import ALLOWED_INTENTS

from .agents.enums import Intent
from .agents.explainer_agent import explain_code
from .agents.intent_classifier import classify_intent
from .agents.navigator_agent import navigate_code
from .agents.retriever_agent import retrieve_code
from .models.assistant_state import AssistantState

logger = logging.getLogger(__name__)


# Routing based on the classified intent
def _route(state: AssistantState):
    intent = state.intent
    logger.debug("Routing intent '%s'", intent)
    if intent in ALLOWED_INTENTS:
        return intent
    logger.warning("Unknown intent '%s', defaulting to 'retrieve'", intent)
    return "retrieve"


def build_graph():
    """
    Build and compile the LangGraph StateGraph for the Codebase Assistant.

    Args:
        cfg: Configuration dict (will be passed through the state)

    Returns:
        A compiled StateGraph instance ready to run.
    """
    logger.info("Initializing LangGraph flow")
    graph = StateGraph(state_schema=AssistantState)

    # Add processing nodes
    graph.add_node(Intent.CLASSIFY.value, classify_intent)
    graph.add_node(Intent.RETRIEVE.value, retrieve_code)
    graph.add_node(Intent.EXPLAIN.value, explain_code)
    graph.add_node(Intent.NAVIGATE.value, navigate_code)

    # Entry point: classify user intent first
    graph.set_entry_point(Intent.CLASSIFY.value)
    graph.add_conditional_edges(Intent.CLASSIFY.value, _route)

    # Mark terminal nodes
    graph.add_edge(Intent.RETRIEVE.value, END)
    graph.add_edge(Intent.EXPLAIN.value, END)
    graph.add_edge(Intent.NAVIGATE.value, END)

    logger.info("LangGraph flow built successfully")
    return graph.compile()
