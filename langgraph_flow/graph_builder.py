import logging

from langgraph.graph import END, StateGraph

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
    if intent == "retrieve":
        return "retrieve"
    if intent == "explain":
        return "explain"
    if intent == "navigate":
        return "navigate"
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
    graph.add_node("classify", classify_intent)
    graph.add_node("retrieve", retrieve_code)
    graph.add_node("explain", explain_code)
    graph.add_node("navigate", navigate_code)

    # Entry point: classify user intent first
    graph.set_entry_point("classify")
    graph.add_conditional_edges("classify", _route)

    # Mark terminal nodes
    graph.add_edge("retrieve", END)
    graph.add_edge("explain", END)
    graph.add_edge("navigate", END)

    logger.info("LangGraph flow built successfully")
    return graph.compile()
