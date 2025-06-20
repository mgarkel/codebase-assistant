from pydantic import BaseModel
from typing import Any, Dict

class AssistantState(BaseModel):
    """
    State schema for the Codebase Assistant LangGraph flow.
    """
    question: str           # incoming user query
    cfg: Dict[str, Any]     # full config passed into each run
    intent: str             # set by intent_classifier
    response: str           # set by the terminal agents