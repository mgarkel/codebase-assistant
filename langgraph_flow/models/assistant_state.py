from typing import Any, Dict, Optional

from pydantic import BaseModel


class AssistantState(BaseModel):
    """
    State schema for the Codebase Assistant LangGraph flow.
    """

    question: str  # required
    cfg: Dict[str, Any]  # required
    intent: Optional[str] = None  # optional, default None
    response: Optional[str] = None  # optional, default None
