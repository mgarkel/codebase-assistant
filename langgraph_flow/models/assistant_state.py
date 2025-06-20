from pydantic import BaseModel, Extra
from typing import Any, Dict, Optional


class AssistantState(BaseModel):
    """
    State schema for the Codebase Assistant LangGraph flow.
    Only declares the inputs and the final output.
    Intermediate keys (intent, etc.) are allowed via extra=allow.
    """

    question: str
    cfg: Dict[str, Any]
    response: Optional[str] = None

    class Config:
        # Permit intermediate keys like "intent" without error
        extra = Extra.allow
