"""Enum class for python agent intents"""

from enum import Enum


class Intent(Enum):
    CLASSIFY = "classify"
    RETRIEVE = "retrieve"
    EXPLAIN = "explain"
    NAVIGATE = "navigate"
