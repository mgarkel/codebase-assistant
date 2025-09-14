"""Token cost calculation for LLM operations."""

import logging
from typing import Dict

logger = logging.getLogger(__name__)


class TokenCostCalculator:
    """Configurable strategy for calculating LLM token costs."""

    def __init__(self, cost_config: Dict[str, Dict[str, float]] = None):
        """
        Initialize with cost configuration.

        Expected format:
        {
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-3.5": {"input": 0.0015, "output": 0.002}
        }
        """
        self.cost_config = cost_config or {}

    def calculate_cost(
        self, model_name: str, input_tokens: int, output_tokens: int
    ) -> float:
        """Calculate cost for the given model and token counts."""
        if not self.cost_config:
            return 0.0

        # Find matching model configuration
        model_config = None
        model_name_lower = model_name.lower()

        for model_key, pricing in self.cost_config.items():
            if model_key.lower() in model_name_lower:
                model_config = pricing
                break

        if not model_config:
            logger.warning(
                f"No cost configuration found for model '{model_name}'. "
                f"Available models: {list(self.cost_config.keys())}. Skipping cost calculation."
            )
            return 0.0

        try:
            input_cost = input_tokens * model_config.get("input", 0) / 1000
            output_cost = output_tokens * model_config.get("output", 0) / 1000
            return input_cost + output_cost
        except (TypeError, KeyError) as e:
            logger.warning(
                f"Error calculating cost for model '{model_name}': {e}"
            )
            return 0.0
