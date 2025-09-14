"""Deferred update system for performance metrics."""

import logging

from .config import PerformanceConfig
from .cost_calculator import TokenCostCalculator
from .models import DeferredUpdate, PerformanceMetric

logger = logging.getLogger(__name__)


class TokenResponseUpdate(DeferredUpdate):
    """Deferred update for LLM response token and cost information."""

    def __init__(
        self,
        response_text: str,
        model_name: str,
        cost_calculator: TokenCostCalculator,
    ):
        self.response_text = response_text
        self.model_name = model_name
        self.cost_calculator = cost_calculator

    def apply(self, metric: PerformanceMetric) -> None:
        """Apply token and cost updates to the metric."""
        # Estimate output tokens
        estimated_output_tokens = (
            len(self.response_text) // PerformanceConfig.CHARS_PER_TOKEN
        )
        metric.tokens_output = estimated_output_tokens

        # Calculate cost if possible
        if metric.tokens_input:
            cost = self.cost_calculator.calculate_cost(
                self.model_name, metric.tokens_input, estimated_output_tokens
            )
            if cost > 0:
                metric.tokens_cost_usd = cost

        logger.info(
            f"LLM response processed: {estimated_output_tokens} output tokens, "
            f"model: {self.model_name}, cost: ${metric.tokens_cost_usd or 0:.4f}"
        )


class VectorStoreResultsUpdate(DeferredUpdate):
    """Deferred update for vector store result count."""

    def __init__(self, results_count: int):
        self.results_count = results_count

    def apply(self, metric: PerformanceMetric) -> None:
        """Apply results count to the metric."""
        metric.vectorstore_results_count = self.results_count
