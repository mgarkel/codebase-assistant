"""LLM performance monitoring."""

import logging
from typing import Dict

from ..config import PerformanceConfig
from ..deferred_updates import TokenResponseUpdate
from ..models import PerformanceMetric
from .base import BaseMonitor

logger = logging.getLogger(__name__)


class LLMMonitor(BaseMonitor):
    """Helper class to monitor LLM operations with token and cost tracking."""

    def __init__(
        self, operation: str, input_text: str, model_name: str = "unknown"
    ):
        super().__init__(operation)
        self.input_text = input_text
        self.model_name = model_name

    def _get_metadata(self) -> Dict:
        """Get LLM-specific metadata for the metric."""
        return {
            "model": self.model_name,
            "input_length": len(self.input_text),
        }

    def _customize_metric(self, metric: PerformanceMetric) -> None:
        """Apply LLM-specific customizations to the metric."""
        # Estimate input tokens
        estimated_input_tokens = (
            len(self.input_text) // PerformanceConfig.CHARS_PER_TOKEN
        )
        metric.tokens_input = estimated_input_tokens

    def _log_completion(self) -> None:
        """Log LLM operation completion."""
        logger.info(
            f"LLM: {self.operation} took {self.metric.duration_seconds:.3f}s, "
            f"~{self.metric.tokens_input} input tokens"
        )

    def set_response(self, response_text: str):
        """Update the metric with response information for output tokens and cost."""
        if not self.enabled:
            return

        # Import here to avoid circular imports
        from ..monitor import monitor

        # Create a deferred update
        update = TokenResponseUpdate(
            response_text, self.model_name, monitor.cost_calculator
        )
        self._add_deferred_update(update)

        logger.info(f"LLM response updated for operation {self.operation}")


def measure_llm_operation(
    operation: str, input_text: str, model_name: str = "unknown"
):
    """Create an LLM monitor context manager."""
    return LLMMonitor(operation, input_text, model_name)
