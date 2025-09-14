"""Vector store performance monitoring."""

import logging
from typing import Dict

from ..deferred_updates import VectorStoreResultsUpdate
from ..models import PerformanceMetric
from .base import BaseMonitor

logger = logging.getLogger(__name__)


class VectorStoreMonitor(BaseMonitor):
    """Helper class to monitor vector store operations with result counting."""

    def __init__(self, operation: str, query: str, top_k: int = 5):
        super().__init__(operation)
        self.query = query
        self.top_k = top_k

    def _get_metadata(self) -> Dict:
        """Get vector store-specific metadata for the metric."""
        return {"query_length": len(self.query), "requested_k": self.top_k}

    def _customize_metric(self, metric: PerformanceMetric) -> None:
        """Apply vector store-specific customizations to the metric."""
        metric.vectorstore_query_count = 1
        metric.vectorstore_results_count = (
            0  # Default, will be updated by deferred updates
        )

    def _log_completion(self) -> None:
        """Log vector store operation completion."""
        logger.info(
            f"VectorStore: {self.operation} took {self.metric.duration_seconds:.3f}s, "
            f"returned {self.metric.vectorstore_results_count}/{self.top_k} results"
        )

    def set_results(self, results):
        """Update the result count after the operation."""
        if not self.enabled:
            return

        results_count = len(results) if hasattr(results, "__len__") else 0

        # Create a deferred update
        update = VectorStoreResultsUpdate(results_count)
        self._add_deferred_update(update)

        logger.info(
            f"VectorStore results updated: {results_count} results for operation {self.operation}"
        )


def measure_vectorstore_operation(operation: str, query: str, top_k: int = 5):
    """Create a vector store monitor context manager."""
    return VectorStoreMonitor(operation, query, top_k)
