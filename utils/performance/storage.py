"""Performance metrics storage and data management."""

from dataclasses import dataclass, field
from typing import List

from .models import PerformanceMetric


@dataclass
class PerformanceMetrics:
    """Storage for performance metrics."""

    metrics: List[PerformanceMetric] = field(default_factory=list)

    def add_metric(self, metric: PerformanceMetric):
        """Add a performance metric to the collection."""
        self.metrics.append(metric)

    def clear(self):
        """Clear all collected metrics."""
        self.metrics.clear()

    def get_metrics(self) -> List[PerformanceMetric]:
        """Get all metrics."""
        return self.metrics.copy()

    def filter_by_operation(
        self, operation_pattern: str
    ) -> List[PerformanceMetric]:
        """Filter metrics by operation name pattern."""
        return [m for m in self.metrics if operation_pattern in m.operation]
