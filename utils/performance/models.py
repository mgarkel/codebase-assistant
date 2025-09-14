"""Performance monitoring data models and type definitions."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Protocol


# Protocols for type safety
class MetricUpdater(Protocol):
    """Protocol for classes that can update metrics with additional data."""

    def update_metric(self, metric: "PerformanceMetric") -> None:
        """Update the metric with additional data."""
        ...


@dataclass
class PerformanceMetric:
    """Single performance measurement."""

    operation: str
    duration_seconds: float
    memory_mb: float
    timestamp: datetime
    metadata: Dict = field(default_factory=dict)

    # Advanced metrics
    tokens_input: Optional[int] = None
    tokens_output: Optional[int] = None
    tokens_cost_usd: Optional[float] = None
    vectorstore_query_count: Optional[int] = None
    vectorstore_results_count: Optional[int] = None
    embedding_batch_size: Optional[int] = None
    cache_hits: Optional[int] = None
    cache_misses: Optional[int] = None


# Deferred update system base class
class DeferredUpdate(ABC):
    """Base class for updates that should be applied when the metric becomes available."""

    @abstractmethod
    def apply(self, metric: PerformanceMetric) -> None:
        """Apply this update to the given metric."""
        pass
