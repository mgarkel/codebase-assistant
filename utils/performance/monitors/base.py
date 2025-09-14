"""Base monitor class for performance monitoring."""

import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from ..models import DeferredUpdate, PerformanceMetric


def _get_memory_measurements() -> float:
    """Get current memory usage in MB."""
    import psutil

    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    return memory_mb


def _create_base_metric(
    operation: str,
    start_time: float,
    end_time: float,
    start_memory: float,
    end_memory: float,
    metadata: Dict = None,
) -> PerformanceMetric:
    """Create a base performance metric with common fields."""
    from datetime import datetime

    return PerformanceMetric(
        operation=operation,
        duration_seconds=end_time - start_time,
        memory_mb=max(end_memory, start_memory),
        timestamp=datetime.now(),
        metadata=metadata or {},
    )


class BaseMonitor(ABC):
    """Base class for specialized performance monitors."""

    def __init__(self, operation: str):
        self.operation = operation
        self.start_time: Optional[float] = None
        self.start_memory: Optional[float] = None
        self.metric: Optional[PerformanceMetric] = None
        self.enabled = False
        self._deferred_updates: List[DeferredUpdate] = []

    def __enter__(self):
        # Import here to avoid circular imports
        from ..monitor import monitor

        if not monitor.enabled or not monitor._should_monitor_operation(
            self.operation
        ):
            self.enabled = False
            return self

        self.enabled = True
        self.start_time = time.time()
        self.start_memory = _get_memory_measurements()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enabled:
            return

        # Import here to avoid circular imports
        from ..monitor import monitor

        end_time = time.time()
        end_memory = _get_memory_measurements()

        # Create the base metric
        self.metric = _create_base_metric(
            self.operation,
            self.start_time,
            end_time,
            self.start_memory,
            end_memory,
            self._get_metadata(),
        )

        # Apply operation-specific customizations
        self._customize_metric(self.metric)

        # Apply any deferred updates
        for update in self._deferred_updates:
            update.apply(self.metric)

        # Add to report and check warnings
        monitor.report.add_metric(self.metric)
        monitor._check_performance_warnings(self.metric)

        # Log completion
        self._log_completion()

    @abstractmethod
    def _get_metadata(self) -> Dict:
        """Get operation-specific metadata for the metric."""
        pass

    @abstractmethod
    def _customize_metric(self, metric: PerformanceMetric) -> None:
        """Apply operation-specific customizations to the metric."""
        pass

    @abstractmethod
    def _log_completion(self) -> None:
        """Log operation completion."""
        pass

    def _add_deferred_update(self, update: DeferredUpdate) -> None:
        """Add a deferred update to be applied when the metric is created."""
        if self.metric is not None:
            # Metric already exists, apply immediately
            update.apply(self.metric)
        else:
            # Store for later application
            self._deferred_updates.append(update)
