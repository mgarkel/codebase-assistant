"""Main performance monitoring singleton."""

import logging
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Dict, Optional

import psutil

from .analysis import PerformanceReport
from .config import PerformanceConfig
from .cost_calculator import TokenCostCalculator
from .models import PerformanceMetric

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Global performance monitoring singleton."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.report = PerformanceReport()
            cls._instance.enabled = True
            cls._instance.config = {}
            cls._instance.memory_warning_mb = (
                PerformanceConfig.DEFAULT_MEMORY_WARNING_MB
            )
            cls._instance.duration_warning_seconds = (
                PerformanceConfig.DEFAULT_DURATION_WARNING_SECONDS
            )
            cls._instance.include_operations = []
            cls._instance.exclude_operations = []
            cls._instance.cost_calculator = TokenCostCalculator()
        return cls._instance

    def enable(self):
        """Enable performance monitoring."""
        self.enabled = True
        logger.info("Performance monitoring enabled")

    def disable(self):
        """Disable performance monitoring."""
        self.enabled = False
        logger.info("Performance monitoring disabled")

    def clear(self):
        """Clear all collected metrics."""
        self.report = PerformanceReport()
        logger.info("Performance metrics cleared")

    def configure(self, config: Dict):
        """Configure performance monitoring from settings."""
        perf_config = config.get("performance", {})
        self.enabled = perf_config.get("enabled", True)
        self.memory_warning_mb = perf_config.get(
            "memory_warning_mb", PerformanceConfig.DEFAULT_MEMORY_WARNING_MB
        )
        self.duration_warning_seconds = perf_config.get(
            "duration_warning_seconds",
            PerformanceConfig.DEFAULT_DURATION_WARNING_SECONDS,
        )
        self.include_operations = perf_config.get("include_operations", [])
        self.exclude_operations = perf_config.get("exclude_operations", [])

        # Configure cost calculator
        cost_config = perf_config.get("token_costs", {})
        self.cost_calculator = TokenCostCalculator(cost_config)

        if cost_config:
            logger.debug(
                f"Token cost calculation configured for models: {list(cost_config.keys())}"
            )

        if not self.enabled:
            logger.info("Performance monitoring disabled via configuration")
        else:
            logger.debug(
                f"Performance monitoring configured: memory warning={self.memory_warning_mb}MB, "
                f"duration warning={self.duration_warning_seconds}s"
            )

    def _should_monitor_operation(self, operation: str) -> bool:
        """Check if an operation should be monitored based on include/exclude lists."""
        if self.exclude_operations:
            for exclude in self.exclude_operations:
                if exclude in operation:
                    return False

        if self.include_operations:
            for include in self.include_operations:
                if include in operation:
                    return True
            return False

        return True

    def _check_performance_warnings(self, metric: PerformanceMetric):
        """Check if metric exceeds warning thresholds and log warnings."""
        if metric.duration_seconds > self.duration_warning_seconds:
            logger.warning(
                f"⚠️  Slow operation: {metric.operation} took {metric.duration_seconds:.2f}s "
                f"(threshold: {self.duration_warning_seconds}s)"
            )

        if metric.memory_mb > self.memory_warning_mb:
            logger.warning(
                f"⚠️  High memory usage: {metric.operation} used {metric.memory_mb:.1f}MB "
                f"(threshold: {self.memory_warning_mb}MB)"
            )

    def get_report(self) -> PerformanceReport:
        """Get the current performance report."""
        return self.report

    @contextmanager
    def measure(self, operation: str, metadata: Optional[Dict] = None):
        """Context manager to measure operation performance."""
        if not self.enabled or not self._should_monitor_operation(operation):
            yield
            return

        start_time = time.time()
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024 / 1024  # MB

        try:
            yield
        finally:
            end_time = time.time()
            end_memory = process.memory_info().rss / 1024 / 1024  # MB

            metric = PerformanceMetric(
                operation=operation,
                duration_seconds=end_time - start_time,
                memory_mb=max(
                    end_memory, start_memory
                ),  # Peak memory during operation
                timestamp=datetime.now(),
                metadata=metadata or {},
            )

            self.report.add_metric(metric)
            self._check_performance_warnings(metric)
            logger.debug(
                f"Performance: {operation} took {metric.duration_seconds:.3f}s, "
                f"memory: {metric.memory_mb:.1f}MB"
            )


# Global instance
monitor = PerformanceMonitor()
