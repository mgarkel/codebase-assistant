"""Performance monitoring utilities for the codebase assistant."""

import json
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import psutil

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Single performance measurement."""

    operation: str
    duration_seconds: float
    memory_mb: float
    timestamp: datetime
    metadata: Dict = field(default_factory=dict)


@dataclass
class PerformanceReport:
    """Collection of performance metrics."""

    metrics: List[PerformanceMetric] = field(default_factory=list)

    def add_metric(self, metric: PerformanceMetric):
        """Add a performance metric to the report."""
        self.metrics.append(metric)

    def get_summary(self) -> Dict:
        """Get summary statistics for all metrics."""
        if not self.metrics:
            return {}

        by_operation = {}
        for metric in self.metrics:
            if metric.operation not in by_operation:
                by_operation[metric.operation] = []
            by_operation[metric.operation].append(metric)

        summary = {}
        for operation, metrics in by_operation.items():
            durations = [m.duration_seconds for m in metrics]
            memory_usage = [m.memory_mb for m in metrics]

            summary[operation] = {
                "count": len(metrics),
                "avg_duration_seconds": sum(durations) / len(durations),
                "max_duration_seconds": max(durations),
                "min_duration_seconds": min(durations),
                "avg_memory_mb": sum(memory_usage) / len(memory_usage),
                "max_memory_mb": max(memory_usage),
            }

        return summary

    def to_json(self) -> str:
        """Export metrics as JSON."""
        data = {
            "summary": self.get_summary(),
            "detailed_metrics": [
                {
                    "operation": m.operation,
                    "duration_seconds": m.duration_seconds,
                    "memory_mb": m.memory_mb,
                    "timestamp": m.timestamp.isoformat(),
                    "metadata": m.metadata,
                }
                for m in self.metrics
            ],
        }
        return json.dumps(data, indent=2)


class PerformanceMonitor:
    """Global performance monitoring singleton."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.report = PerformanceReport()
            cls._instance.enabled = True
            cls._instance.config = {}
            cls._instance.memory_warning_mb = 1000
            cls._instance.duration_warning_seconds = 30
            cls._instance.include_operations = []
            cls._instance.exclude_operations = []
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
        self.memory_warning_mb = perf_config.get("memory_warning_mb", 1000)
        self.duration_warning_seconds = perf_config.get(
            "duration_warning_seconds", 30
        )
        self.include_operations = perf_config.get("include_operations", [])
        self.exclude_operations = perf_config.get("exclude_operations", [])

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


def measure_performance(operation: str = None, metadata: Optional[Dict] = None):
    """Decorator to measure function performance."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            op_name = operation or f"{func.__module__}.{func.__name__}"
            with monitor.measure(op_name, metadata):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def auto_instrument_functions(module_name: str, functions: List[str]):
    """Automatically instrument a list of functions in a module."""
    import importlib

    try:
        module = importlib.import_module(module_name)
        for func_name in functions:
            if hasattr(module, func_name):
                original_func = getattr(module, func_name)
                instrumented_func = measure_performance(
                    f"{module_name}.{func_name}"
                )(original_func)
                setattr(module, func_name, instrumented_func)
                logger.debug(f"Instrumented {module_name}.{func_name}")
    except Exception as e:
        logger.warning(f"Failed to instrument {module_name}: {e}")


def instrument_pipeline():
    """Automatically instrument key pipeline functions."""
    # Instrument ingestion pipeline
    auto_instrument_functions("ingestion.ingest_repo", ["clone_or_update_repo"])
    auto_instrument_functions("ingestion.chunk_code", ["chunk_repository"])
    auto_instrument_functions(
        "ingestion.embed_chunks_into_vectorstore", ["embed_documents"]
    )

    # Instrument agent functions
    auto_instrument_functions("langgraph_flow.graph_builder", ["build_graph"])
    auto_instrument_functions(
        "utils.agent_utils",
        ["get_relevant_code_context_chunks_from_vectorstore", "run_llm"],
    )
