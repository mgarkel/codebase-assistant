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

    # Advanced metrics
    tokens_input: Optional[int] = None
    tokens_output: Optional[int] = None
    tokens_cost_usd: Optional[float] = None
    vectorstore_query_count: Optional[int] = None
    vectorstore_results_count: Optional[int] = None
    embedding_batch_size: Optional[int] = None
    cache_hits: Optional[int] = None
    cache_misses: Optional[int] = None


@dataclass
class PerformanceReport:
    """Collection of performance metrics."""

    metrics: List[PerformanceMetric] = field(default_factory=list)

    def add_metric(self, metric: PerformanceMetric):
        """Add a performance metric to the report."""
        self.metrics.append(metric)

    def _group_metrics_by_operation(self) -> Dict[str, List[PerformanceMetric]]:
        """Group metrics by operation name."""
        by_operation = {}
        for metric in self.metrics:
            if metric.operation not in by_operation:
                by_operation[metric.operation] = []
            by_operation[metric.operation].append(metric)
        return by_operation

    def _get_basic_stats(self, metrics: List[PerformanceMetric]) -> Dict:
        """Calculate basic duration and memory statistics."""
        durations = [m.duration_seconds for m in metrics]
        memory_usage = [m.memory_mb for m in metrics]

        return {
            "count": len(metrics),
            "avg_duration_seconds": sum(durations) / len(durations),
            "max_duration_seconds": max(durations),
            "min_duration_seconds": min(durations),
            "avg_memory_mb": sum(memory_usage) / len(memory_usage),
            "max_memory_mb": max(memory_usage),
        }

    def _add_token_stats(
        self, op_summary: Dict, metrics: List[PerformanceMetric]
    ):
        """Add LLM token statistics to operation summary."""
        tokens_input = [
            m.tokens_input for m in metrics if m.tokens_input is not None
        ]
        tokens_output = [
            m.tokens_output for m in metrics if m.tokens_output is not None
        ]
        tokens_cost = [
            m.tokens_cost_usd for m in metrics if m.tokens_cost_usd is not None
        ]

        if tokens_input:
            op_summary["total_tokens_input"] = sum(tokens_input)
            op_summary["avg_tokens_input"] = sum(tokens_input) / len(
                tokens_input
            )
        if tokens_output:
            op_summary["total_tokens_output"] = sum(tokens_output)
            op_summary["avg_tokens_output"] = sum(tokens_output) / len(
                tokens_output
            )
        if tokens_cost:
            op_summary["total_cost_usd"] = sum(tokens_cost)
            op_summary["avg_cost_usd"] = sum(tokens_cost) / len(tokens_cost)

    def _add_vectorstore_stats(
        self, op_summary: Dict, metrics: List[PerformanceMetric]
    ):
        """Add vector store statistics to operation summary."""
        vs_queries = [
            m.vectorstore_query_count
            for m in metrics
            if m.vectorstore_query_count is not None
        ]
        vs_results = [
            m.vectorstore_results_count
            for m in metrics
            if m.vectorstore_results_count is not None
        ]

        if vs_queries:
            op_summary["total_vectorstore_queries"] = sum(vs_queries)
        if vs_results:
            op_summary["total_vectorstore_results"] = sum(vs_results)
            op_summary["avg_results_per_query"] = sum(vs_results) / len(
                vs_results
            )

    def _add_cache_stats(
        self, op_summary: Dict, metrics: List[PerformanceMetric]
    ):
        """Add cache statistics to operation summary."""
        cache_hits = [m.cache_hits for m in metrics if m.cache_hits is not None]
        cache_misses = [
            m.cache_misses for m in metrics if m.cache_misses is not None
        ]

        if cache_hits and cache_misses:
            total_hits = sum(cache_hits)
            total_misses = sum(cache_misses)
            total_requests = total_hits + total_misses
            if total_requests > 0:
                op_summary["cache_hit_rate"] = total_hits / total_requests
                op_summary["cache_total_requests"] = total_requests

    def get_summary(self) -> Dict:
        """Get summary statistics for all metrics."""
        if not self.metrics:
            return {}

        by_operation = self._group_metrics_by_operation()
        summary = {}

        for operation, metrics in by_operation.items():
            op_summary = self._get_basic_stats(metrics)

            # Add advanced metrics
            self._add_token_stats(op_summary, metrics)
            self._add_vectorstore_stats(op_summary, metrics)
            self._add_cache_stats(op_summary, metrics)

            summary[operation] = op_summary

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


# Advanced measurement functions
def _create_base_metric(
    operation: str,
    start_time: float,
    end_time: float,
    start_memory: float,
    end_memory: float,
    metadata: Dict = None,
) -> PerformanceMetric:
    """Create a base performance metric with common fields."""
    return PerformanceMetric(
        operation=operation,
        duration_seconds=end_time - start_time,
        memory_mb=max(end_memory, start_memory),
        timestamp=datetime.now(),
        metadata=metadata or {},
    )


def _get_memory_measurements() -> tuple:
    """Get current memory usage in MB."""
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    return memory_mb


class LLMMonitor:
    """Helper class to monitor LLM operations with token and cost tracking."""

    def __init__(
        self, operation: str, input_text: str, model_name: str = "unknown"
    ):
        self.operation = operation
        self.input_text = input_text
        self.model_name = model_name
        self.start_time = None
        self.start_memory = None

    def __enter__(self):
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

        end_time = time.time()
        end_memory = _get_memory_measurements()

        # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
        estimated_input_tokens = len(self.input_text) // 4
        metadata = {
            "model": self.model_name,
            "input_length": len(self.input_text),
        }

        metric = _create_base_metric(
            self.operation,
            self.start_time,
            end_time,
            self.start_memory,
            end_memory,
            metadata,
        )
        metric.tokens_input = estimated_input_tokens

        monitor.report.add_metric(metric)
        monitor._check_performance_warnings(metric)
        logger.debug(
            f"LLM: {self.operation} took {metric.duration_seconds:.3f}s, "
            f"~{estimated_input_tokens} input tokens"
        )

    def set_response(self, response_text: str):
        """Update the metric with response information for output tokens and cost."""
        if (
            self.enabled
            and hasattr(monitor.report, "metrics")
            and monitor.report.metrics
        ):
            # Update the most recent metric (which should be ours)
            latest_metric = monitor.report.metrics[-1]

            if latest_metric.operation == self.operation:
                # Estimate output tokens
                estimated_output_tokens = len(response_text) // 4
                latest_metric.tokens_output = estimated_output_tokens

                # Estimate cost (rough pricing for GPT-4: $0.03/1k input, $0.06/1k output)
                if "gpt-4" in self.model_name.lower():
                    input_cost = (latest_metric.tokens_input or 0) * 0.03 / 1000
                    output_cost = estimated_output_tokens * 0.06 / 1000
                    latest_metric.tokens_cost_usd = input_cost + output_cost
                elif "gpt-3.5" in self.model_name.lower():
                    # GPT-3.5-turbo pricing: $0.0015/1k input, $0.002/1k output
                    input_cost = (
                        (latest_metric.tokens_input or 0) * 0.0015 / 1000
                    )
                    output_cost = estimated_output_tokens * 0.002 / 1000
                    latest_metric.tokens_cost_usd = input_cost + output_cost

                logger.debug(
                    f"LLM response processed: {estimated_output_tokens} output tokens, model: {self.model_name}, cost: ${latest_metric.tokens_cost_usd or 0:.4f}"
                )


def measure_llm_operation(
    operation: str, input_text: str, model_name: str = "unknown"
):
    """Create an LLM monitor context manager."""
    return LLMMonitor(operation, input_text, model_name)


class VectorStoreMonitor:
    """Helper class to monitor vector store operations with result counting."""

    def __init__(self, operation: str, query: str, top_k: int = 5):
        self.operation = operation
        self.query = query
        self.top_k = top_k
        self.start_time = None
        self.start_memory = None
        self.metric = None  # Store reference to our specific metric

    def __enter__(self):
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

        end_time = time.time()
        end_memory = _get_memory_measurements()

        # Default to 0 results since we can't capture the result automatically
        metadata = {"query_length": len(self.query), "requested_k": self.top_k}

        self.metric = _create_base_metric(
            self.operation,
            self.start_time,
            end_time,
            self.start_memory,
            end_memory,
            metadata,
        )
        self.metric.vectorstore_query_count = 1
        self.metric.vectorstore_results_count = (
            0  # Will be updated by set_results if called
        )

        monitor.report.add_metric(self.metric)
        monitor._check_performance_warnings(self.metric)
        logger.debug(
            f"VectorStore: {self.operation} took {self.metric.duration_seconds:.3f}s, "
            f"returned {self.metric.vectorstore_results_count}/{self.top_k} results"
        )

    def set_results(self, results):
        """Update the result count after the operation."""
        if self.enabled and self.metric is not None:
            results_count = len(results) if hasattr(results, "__len__") else 0
            self.metric.vectorstore_results_count = results_count
            logger.debug(
                f"VectorStore results updated: {results_count} results for operation {self.operation}"
            )


def measure_vectorstore_operation(operation: str, query: str, top_k: int = 5):
    """Create a vector store monitor context manager."""
    return VectorStoreMonitor(operation, query, top_k)


@contextmanager
def measure_embedding_operation(operation: str, batch_size: int):
    """Context manager for measuring embedding generation."""
    if not monitor.enabled or not monitor._should_monitor_operation(operation):
        yield
        return

    start_time = time.time()
    start_memory = _get_memory_measurements()

    yield

    end_time = time.time()
    end_memory = _get_memory_measurements()

    rate = batch_size / (end_time - start_time) if end_time > start_time else 0
    metadata = {"embeddings_per_second": rate}

    metric = _create_base_metric(
        operation, start_time, end_time, start_memory, end_memory, metadata
    )
    metric.embedding_batch_size = batch_size

    monitor.report.add_metric(metric)
    monitor._check_performance_warnings(metric)

    if rate > 0:
        logger.debug(
            f"Embeddings: {operation} took {metric.duration_seconds:.3f}s, "
            f"batch_size={batch_size}, rate={rate:.1f}/s"
        )


def instrument_pipeline():
    """Automatically instrument key pipeline functions."""
    # Instrument ingestion pipeline
    auto_instrument_functions("ingestion.ingest_repo", ["clone_or_update_repo"])
    auto_instrument_functions("ingestion.chunk_code", ["chunk_repository"])
    # Note: embed_documents now uses measure_embedding_operation internally

    # Instrument agent functions
    auto_instrument_functions("langgraph_flow.graph_builder", ["build_graph"])
    # Note: run_llm and get_relevant_code_context_chunks_from_vectorstore now use
    # specialized context managers internally
