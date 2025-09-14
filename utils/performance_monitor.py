"""Performance monitoring utilities for the codebase assistant."""

import json
import logging
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Protocol

import psutil

logger = logging.getLogger(__name__)


# Configuration and Constants
class PerformanceConfig:
    """Configuration constants for performance monitoring."""

    # Token estimation
    CHARS_PER_TOKEN = 4

    # Default thresholds
    DEFAULT_MEMORY_WARNING_MB = 1000
    DEFAULT_DURATION_WARNING_SECONDS = 30


# Protocols for type safety
class MetricUpdater(Protocol):
    """Protocol for classes that can update metrics with additional data."""

    def update_metric(self, metric: "PerformanceMetric") -> None:
        """Update the metric with additional data."""
        ...


# Deferred update system
class DeferredUpdate(ABC):
    """Base class for updates that should be applied when the metric becomes available."""

    @abstractmethod
    def apply(self, metric: "PerformanceMetric") -> None:
        """Apply this update to the given metric."""
        pass


class TokenResponseUpdate(DeferredUpdate):
    """Deferred update for LLM response token and cost information."""

    def __init__(
        self,
        response_text: str,
        model_name: str,
        cost_calculator: "TokenCostCalculator",
    ):
        self.response_text = response_text
        self.model_name = model_name
        self.cost_calculator = cost_calculator

    def apply(self, metric: "PerformanceMetric") -> None:
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

    def apply(self, metric: "PerformanceMetric") -> None:
        """Apply results count to the metric."""
        metric.vectorstore_results_count = self.results_count


# Cost calculation strategy
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


class PerformanceAnalyzer:
    """Analyzes performance metrics and generates statistics."""

    def __init__(self, metrics: PerformanceMetrics):
        self.metrics = metrics

    def _group_metrics_by_operation(self) -> Dict[str, List[PerformanceMetric]]:
        """Group metrics by operation name."""
        by_operation = {}
        for metric in self.metrics.metrics:
            if metric.operation not in by_operation:
                by_operation[metric.operation] = []
            by_operation[metric.operation].append(metric)
        return by_operation

    def _get_basic_stats(self, metrics: List[PerformanceMetric]) -> Dict:
        """Calculate basic duration and memory statistics."""
        if not metrics:
            return {}

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
        if not self.metrics.metrics:
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
                for m in self.metrics.metrics
            ],
        }
        return json.dumps(data, indent=2)

    def get_operation_trends(
        self, operation: str, limit: Optional[int] = None
    ) -> List[Dict]:
        """Get time-series trend data for a specific operation."""
        operation_metrics = [
            m for m in self.metrics.metrics if m.operation == operation
        ]
        operation_metrics.sort(key=lambda x: x.timestamp)

        if limit:
            operation_metrics = operation_metrics[-limit:]

        return [
            {
                "timestamp": m.timestamp.isoformat(),
                "duration_seconds": m.duration_seconds,
                "memory_mb": m.memory_mb,
                "tokens_input": m.tokens_input,
                "tokens_output": m.tokens_output,
                "tokens_cost_usd": m.tokens_cost_usd,
            }
            for m in operation_metrics
        ]


# Backward compatibility wrapper
class PerformanceReport:
    """Backward-compatible wrapper that combines storage and analysis."""

    def __init__(self):
        self.metrics = PerformanceMetrics()
        self._analyzer = None

    @property
    def analyzer(self) -> PerformanceAnalyzer:
        """Lazy-loaded analyzer instance."""
        if self._analyzer is None:
            self._analyzer = PerformanceAnalyzer(self.metrics)
        return self._analyzer

    def add_metric(self, metric: PerformanceMetric):
        """Add a performance metric to the report."""
        self.metrics.add_metric(metric)
        # Reset analyzer to pick up new data
        self._analyzer = None

    def get_summary(self) -> Dict:
        """Get summary statistics for all metrics."""
        return self.analyzer.get_summary()

    def to_json(self) -> str:
        """Export metrics as JSON."""
        return self.analyzer.to_json()


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


def _get_memory_measurements() -> float:
    """Get current memory usage in MB."""
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    return memory_mb


# Base monitor class to eliminate code duplication
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
