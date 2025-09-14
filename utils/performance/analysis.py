"""Performance metrics analysis and statistics generation."""

import json
from typing import Dict, List, Optional

from .models import PerformanceMetric
from .storage import PerformanceMetrics


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
