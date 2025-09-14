"""Embedding operation performance monitoring."""

import logging
from contextlib import contextmanager

from .base import _create_base_metric, _get_memory_measurements

logger = logging.getLogger(__name__)


@contextmanager
def measure_embedding_operation(operation: str, batch_size: int):
    """Context manager for measuring embedding generation."""
    # Import here to avoid circular imports
    from ..monitor import monitor

    if not monitor.enabled or not monitor._should_monitor_operation(operation):
        yield
        return

    import time

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
