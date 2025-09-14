"""
Performance monitoring utilities for the codebase assistant.

This module has been refactored into a modular structure under utils/performance/.
This file maintains backward compatibility by re-exporting all the main components.

For new code, prefer importing directly from utils.performance instead.
"""

# Explicitly import the main components that were previously defined in this file
# Import everything from the new modular structure to maintain backward compatibility
from utils.performance import *  # noqa: F403, F401
from utils.performance import (  # Core monitoring; Data models; Configuration; Monitors; Factory functions; Utilities; Advanced components
    BaseMonitor,
    DeferredUpdate,
    LLMMonitor,
    MetricUpdater,
    PerformanceAnalyzer,
    PerformanceConfig,
    PerformanceMetric,
    PerformanceMetrics,
    PerformanceMonitor,
    PerformanceReport,
    TokenCostCalculator,
    TokenResponseUpdate,
    VectorStoreMonitor,
    VectorStoreResultsUpdate,
    auto_instrument_functions,
    instrument_pipeline,
    measure_embedding_operation,
    measure_llm_operation,
    measure_performance,
    measure_vectorstore_operation,
    monitor,
)

# Export everything for backward compatibility
__all__ = [
    "monitor",
    "PerformanceMonitor",
    "PerformanceMetric",
    "PerformanceReport",
    "PerformanceMetrics",
    "PerformanceAnalyzer",
    "PerformanceConfig",
    "TokenCostCalculator",
    "BaseMonitor",
    "LLMMonitor",
    "VectorStoreMonitor",
    "measure_llm_operation",
    "measure_vectorstore_operation",
    "measure_embedding_operation",
    "measure_performance",
    "auto_instrument_functions",
    "instrument_pipeline",
    "DeferredUpdate",
    "TokenResponseUpdate",
    "VectorStoreResultsUpdate",
    "MetricUpdater",
]
