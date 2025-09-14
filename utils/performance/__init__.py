"""
Performance monitoring module for the codebase assistant.

This module provides comprehensive performance monitoring capabilities including:
- LLM operation tracking with token counting and cost calculation
- Vector store operation monitoring with result counting
- Memory and duration tracking for all operations
- Configurable cost calculation via settings.toml
- Flexible reporting and analysis

Main components:
- monitor: Global performance monitoring singleton
- Various monitor classes for different operation types
- Analysis and reporting utilities
- Backward-compatible API
"""

from .analysis import PerformanceAnalyzer, PerformanceReport

# Configuration and utilities
from .config import PerformanceConfig
from .cost_calculator import TokenCostCalculator

# Deferred updates
from .deferred_updates import TokenResponseUpdate, VectorStoreResultsUpdate

# Data models and storage
from .models import DeferredUpdate, MetricUpdater, PerformanceMetric

# Core monitoring
from .monitor import PerformanceMonitor, monitor

# Specialized monitors
from .monitors import (
    BaseMonitor,
    LLMMonitor,
    VectorStoreMonitor,
    measure_embedding_operation,
    measure_llm_operation,
    measure_vectorstore_operation,
)
from .storage import PerformanceMetrics

# Utilities and decorators
from .utils import (
    auto_instrument_functions,
    instrument_pipeline,
    measure_performance,
)

# Backward compatibility - these are the main exports that existing code expects
__all__ = [
    # Main monitor
    "monitor",
    "PerformanceMonitor",
    # Data models
    "PerformanceMetric",
    "PerformanceReport",
    "PerformanceMetrics",
    "PerformanceAnalyzer",
    # Monitors
    "BaseMonitor",
    "LLMMonitor",
    "VectorStoreMonitor",
    # Monitor factory functions
    "measure_llm_operation",
    "measure_vectorstore_operation",
    "measure_embedding_operation",
    # Utilities
    "measure_performance",
    "auto_instrument_functions",
    "instrument_pipeline",
    # Configuration
    "PerformanceConfig",
    "TokenCostCalculator",
    # Advanced
    "DeferredUpdate",
    "TokenResponseUpdate",
    "VectorStoreResultsUpdate",
    "MetricUpdater",
]
