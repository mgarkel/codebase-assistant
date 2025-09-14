"""Performance monitoring utilities and decorators."""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def measure_performance(operation: str = None, metadata: Optional[Dict] = None):
    """Decorator to measure function performance."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            from .monitor import (  # Import here to avoid circular imports
                monitor,
            )

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
    # Note: embed_documents now uses measure_embedding_operation internally

    # Instrument agent functions
    auto_instrument_functions("langgraph_flow.graph_builder", ["build_graph"])
    # Note: run_llm and get_relevant_code_context_chunks_from_vectorstore now use
    # specialized context managers internally
