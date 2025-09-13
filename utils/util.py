import logging
import sys

import toml

from utils.constants import (
    KEY_CONFIG,
    KEY_EXIT,
    KEY_INFO,
    KEY_QUESTION,
    KEY_QUIT,
    LOG_FORMAT_STYLE,
)
from utils.performance_monitor import instrument_pipeline, monitor

# Auto-instrument functions BEFORE importing them
instrument_pipeline()

from ingestion.chunk_code import chunk_repository
from ingestion.embed_chunks_into_vectorstore import embed_documents
from ingestion.ingest_repo import clone_or_update_repo
from langgraph_flow.graph_builder import build_graph

logger = logging.getLogger(__name__)


def setup_logging(level: str = KEY_INFO, log_file: str = None):
    """
    Configure the root logger with a console handler and optional file handler.
    """
    root = logging.getLogger()
    root.setLevel(level.upper())

    fmt = logging.Formatter(LOG_FORMAT_STYLE)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # File handler (if specified)
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        root.addHandler(fh)


def load_config(path: str):
    """
    Load a TOML config file. Exit on error.
    """
    try:
        cfg = toml.load(path)
        logging.debug(f"Loaded configuration from {path}")
        return cfg
    except Exception as e:
        logging.error(f"Failed to load configuration '{path}': {e}")
        sys.exit(1)


def _get_llm_metrics(operation: str, stats: dict) -> list:
    """Extract LLM-specific metrics for display."""
    extras = []
    if "llm" in operation.lower() or "run_llm" in operation:
        if "total_tokens_input" in stats and "total_tokens_output" in stats:
            total_tokens = (
                stats["total_tokens_input"] + stats["total_tokens_output"]
            )
            extras.append(
                f"~{total_tokens} tokens ({stats['total_tokens_input']}+{stats['total_tokens_output']})"
            )
        elif "total_tokens_input" in stats:
            extras.append(f"~{stats['total_tokens_input']} input tokens")

        if "total_cost_usd" in stats:
            extras.append(f"~${stats['total_cost_usd']:.4f}")
    return extras


def _get_vectorstore_metrics(operation: str, stats: dict) -> list:
    """Extract vector store-specific metrics for display."""
    extras = []
    vectorstore_keywords = [
        "vectorstore",
        "similarity_search",
        "get_relevant_code",
    ]
    if any(keyword in operation.lower() for keyword in vectorstore_keywords):
        if "total_vectorstore_results" in stats:
            extras.append(f"{stats['total_vectorstore_results']} results")
        if "avg_results_per_query" in stats:
            extras.append(f"{stats['avg_results_per_query']:.1f} avg/query")
    return extras


def _get_embedding_metrics(operation: str, stats: dict) -> list:
    """Extract embedding-specific metrics for display."""
    extras = []
    if "embed" in operation.lower():
        if (
            stats.get("count", 0) > 0
            and stats.get("avg_duration_seconds", 0) > 0
        ):
            extras.append("embedding rate tracked")
    return extras


def _get_cache_metrics(stats: dict) -> list:
    """Extract cache-specific metrics for display."""
    extras = []
    if "cache_hit_rate" in stats:
        extras.append(f"{stats['cache_hit_rate']:.1%} cache hit")
    return extras


def _format_operation_summary(operation: str, stats: dict) -> str:
    """Format a single operation's performance summary with relevant metrics."""
    base_info = f"  {operation}: {stats['avg_duration_seconds']:.2f}s avg, {stats['max_memory_mb']:.1f}MB peak"

    # Collect all relevant metrics
    extras = []
    extras.extend(_get_llm_metrics(operation, stats))
    extras.extend(_get_vectorstore_metrics(operation, stats))
    extras.extend(_get_embedding_metrics(operation, stats))
    extras.extend(_get_cache_metrics(stats))

    if extras:
        base_info += f" ({', '.join(extras)})"

    return base_info


def _print_performance_summary(title: str = "📊 Performance Summary:"):
    """Print formatted performance summary with operation-specific metrics."""
    summary = monitor.get_report().get_summary()
    if summary:
        logger.info(title)
        for operation, stats in summary.items():
            formatted_line = _format_operation_summary(operation, stats)
            logger.info(formatted_line)


def ingest_flow(cfg: dict):
    """
    Ingestion pipeline:
      1. Clone the repository
      2. Chunk source files
      3. Embed chunks into the vector store
    """
    monitor.configure(cfg)

    with monitor.measure("ingest_pipeline_total"):
        logger.info("🔄 Starting ingestion pipeline")
        repo_path = clone_or_update_repo(cfg)
        docs = chunk_repository(repo_path)
        embed_documents(docs, cfg)
        logger.info("✅ Ingestion pipeline completed")

    _print_performance_summary()


def _format_chat_summary_line(operation: str, stats: dict) -> str:
    """Format a single operation's chat session summary with call counts."""
    base_info = f"  {operation}: {stats['count']} calls, {stats['avg_duration_seconds']:.2f}s avg"

    # Collect all relevant metrics
    extras = []
    extras.extend(_get_llm_metrics(operation, stats))
    extras.extend(_get_vectorstore_metrics(operation, stats))
    extras.extend(_get_cache_metrics(stats))

    if extras:
        base_info += f" ({', '.join(extras)})"

    return base_info


def _print_chat_summary():
    """Print chat session performance summary on exit."""
    summary = monitor.get_report().get_summary()
    if summary:
        logger.info("📊 Chat Session Performance Summary:")
        for operation, stats in summary.items():
            formatted_line = _format_chat_summary_line(operation, stats)
            logger.info(formatted_line)


def _handle_user_question(graph, question: str, cfg: dict):
    """Process a single user question through the graph."""
    with monitor.measure("chat_query", {"question_length": len(question)}):
        state = graph.invoke({KEY_QUESTION: question, KEY_CONFIG: cfg})
        response = state.get("response", "No answer available.")
        logger.info(f"\n💡 {response}\n")


def _should_exit(question: str) -> bool:
    """Check if user wants to exit the chat."""
    return question.lower() in (KEY_EXIT, KEY_QUIT)


def chat_flow(cfg: dict):
    """
    Interactive chat loop:
      - Builds the LangGraph flow
      - Prompts the user for questions
      - Routes through agents and prints responses
    """
    monitor.configure(cfg)

    logger.info("🔧 Building LangGraph flow")
    graph = build_graph()
    logger.info(
        f"💬 Entering interactive chat (type {KEY_EXIT} or {KEY_QUIT} to stop)"
    )

    try:
        while True:
            question = input("\n❓ Ask your codebase: ").strip()

            if _should_exit(question):
                logger.info("👋 Exiting chat loop")
                _print_chat_summary()
                break

            try:
                _handle_user_question(graph, question, cfg)
            except Exception:
                logger.exception("Error during graph execution")

    except KeyboardInterrupt:
        logger.info("⚡ Chat interrupted by user")
