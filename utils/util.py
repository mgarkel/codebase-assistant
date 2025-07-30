import logging
import sys

import toml

from ingestion.chunk_code import chunk_repository
from ingestion.embed_chunks_into_vectorstore import embed_documents
from ingestion.ingest_repo import clone_or_update_repo
from langgraph_flow.graph_builder import build_graph
from utils.constants import (
    KEY_CONFIG,
    KEY_EXIT,
    KEY_INFO,
    KEY_QUESTION,
    KEY_QUIT,
    LOG_FORMAT_STYLE,
)

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


def ingest_flow(cfg: dict):
    """
    Ingestion pipeline:
      1. Clone the repository
      2. Chunk source files
      3. Embed chunks into the vector store
    """
    logger.info("🔄 Starting ingestion pipeline")
    dest_path = clone_or_update_repo(cfg)
    docs = chunk_repository(dest_path)
    embed_documents(docs, cfg)
    logger.info("✅ Ingestion pipeline completed")


def chat_flow(cfg: dict):
    """
    Interactive chat loop:
      - Builds the LangGraph flow
      - Prompts the user for questions
      - Routes through agents and prints responses
    """
    logger.info("🔧 Building LangGraph flow")
    graph = build_graph()
    logger.info(
        f"💬 Entering interactive chat (type {KEY_EXIT} or {KEY_QUIT} to stop)"
    )

    try:
        while True:
            question = input("\n❓ Ask your codebase: ").strip()
            if question.lower() in (KEY_EXIT, KEY_QUIT):
                logging.info("👋 Exiting chat loop")
                break

            try:
                # Pass both the question and the full config into the graph state
                state = graph.invoke({KEY_QUESTION: question, KEY_CONFIG: cfg})
                response = state.get("response", "No answer available.")
                logger.info(f"\n💡 {response}\n")
            except Exception:
                logger.exception("Error during graph execution")
    except KeyboardInterrupt:
        logger.info("⚡ Chat interrupted by user")
