import sys
import logging

import toml

from ingestion.ingest_repo import clone_repo
from ingestion.chunk_code import chunk_repository
from ingestion.embed_chunks import embed_documents
from langgraph_flow.graph_builder import build_graph

logger = logging.getLogger(__name__)

def setup_logging(level: str = "INFO", log_file: str = None):
    """
    Configure the root logger with a console handler and optional file handler.
    """
    root = logging.getLogger()
    root.setLevel(level.upper())

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

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
    clone_repo(cfg["repo"]["url"], cfg["repo"]["local_path"])
    docs = chunk_repository(cfg["repo"]["local_path"])
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
    logger.info("💬 Entering interactive chat (type 'exit' or 'quit' to stop)")

    try:
        while True:
            question = input("\n❓ Ask your codebase: ").strip()
            if question.lower() in ("exit", "quit"):
                logging.info("👋 Exiting chat loop")
                break

            try:
                # Pass both the question and the full config into the graph state
                state = graph.invoke({"question": question, "cfg": cfg})
                response = state.get("response", "No answer available.")
                logger.info(f"\n💡 {response}\n")
            except Exception:
                logger.exception("Error during graph execution")
    except KeyboardInterrupt:
        logger.info("⚡ Chat interrupted by user")
