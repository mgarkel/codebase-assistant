import argparse
import logging
from utils.util import chat_flow, ingest_flow, load_config, setup_logging


def main():
    parser = argparse.ArgumentParser(
        prog="codebase-assistant",
        description="LLM-powered assistant for navigating codebases",
    )
    parser.add_argument(
        "command",
        choices=["ingest", "chat"],
        nargs="?",
        default="chat",
        help="Mode: 'ingest' to build embeddings, 'chat' to start interactive Q&A",
    )
    parser.add_argument(
        "-c",
        "--config",
        default="config/settings.toml",
        help="Path to the TOML configuration file",
    )
    parser.add_argument(
        "-l",
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    parser.add_argument("--log-file", help="Optional file to write logs to")

    args = parser.parse_args()

    # Initialize logging first
    setup_logging(args.log_level, args.log_file)
    logging.debug(f"Arguments: {args}")

    # Load project config
    cfg = load_config(args.config)

    # Dispatch based on command
    if args.command == "ingest":
        ingest_flow(cfg)
    else:
        chat_flow(
            cfg
        )  # TODO - add response formatter for each agent, add template format for each of the prompts


if __name__ == "__main__":
    main()
