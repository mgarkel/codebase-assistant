import argparse
import logging

from utils.constants import KEY_CHAT, KEY_INGEST
from utils.util import chat_flow, ingest_flow, load_config, setup_logging


def main():
    parser = argparse.ArgumentParser(
        prog="codebase-assistant",
        description="LLM-powered assistant for navigating codebases",
    )
    parser.add_argument(
        "command",
        choices=[KEY_INGEST, KEY_CHAT],
        nargs="?",
        default=KEY_CHAT,
        help=f"Mode: '{KEY_INGEST}' to build embeddings, '{KEY_CHAT}' to start interactive Q&A",
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
    if args.command == KEY_INGEST:
        ingest_flow(cfg)
    else:
        chat_flow(cfg)


if __name__ == "__main__":
    # TODO - Delete previous embedding before adding new ones - need to fix code contexts returned.
    main()
