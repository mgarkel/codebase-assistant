import logging
from typing import List, Dict

from langchain.schema import Document

from ingestion.load_vectorstore import load_vectorstore
from langgraph_flow.models.assistant_state import AssistantState
from utils.agent_utils import (
    get_combined_text_from_docs,
    get_question_and_config_from_state,
)
from utils.constants import (
    DEFAULT_TOK_K_RETRIEVER,
    KEY_CHUNK,
    KEY_CONFIG,
    KEY_CONFIG_RETRIEVER,
    KEY_CONFIG_TOP_K,
    KEY_QUESTION,
    KEY_RESPONSE,
    KEY_SOURCE,
    KEY_UNKNOWN,
)

logger = logging.getLogger(__name__)


def retrieve_code(state: AssistantState) -> Dict:
    """
    Query the vectorstore for the top-k code chunks relevant to the user's question.

    Expects in state:
      - "question": str
      - "cfg": dict (the full config)

    Returns updated state with:
      - "response": str   # formatted top results
    """
    question, cfg = get_question_and_config_from_state(state)
    top_k = cfg.get(KEY_CONFIG_RETRIEVER, {}).get(
        KEY_CONFIG_TOP_K, DEFAULT_TOK_K_RETRIEVER
    )

    try:
        store = load_vectorstore(cfg)
        logger.info(
            "Performing similarity search (k=%d) for question: %s",
            top_k,
            question,
        )
        docs: List[Document] = store.similarity_search(question, k=top_k)

        if not docs:
            logger.warning("No documents found for query: %s", question)
            response = "I couldn't find any relevant code snippets."
        else:
            code_context = get_combined_text_from_docs(docs)
            response = (
                f"Here are the top {len(docs)} relevant code snippets:\n\n"
                + code_context
            )

        return {**state, KEY_RESPONSE: response}

    except Exception as e:
        logger.error("Error during code retrieval: %s", e, exc_info=True)
        return {
            **state,
            KEY_RESPONSE: "Sorry, I ran into an error while searching the codebase.",
        }
