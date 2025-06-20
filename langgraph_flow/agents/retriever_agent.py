import logging
from typing import List, Dict

from langchain.schema import Document

from ingestion.load_vectorstore import load_vectorstore

logger = logging.getLogger(__name__)


def retrieve_code(state: Dict) -> Dict:
    """
    Query the vectorstore for the top-k code chunks relevant to the user's question.

    Expects in state:
      - "question": str
      - "cfg": dict (the full config)

    Returns updated state with:
      - "response": str   # formatted top results
    """
    question = state.get("question", "").strip()
    cfg = state.get("cfg", {})
    top_k = cfg.get("retriever", {}).get("top_k", 5)

    try:
        store = load_vectorstore(cfg)
        logger.info("Performing similarity search (k=%d) for question: %s", top_k, question)
        docs: List[Document] = store.similarity_search(question, k=top_k)

        if not docs:
            logger.warning("No documents found for query: %s", question)
            response = "I couldn't find any relevant code snippets."
        else:
            # Format each snippet with its source path and chunk index
            pieces = []
            for doc in docs:
                meta = doc.metadata or {}
                src = meta.get("source", "unknown")
                idx = meta.get("chunk", "?")
                snippet = doc.page_content.strip()
                pieces.append(f"---\n**{src} (chunk {idx})**\n\n```\n{snippet}\n```")

            response = (
                f"Here are the top {len(docs)} relevant code snippets:\n\n"
                + "\n\n".join(pieces)
            )

        return {**state, "response": response}

    except Exception as e:
        logger.error("Error during code retrieval: %s", e, exc_info=True)
        return {**state, "response": "Sorry, I ran into an error while searching the codebase."}
