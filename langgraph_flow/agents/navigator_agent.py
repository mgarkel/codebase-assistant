import os
import logging
from typing import List, Dict

from langchain.chat_models import ChatOpenAI
from langchain.schema import Document

from ingestion.load_vectorstore import load_vectorstore

logger = logging.getLogger(__name__)


def navigate_code(state: Dict) -> Dict:
    """
    Trace symbol usage or workflow through the codebase.

    Expects in state:
      - "question": str   # typically a navigation request, e.g., "Where is X implemented?"
      - "cfg": dict

    Returns updated state with:
      - "response": str   # navigation summary or trace steps
    """
    question = state.get("question", "").strip()
    cfg = state.get("cfg", {})

    # Load vectorstore
    try:
        store = load_vectorstore(cfg)
    except Exception as e:
        logger.error(
            "Failed to load vectorstore for navigation: %s", e, exc_info=True
        )
        return {**state, "response": "Error: could not access the code index."}

    # Determine how many snippets to consider for navigation
    top_k = cfg.get("navigator", {}).get("top_k", 5)

    # Perform similarity search to find relevant code contexts
    try:
        logger.info(
            "Retrieving top %d snippets for navigation of: %s", top_k, question
        )
        docs: List[Document] = store.similarity_search(question, k=top_k)
    except Exception as e:
        logger.error(
            "Similarity search failed in navigator_agent: %s", e, exc_info=True
        )
        return {
            **state,
            "response": "Error: failed to retrieve code snippets for navigation.",
        }

    if not docs:
        logger.warning("No snippets found for navigation query: %s", question)
        return {
            **state,
            "response": "I couldn't find any relevant code to navigate.",
        }

    # Combine snippets with metadata
    combined = []
    for doc in docs:
        meta = doc.metadata or {}
        src = meta.get("source", "unknown")
        idx = meta.get("chunk", "?")
        snippet = doc.page_content.strip()
        combined.append(f"# Source: {src} (chunk {idx})\n{snippet}")

    code_context = "\n\n".join(combined)

    # Load navigation prompt template
    tmpl_path = os.path.join(
        os.path.dirname(__file__), "..", "prompts", "navigation_prompt.txt"
    )
    with open(tmpl_path, "r", encoding="utf-8") as f:
        template = f.read()

    prompt = template.format(question=question, code=code_context)

    # Initialize LLM
    model_name = cfg.get("openai", {}).get("inference_model", "gpt-4")
    openai_api_key = os.getenv("OPENAPI_KEY")
    llm = ChatOpenAI(
        model=model_name, temperature=0, openai_api_key=openai_api_key
    )

    # Generate navigation summary
    try:
        logger.debug("Sending navigation prompt to LLM")
        navigation = llm.predict(prompt).strip()
    except Exception as e:
        logger.error("LLM navigation generation failed: %s", e, exc_info=True)
        return {
            **state,
            "response": "Error: failed to generate navigation summary.",
        }

    logger.info("Generated navigation summary successfully")
    return {**state, "response": navigation}
