import os
import logging
from typing import List, Dict

from langchain.chat_models import ChatOpenAI
from langchain.schema import Document

from ingestion.load_vectorstore import load_vectorstore

logger = logging.getLogger(__name__)


def explain_code(state: Dict) -> Dict:
    """
    Retrieve top-k relevant code snippets and generate a natural-language explanation.

    Expects in state:
      - "question": str
      - "cfg": dict

    Returns updated state with:
      - "response": str  # explanation text
    """
    question = state.get("question", "").strip()
    cfg = state.get("cfg", {})

    # Load vectorstore
    try:
        store = load_vectorstore(cfg)
    except Exception as e:
        logger.error(
            "Failed to load vectorstore for explanation: %s", e, exc_info=True
        )
        return {**state, "response": "Error: could not access the code index."}

    # Determine how many snippets to explain
    top_k = cfg.get("explainer", {}).get("top_k", 3)

    # Perform similarity search
    try:
        logger.info(
            "Retrieving top %d snippets for explanation of: %s", top_k, question
        )
        docs: List[Document] = store.similarity_search(question, k=top_k)
    except Exception as e:
        logger.error(
            "Similarity search failed in explainer_agent: %s", e, exc_info=True
        )
        return {
            **state,
            "response": "Error: failed to retrieve code snippets for explanation.",
        }

    if not docs:
        logger.warning("No snippets found for explanation query: %s", question)
        return {
            **state,
            "response": "I couldn't find any relevant code to explain.",
        }

    # Concatenate snippets with source info
    combined = []
    for doc in docs:
        meta = doc.metadata or {}
        src = meta.get("source", "unknown")
        idx = meta.get("chunk", "?")
        snippet = doc.page_content.strip()
        combined.append(f"# Source: {src} (chunk {idx})\n{snippet}")

    code_context = "\n\n".join(combined)

    # Load prompt template
    tmpl_path = os.path.join(
        os.path.dirname(__file__), "..", "prompts", "explanation_prompt.txt"
    )
    with open(tmpl_path, "r", encoding="utf-8") as f:
        template = f.read()

    prompt = template.format(code=code_context)

    # Initialize LLM
    model_name = cfg.get("openai", {}).get("inference_model", "gpt-4")
    openai_api_key = os.getenv("OPENAPI_KEY")
    llm = ChatOpenAI(
        model=model_name, temperature=0, openai_api_key=openai_api_key
    )

    # Generate explanation
    try:
        logger.debug("Sending explanation prompt to LLM")
        explanation = llm.predict(prompt).strip()
    except Exception as e:
        logger.error("LLM explanation generation failed: %s", e, exc_info=True)
        return {**state, "response": "Error: failed to generate explanation."}

    logger.info("Generated explanation successfully")
    return {**state, "response": explanation}
