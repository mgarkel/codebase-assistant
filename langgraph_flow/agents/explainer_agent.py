import os
import logging
from typing import List, Dict

from langchain.chat_models import ChatOpenAI
from langchain.schema import Document

from ingestion.load_vectorstore import load_vectorstore
from langgraph_flow.models.assistant_state import AssistantState
from utils.agent_utils import (
    OpenAIModel,
    get_combined_text_from_docs,
    get_question_and_config_from_state,
)
from utils.constants import (
    DEFAULT_TOP_K_EXPLAINER,
    ENV_OPENAIAPI_KEY,
    KEY_CHUNK,
    KEY_CONFIG,
    KEY_CONFIG_EXPLAINER,
    KEY_CONFIG_TOP_K,
    KEY_INFERENCE_MODEL,
    KEY_OPENAI,
    KEY_QUESTION,
    KEY_RESPONSE,
    KEY_SOURCE,
    KEY_UNKNOWN,
    MODEL_INFERENCE_OPEN_AI,
    VALUES_UTF_8,
)

logger = logging.getLogger(__name__)


def explain_code(state: AssistantState) -> Dict:
    """
    Retrieve top-k relevant code snippets and generate a natural-language explanation.

    Expects in state:
      - "question": str
      - "cfg": dict

    Returns updated state with:
      - "response": str  # explanation text
    """
    question, cfg = get_question_and_config_from_state(state)

    # Load vectorstore
    try:
        store = load_vectorstore(cfg)
    except Exception as e:
        logger.error(
            "Failed to load vectorstore for explanation: %s", e, exc_info=True
        )
        return {
            **state,
            KEY_RESPONSE: "Error: could not access the code index.",
        }

    # Determine how many snippets to explain
    top_k = cfg.get(KEY_CONFIG_EXPLAINER, {}).get(
        KEY_CONFIG_TOP_K, DEFAULT_TOP_K_EXPLAINER
    )

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
            KEY_RESPONSE: "Error: failed to retrieve code snippets for explanation.",
        }

    if not docs:
        logger.warning("No snippets found for explanation query: %s", question)
        return {
            **state,
            KEY_RESPONSE: "I couldn't find any relevant code to explain.",
        }

    code_context = get_combined_text_from_docs(docs)

    # Load prompt template
    tmpl_path = os.path.join(
        os.path.dirname(__file__), "..", "prompts", "explanation_prompt.txt"
    )
    # TODO Extract this prompt into a utility function that can be used to grab all prompts
    with open(tmpl_path, "r", encoding=VALUES_UTF_8) as f:
        template = f.read()

    prompt = template.format(code=code_context)

    # Initialize LLM
    llm = OpenAIModel(cfg).inference_model

    # Generate explanation
    try:
        logger.debug("Sending explanation prompt to LLM")
        explanation = llm.predict(prompt).strip()
    except Exception as e:
        logger.error("LLM explanation generation failed: %s", e, exc_info=True)
        return {**state, KEY_RESPONSE: "Error: failed to generate explanation."}

    logger.info("Generated explanation successfully")
    return {**state, KEY_RESPONSE: explanation}
