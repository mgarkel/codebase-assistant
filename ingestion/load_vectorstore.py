import logging
import os
from typing import Dict

from langchain_chroma import Chroma

from langgraph_flow.models.openai_model import OpenAIModel
from utils.constants import (
    COLLECTION_NAME,
    KEY_PERSIST_DIRECTORY,
    KEY_VECTORSTORE,
)

logger = logging.getLogger(__name__)

# Cache the loaded vectorstore so we don’t re-open it on every call
_VECTORSTORE = None


def load_vectorstore(cfg: Dict):
    global _VECTORSTORE
    if _VECTORSTORE:
        return _VECTORSTORE

    store_cfg = cfg.get(KEY_VECTORSTORE, {})
    persist_dir = store_cfg.get(
        KEY_PERSIST_DIRECTORY, "vectorstore/chroma_index/"
    )
    os.makedirs(persist_dir, exist_ok=True)
    embeddings = OpenAIModel(cfg).embedding_model
    try:
        logger.info("Loading Chroma vectorstore from '%s'", persist_dir)
        _VECTORSTORE = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME,
        )
        logger.info("Vectorstore loaded successfully")
        return _VECTORSTORE

    except Exception as e:
        logger.error("Failed to load vectorstore: %s", e, exc_info=True)
        raise
