import logging
from typing import Dict

from langchain_chroma import Chroma

from ingestion.ingestion_util import (
    get_persist_dir_and_collection_name_from_config,
)
from langgraph_flow.models.openai_model import OpenAIModel

logger = logging.getLogger(__name__)

# Cache the loaded vectorstore so we don’t re-open it on every call
_VECTORSTORE = None


def load_vectorstore(cfg: Dict):
    global _VECTORSTORE
    if _VECTORSTORE:
        return _VECTORSTORE

    persist_dir, collection_name = (
        get_persist_dir_and_collection_name_from_config(cfg)
    )
    embeddings = OpenAIModel(cfg).embedding_model
    try:
        logger.info("Loading Chroma vectorstore from '%s'", persist_dir)
        _VECTORSTORE = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_name=collection_name,
        )
        logger.info("Vectorstore loaded successfully")
        return _VECTORSTORE

    except Exception as e:
        logger.error("Failed to load vectorstore: %s", e, exc_info=True)
        raise
