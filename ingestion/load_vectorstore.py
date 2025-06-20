import logging
import os
from typing import Dict

from langchain_community.vectorstores import FAISS, Chroma

from langgraph_flow.models.openai_model import OpenAIModel
from utils.constants import (
    KEY_CHROMA,
    KEY_FAISS,
    KEY_PERSIST_DIRECTORY,
    KEY_TYPE,
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
    store_type = store_cfg.get(KEY_TYPE, KEY_CHROMA).lower()
    persist_dir = store_cfg.get(
        KEY_PERSIST_DIRECTORY, "vectorstore/chroma_index/"
    )
    os.makedirs(persist_dir, exist_ok=True)

    embeddings = OpenAIModel(cfg).embedding_model

    try:
        if store_type == KEY_CHROMA:
            logger.info("Loading Chroma vectorstore from '%s'", persist_dir)
            _VECTORSTORE = Chroma(
                persist_directory=persist_dir, embedding_function=embeddings
            )
        elif store_type == KEY_FAISS:
            index_path = os.path.join(persist_dir, "faiss_index")
            logger.info("Loading FAISS vectorstore from '%s'", index_path)
            _VECTORSTORE = FAISS.load_local(
                folder_path=index_path, embedding=embeddings
            )
        else:
            msg = f"Unsupported vectorstore type: '{store_type}'"
            logger.error(msg)
            raise ValueError(msg)

        logger.info("Vectorstore loaded successfully")
        return _VECTORSTORE

    except Exception as e:
        logger.error("Failed to load vectorstore: %s", e, exc_info=True)
        raise
