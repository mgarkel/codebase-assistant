import os
import logging
from typing import Dict

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma, FAISS

from utils.constants import (
    ENV_OPENAIAPI_KEY,
    KEY_CHROMA,
    KEY_EMBEDDING_MODEL,
    KEY_FAISS,
    KEY_OPENAI,
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

    model_name = cfg[KEY_OPENAI][KEY_EMBEDDING_MODEL]
    openai_api_key = os.getenv(ENV_OPENAIAPI_KEY)
    logger.info("Using embedding model: %s", model_name)
    embeddings = OpenAIEmbeddings(
        model=model_name, openai_api_key=openai_api_key
    )

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
