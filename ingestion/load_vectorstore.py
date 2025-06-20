import os
import logging
from typing import Dict

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma, FAISS

logger = logging.getLogger(__name__)

# Cache the loaded vectorstore so we don’t re-open it on every call
_VECTORSTORE = None


def load_vectorstore(cfg: Dict):
    global _VECTORSTORE
    if _VECTORSTORE:
        return _VECTORSTORE

    store_cfg = cfg.get("vectorstore", {})
    store_type = store_cfg.get("type", "chroma").lower()
    persist_dir = store_cfg.get(
        "persist_directory", "vectorstore/chroma_index/"
    )
    os.makedirs(persist_dir, exist_ok=True)

    model_name = cfg["openai"]["embedding_model"]
    openai_api_key = os.getenv("OPENAPI_KEY")
    logger.info("Using embedding model: %s", model_name)
    embeddings = OpenAIEmbeddings(
        model=model_name, openai_api_key=openai_api_key
    )

    try:
        if store_type == "chroma":
            logger.info("Loading Chroma vectorstore from '%s'", persist_dir)
            _VECTORSTORE = Chroma(
                persist_directory=persist_dir, embedding_function=embeddings
            )
        elif store_type == "faiss":
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
