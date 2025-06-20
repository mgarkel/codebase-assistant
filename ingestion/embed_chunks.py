import os
import logging
import pickle
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma, FAISS

logger = logging.getLogger(__name__)

def embed_documents(docs: list, cfg: dict):
    """
    Embed documents using OpenAI embeddings and store into the configured vector store.

    Args:
        docs: List of dicts with 'content' (str) and 'meta' (dict) keys.
        cfg: Configuration dict from settings.toml with sections:
             - openai.model
             - openai.api_key (optional)
             - vectorstore.type ('chroma' or 'faiss')
             - vectorstore.persist_directory
    Raises:
        RuntimeError: On any failure during embedding or persistence.
    """
    try:
        model_name = cfg["openai"]["embedding_model"]
        openai_api_key = os.getenv("OPENAPI_KEY")
        logger.info("Initializing OpenAI embeddings with model '%s'", model_name)
        embeddings = OpenAIEmbeddings(model=model_name, openai_api_key = openai_api_key)

        store_type = cfg["vectorstore"]["type"].lower()
        persist_dir = cfg["vectorstore"]["persist_directory"]
        os.makedirs(persist_dir, exist_ok=True)

        texts = [doc["content"] for doc in docs]
        metadatas = [doc["meta"] for doc in docs]
        logger.info("Embedding %d documents", len(texts))

        if store_type == "chroma":
            logger.info("Creating Chroma vectorstore at '%s'", persist_dir)
            store = Chroma.from_texts(
                    texts=texts,
                    embedding=embeddings,
                    metadatas=metadatas,
                    persist_directory=persist_dir
                )
            store.persist()
            logger.debug("Chroma vectorstore persisted")

        elif store_type == "faiss":
            logger.info("Creating FAISS vectorstore in '%s'", persist_dir)
            store = FAISS.from_texts(
                texts=texts,
                embedding=embeddings,
                metadatas=metadatas
            )
            # Persist FAISS index and metadata
            index_path = os.path.join(persist_dir, "faiss_index")
            os.makedirs(index_path, exist_ok=True)
            store.save_local(index_path)
            logger.debug("FAISS vectorstore saved to '%s'", index_path)

        else:
            msg = f"Unsupported vectorstore type: '{store_type}'"
            logger.error(msg)
            raise ValueError(msg)

        logger.info("Embedding pipeline completed successfully")
    except Exception as e:
        logger.error("Failed to embed documents: %s", e, exc_info=True)
        raise RuntimeError(f"Embedding error: {e}") from e
