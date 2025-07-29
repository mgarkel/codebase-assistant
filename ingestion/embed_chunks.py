import hashlib
import logging
from pathlib import Path
from typing import Dict, List

from langchain_chroma import Chroma

from langgraph_flow.models.openai_model import OpenAIModel
from utils.constants import (
    COLLECTION_NAME,
    KEY_CONTENT,
    KEY_META,
    KEY_PERSIST_DIRECTORY,
    KEY_VECTORSTORE,
)

logger = logging.getLogger(__name__)


def _filter_new_ids_only(texts, metadatas, ids, existing_ids):
    to_add_texts = []
    to_add_meta = []
    to_add_ids = []
    for text, meta, id_ in zip(texts, metadatas, ids):
        if id_ not in existing_ids:
            to_add_texts.append(text)
            to_add_meta.append(meta)
            to_add_ids.append(id_)
    return to_add_texts, to_add_meta, to_add_ids


def _delete_stale_ids(store, ids):
    existing = store.get()
    existing_ids = set(existing["ids"])
    new_ids_set = set(ids)
    stale_ids = existing_ids - new_ids_set
    if stale_ids:
        logger.info("Deleting %d stale chunks", len(stale_ids))
        store.delete(ids=list(stale_ids))
    return existing_ids


def embed_documents(
    docs: List[Dict],
    cfg: Dict,
    *,
    reset_index: bool = False,
    batch_size: int = 256,
) -> None:
    """
    Embed and persist documents into a Chroma collection, with:
      - stale‐ID deletion
      - upsert of only new IDs
      - stable chunk IDs via content hashing

    Args:
        docs: List of {"content": str, "meta": dict} items.
        cfg:  Your settings.toml dict.
        reset_index: If True, drop and rebuild the index from scratch.
        batch_size: Chunk count per embedding/API call.
    """
    if not docs:
        logger.warning("No documents to embed; skipping.")
        return

    # Unpack config
    store_cfg = cfg[KEY_VECTORSTORE]
    persist_dir = Path(store_cfg[KEY_PERSIST_DIRECTORY])
    persist_dir.mkdir(parents=True, exist_ok=True)

    # Prepare data
    embeddings = OpenAIModel(cfg).embedding_model
    texts = [d[KEY_CONTENT] for d in docs]
    metadatas = [d[KEY_META] for d in docs]

    # stable ID = SHA256 of the chunk text
    ids = [hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts]

    logger.info("Embedding %d chunks into Chroma", len(texts))

    # Full rebuild
    if reset_index:
        logger.info("Rebuilding Chroma index from scratch")
        store = Chroma.from_texts(
            texts=texts,
            embedding=embeddings,
            metadatas=metadatas,
            ids=ids,
            persist_directory=str(persist_dir),
            collection_name=COLLECTION_NAME,
        )

    # Incremental upsert
    else:
        logger.info("Loading existing Chroma index (or creating new)")
        store = Chroma(
            persist_directory=str(persist_dir),
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME,
        )

        # Delete Stale ID's
        existing_ids = _delete_stale_ids(store, ids)

        # Filter for only new ID's
        to_add_texts, to_add_meta, to_add_ids = _filter_new_ids_only(
            texts, metadatas, ids, existing_ids
        )

        if not to_add_ids:
            logger.info("No new chunks to add; skipping upsert.")
            return

            # Batch‑upsert only new chunks
        else:
            for i in range(0, len(to_add_texts), batch_size):
                batch_texts = to_add_texts[i : i + batch_size]
                batch_meta = to_add_meta[i : i + batch_size]
                batch_ids = to_add_ids[i : i + batch_size]
                store.add_texts(
                    texts=batch_texts,
                    metadatas=batch_meta,
                    ids=batch_ids,
                )
                logger.info(
                    "Upserted new chunks %d–%d", i, i + len(batch_texts)
                )

    logger.info("Chroma index updated successfully at %s", persist_dir)
