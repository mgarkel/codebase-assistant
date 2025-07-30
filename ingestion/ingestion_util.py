"""utils for ingestion flow."""

from pathlib import Path
from typing import Dict, Tuple

from utils.constants import (
    KEY_BASE_DIRECTORY,
    KEY_COLLECTION,
    KEY_PROJECT_NAME,
    KEY_REPO,
    KEY_SUBPATH,
    KEY_VECTORSTORE,
)


def get_persist_dir_and_collection_name_from_config(cfg: Dict) -> Tuple:
    repo_name = cfg[KEY_REPO][KEY_PROJECT_NAME]
    store_cfg = cfg[KEY_VECTORSTORE]
    collection_name = f"{cfg[KEY_VECTORSTORE][KEY_COLLECTION]}_{repo_name}"
    persist_dir = (
        Path(store_cfg[KEY_BASE_DIRECTORY]) / repo_name / store_cfg[KEY_SUBPATH]
    )
    return persist_dir, collection_name
