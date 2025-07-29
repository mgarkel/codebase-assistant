import ast
import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Set

from git import InvalidGitRepositoryError, Repo
from langchain.text_splitter import TokenTextSplitter

logger = logging.getLogger(__name__)

DEFAULT_EXTENSIONS = {".py", ".js", ".java", ".ts", ".md"}
DEFAULT_IGNORED_DIRS = {
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "env",
    "build",
}
# Adjust these to your embedding model’s max tokens (e.g. 800, 1000)
DEFAULT_CHUNK_TOKENS = 500
DEFAULT_CHUNK_OVERLAP = 50
# Encoding for tiktoken (OpenAI embeddings)
DEFAULT_ENCODING = "cl100k_base"


def _extract_python_blocks(text: str) -> List[str]:
    """Return a list of top-level function/class source segments, or the whole text if none."""
    try:
        tree = ast.parse(text)
        blocks = [
            ast.get_source_segment(text, node)
            for node in tree.body
            if isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            )
            and ast.get_source_segment(text, node)
        ]
        return blocks or [text]
    except Exception:
        return [text]


def _chunk_file(
    path: Path,
    repo_root: Path,
    splitter: TokenTextSplitter,
    seen_hashes: Set[str],
    repo_url: Optional[str],
    commit_hash: Optional[str],
) -> List[Dict]:
    """Read a file, split into semantically‑aware text segments, then tokens‑split & dedupe."""
    lang = path.suffix.lstrip(".")
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        logger.error("Failed to read %s: %s", path, e)
        return []

    if not text.strip():
        logger.debug("Skipping empty file %s", path)
        return []

    # Language‑aware pre‑splitting
    segments = _extract_python_blocks(text) if lang == "py" else [text]

    docs: List[Dict] = []
    for seg in segments:
        # Token‑aware splitting with overlap
        chunks = splitter.split_text(seg)
        for idx, chunk in enumerate(chunks):
            # Deduplication by SHA‑256
            h = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

            docs.append(
                {
                    "content": chunk,
                    "meta": {
                        "relative_path": str(path.relative_to(repo_root)),
                        "chunk_index": idx,
                        "language": lang,
                        **({"repo_url": repo_url} if repo_url else {}),
                        **({"commit_hash": commit_hash} if commit_hash else {}),
                    },
                }
            )

    logger.debug("Chunked %s into %d pieces", path, len(docs))
    return docs


def chunk_repository(
    repo_path: str,
    *,
    extensions: Optional[List[str]] = None,
    ignored_dirs: Optional[List[str]] = None,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    max_workers: int = 4,
) -> List[Dict]:
    """
    Walk a Git repo, split code/docs into token‑aware chunks, and return docs for vector DB.

    Args:
        repo_path:      Path to local repo.
        extensions:     File extensions to include (default common code + markdown).
        ignored_dirs:   Directory names to skip (default .git, node_modules, etc.).
        chunk_tokens:   Approximate max tokens per chunk.
        chunk_overlap:  Token overlap between chunks.
        max_workers:    Threads for parallel file processing.

    Returns:
        List of {"content": str, "meta": {...}} ready for ingestion.

    Raises:
        RuntimeError if `repo_path` isn’t a directory.
    """
    root = Path(repo_path)
    if not root.is_dir():
        raise RuntimeError(f"Invalid repository path: {repo_path}")

    extensions = set(extensions or DEFAULT_EXTENSIONS)
    ignored_dirs = set(ignored_dirs or DEFAULT_IGNORED_DIRS)

    # Attempt to read Git metadata
    repo_url = commit_hash = None
    try:
        repo = Repo(root)
        commit_hash = repo.head.commit.hexsha
        # pick the first origin URL
        repo_url = next(repo.remotes.origin.urls, None)
    except InvalidGitRepositoryError:
        logger.warning(
            "%s is not a Git repo; skipping repo metadata", repo_path
        )

    # Prepare a token‑based splitter (uses tiktoken under the hood)
    splitter = TokenTextSplitter(
        encoding_name=DEFAULT_ENCODING,
        chunk_size=chunk_tokens,
        chunk_overlap=chunk_overlap,
    )

    # Gather all files to process
    all_files = [
        p
        for p in root.rglob("*")
        if p.is_file()
        and p.suffix in extensions
        and not any(part in ignored_dirs for part in p.parts)
    ]
    logger.info("Found %d files to chunk in %s", len(all_files), repo_path)

    docs: List[Dict] = []
    seen_hashes: Set[str] = set()

    # Parallelize file chunking
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                _chunk_file,
                path,
                root,
                splitter,
                seen_hashes,
                repo_url,
                commit_hash,
            ): path
            for path in all_files
        }

        for future in as_completed(futures):
            file_path = futures[future]
            try:
                docs.extend(future.result())
            except Exception as e:
                logger.error(
                    "Error chunking %s: %s", file_path, e, exc_info=True
                )

    logger.info("Generated %d chunks from %d files", len(docs), len(all_files))
    return docs
