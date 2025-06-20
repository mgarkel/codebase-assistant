import os
import logging
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


def chunk_repository(
    repo_path: str,
    extensions: List[str] = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Dict]:
    """
    Walk through the repository directory and split eligible files into text chunks.

    Args:
        repo_path: Path to the local repository.
        extensions: List of file extensions to include (e.g., ['.py', '.js']).
                    Defaults to common code and markdown files.
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of overlapping characters between chunks.

    Returns:
        A list of documents, each represented as a dict:
        {
            "content": <chunk_text>,
            "meta": {
                "source": <file_path>,
                "chunk": <chunk_index>
            }
        }

    Raises:
        RuntimeError: If the repository path is invalid.
    """
    if not os.path.isdir(repo_path):
        msg = f"Invalid repository path: {repo_path}"
        logger.error(msg)
        raise RuntimeError(msg)

    if extensions is None:
        extensions = [".py", ".js", ".java", ".ts", ".md"]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    documents: List[Dict] = []
    total_files = 0
    total_chunks = 0

    for root, _, files in os.walk(repo_path):
        for fname in files:
            if not any(fname.endswith(ext) for ext in extensions):
                continue
            total_files += 1
            file_path = os.path.join(root, fname)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                if not text.strip():
                    logger.warning("Skipping empty file: %s", file_path)
                    continue

                chunks = splitter.split_text(text)
                for idx, chunk in enumerate(chunks):
                    documents.append(
                        {
                            "content": chunk,
                            "meta": {"source": file_path, "chunk": idx},
                        }
                    )

                total_chunks += len(chunks)
                logger.debug(
                    "Chunked '%s' into %d pieces", file_path, len(chunks)
                )

            except Exception as e:
                logger.error(
                    "Error processing file '%s': %s",
                    file_path,
                    e,
                    exc_info=True,
                )

    if total_files == 0:
        logger.warning(
            "No files found in '%s' matching extensions %s",
            repo_path,
            extensions,
        )

    logger.info(
        "Completed chunking: %d files processed into %d chunks",
        total_files,
        total_chunks,
    )

    return documents
