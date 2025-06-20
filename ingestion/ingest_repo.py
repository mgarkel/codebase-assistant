import os
import logging
from git import Repo, GitCommandError, InvalidGitRepositoryError

logger = logging.getLogger(__name__)


def clone_repo(repo_url: str, dest: str = "repo/") -> Repo:
    """
    Clone a Git repository, or update it if it already exists.

    Args:
        repo_url: URL of the remote Git repository.
        dest: Local directory path to clone into (or update).

    Returns:
        A GitPython Repo object pointing to the local repository.

    Raises:
        RuntimeError: If cloning or pulling fails.
    """
    try:
        if not os.path.exists(dest):
            logger.info(f"Cloning repository {repo_url} into '{dest}'")
            repo = Repo.clone_from(repo_url, dest)
            logger.debug(f"Successfully cloned to '{dest}'")
        else:
            # If dest exists, verify it's a Git repo
            try:
                repo = Repo(dest)
                logger.info(
                    f"Repository already exists at '{dest}', fetching updates"
                )
                origin = repo.remotes.origin
                origin.fetch()
                origin.pull()
                logger.debug(f"Successfully updated repository at '{dest}'")
            except InvalidGitRepositoryError:
                msg = (
                    f"Destination '{dest}' exists but is not a Git repository."
                )
                logger.error(msg)
                raise RuntimeError(msg)
        return repo

    except GitCommandError as e:
        msg = f"Git command failed: {e}"
        logger.error(msg, exc_info=True)
        raise RuntimeError(msg)

    except Exception as e:
        msg = f"Unexpected error during clone/update: {e}"
        logger.error(msg, exc_info=True)
        raise RuntimeError(msg)
