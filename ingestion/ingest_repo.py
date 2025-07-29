import logging
from pathlib import Path

from git import GitCommandError, InvalidGitRepositoryError, Repo

logger = logging.getLogger(__name__)


def clone_or_update_repo(repo_url: str, dest: str = "repo/") -> Repo:
    """
    Clone a Git repository to `dest` if it doesn't exist,
    otherwise fetch & pull updates on the existing repo.

    Args:
        repo_url: URL of the remote Git repository.
        dest:    Local directory path.

    Returns:
        A GitPython Repo object pointing at `dest`.

    Raises:
        RuntimeError: If `dest` exists but isn’t a Git repo,
                      or if any Git command fails,
                      or on any unexpected I/O error.
    """
    dest_path = Path(dest)

    try:
        if dest_path.exists():
            # Update existing repo
            repo = Repo(dest_path)
            logger.info(f"Updating existing repository at '{dest}'")
            origin = repo.remotes.origin
            origin.fetch()
            origin.pull()
            logger.debug(f"Repository at '{dest}' updated successfully")
        else:
            # Clone new repo
            logger.info(f"Cloning repository {repo_url!r} into '{dest}'")
            repo = Repo.clone_from(repo_url, dest)
            logger.info(f"Repository cloned to '{dest}' successfully")

        return repo

    except InvalidGitRepositoryError:
        msg = f"Destination '{dest}' exists but is not a Git repository."
        logger.error(msg)
        raise RuntimeError(msg)

    except GitCommandError as e:
        msg = f"Git command failed: {e}"
        logger.error(msg, exc_info=True)
        raise RuntimeError(msg)

    except Exception as e:
        msg = f"Unexpected error during clone/update: {e}"
        logger.error(msg, exc_info=True)
        raise RuntimeError(msg)
