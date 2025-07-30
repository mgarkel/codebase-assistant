import logging
from pathlib import Path
from urllib.parse import urlparse

from git import GitCommandError, InvalidGitRepositoryError, Repo

logger = logging.getLogger(__name__)


def clone_or_update_repo(repo_url: str, dest: str = "repo/") -> str:
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
            Repo.clone_from(repo_url, dest)
            logger.info(f"Repository cloned to '{dest}' successfully")
        repo_name = get_project_name_from_url(repo_url)
        return repo_name

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


def get_project_name_from_url(repo_url: str) -> str:
    """
    Given a GitHub repo URL, returns the repository name.

    Examples:
        >>> get_project_name_from_url("https://github.com/user/my-repo.git")
        "my-repo"
        >>> get_project_name_from_url("git@github.com:user/my-repo.git")
        "my-repo"
        >>> get_project_name_from_url("https://github.com/user/my-repo")
        "my-repo"
    """
    # Handle both HTTPS and SSH-style URLs
    if repo_url.startswith("git@"):
        # e.g. git@github.com:user/my-repo.git
        path = repo_url.split(":", 1)[1]
    else:
        parsed = urlparse(repo_url)
        path = parsed.path  # e.g. '/user/my-repo.git'

    # Strip leading slash if present
    if path.startswith("/"):
        path = path[1:]

    # The repo is the last component
    name = path.rstrip("/").split("/")[-1]

    # Remove a trailing .git if it exists
    if name.endswith(".git"):
        name = name[:-4]

    return name
