"""GitHub API client using PyGithub."""

import logging
from typing import Optional

import httpx
from github import Github, GithubException
from github.PullRequest import PullRequest
from github.Repository import Repository

logger = logging.getLogger(__name__)


class GitHubClient:
    """
    Thin wrapper around PyGithub for the operations needed by PRReviewer.

    All methods that post to GitHub are safe to call even when the PR has
    already been merged — PyGithub raises GithubException and the caller
    is expected to handle it.
    """

    def __init__(self, token: str):
        """
        Initialise the client.

        Args:
            token: A GitHub personal access token or installation token.
                   Must have ``pull_requests: write`` and ``contents: read``
                   permissions.
        """
        if not token:
            raise ValueError("GITHUB_TOKEN must not be empty.")
        self._gh = Github(token)
        self._token = token

    def _get_repo(self, repo_name: str) -> Repository:
        """Return a PyGithub Repository object for the given full name."""
        return self._gh.get_repo(repo_name)

    def _get_pr(self, repo_name: str, pr_number: int) -> PullRequest:
        """Return a PyGithub PullRequest object."""
        return self._get_repo(repo_name).get_pull(pr_number)

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get_pr_diff(self, repo_name: str, pr_number: int) -> str:
        """
        Fetch the raw unified diff for a pull request.

        Uses the GitHub REST API directly (via httpx) because PyGithub does
        not expose the ``application/vnd.github.v3.diff`` content type.

        Args:
            repo_name:  Full repository name, e.g. ``"owner/repo"``.
            pr_number:  Pull-request number.

        Returns:
            Raw unified diff string.

        Raises:
            httpx.HTTPStatusError: If the API request fails.
        """
        url = f"https://api.github.com/repos/{repo_name}/pulls/{pr_number}"
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Accept": "application/vnd.github.v3.diff",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        response = httpx.get(url, headers=headers, follow_redirects=True)
        response.raise_for_status()
        return response.text

    def get_pr_files(self, repo_name: str, pr_number: int) -> list[str]:
        """
        Return the list of file paths changed in a pull request.

        Args:
            repo_name:  Full repository name.
            pr_number:  Pull-request number.

        Returns:
            List of file paths relative to the repo root.
        """
        pr = self._get_pr(repo_name, pr_number)
        return [f.filename for f in pr.get_files()]

    def get_pr_head_sha(self, repo_name: str, pr_number: int) -> str:
        """
        Return the SHA of the HEAD commit on the pull-request branch.

        This is required when creating a pull-request review via
        ``POST /repos/{owner}/{repo}/pulls/{pull_number}/reviews``.

        Args:
            repo_name:  Full repository name.
            pr_number:  Pull-request number.

        Returns:
            40-character hex SHA string.
        """
        pr = self._get_pr(repo_name, pr_number)
        return pr.head.sha

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def post_inline_comment(
        self,
        repo_name: str,
        pr_number: int,
        commit_sha: str,
        file_path: str,
        diff_position: int,
        body: str,
    ) -> None:
        """
        Post an inline review comment on a specific line of the PR diff.

        Uses the ``pull_request_review`` API (not ``issues/comments``) so the
        comment appears anchored to the diff line in the GitHub UI.

        The ``diff_position`` is the 1-indexed offset within the *file's diff
        block*: position 1 is the ``@@ ... @@`` hunk header, and every
        subsequent line (context, added, or removed) increments the counter.
        Only ``+`` lines can receive inline comments.

        Args:
            repo_name:     Full repository name.
            pr_number:     Pull-request number.
            commit_sha:    SHA of the latest commit on the PR branch.
            file_path:     Path of the file being commented on.
            diff_position: Position within the diff (NOT the file line number).
            body:          Markdown comment body.
        """
        url = (
            f"https://api.github.com/repos/{repo_name}/pulls/{pr_number}/reviews"
        )
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        payload = {
            "commit_id": commit_sha,
            "event": "COMMENT",
            "comments": [
                {
                    "path": file_path,
                    "position": diff_position,
                    "body": body,
                }
            ],
        }
        response = httpx.post(url, json=payload, headers=headers)
        if response.status_code not in (200, 201):
            logger.warning(
                "Failed to post inline comment on %s#%d %s@%d: %s %s",
                repo_name,
                pr_number,
                file_path,
                diff_position,
                response.status_code,
                response.text,
            )
        else:
            logger.debug(
                "Posted inline comment on %s#%d %s@pos=%d",
                repo_name,
                pr_number,
                file_path,
                diff_position,
            )

    def post_pr_summary(
        self,
        repo_name: str,
        pr_number: int,
        body: str,
    ) -> None:
        """
        Post a general (non-inline) comment on the pull request.

        This is used for the overall review summary block at the end of the
        review run.  It uses the ``issues/comments`` endpoint so it appears
        as a regular PR conversation comment.

        Args:
            repo_name:  Full repository name.
            pr_number:  Pull-request number.
            body:       Markdown comment body.
        """
        try:
            pr = self._get_pr(repo_name, pr_number)
            pr.create_issue_comment(body)
            logger.info(
                "Posted PR summary on %s#%d (%d chars)",
                repo_name,
                pr_number,
                len(body),
            )
        except GithubException as exc:
            logger.error(
                "Failed to post summary on %s#%d: %s",
                repo_name,
                pr_number,
                exc,
            )
