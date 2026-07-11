"""Unit tests for GitHubClient."""

from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from code_reviewer.integrations.github_client import GitHubClient


REPO_NAME = "owner/awesome-repo"
PR_NUMBER = 42
COMMIT_SHA = "abc123def456abc123def456abc123def456abc1"
SAMPLE_DIFF = """\
diff --git a/src/foo.py b/src/foo.py
--- a/src/foo.py
+++ b/src/foo.py
@@ -1,3 +1,4 @@
 def foo():
+    x = 1
-    y = 2
     return x
"""


@pytest.fixture
def client():
    """Return a GitHubClient with a mocked PyGithub instance."""
    with patch("code_reviewer.integrations.github_client.Github") as MockGithub:
        mock_gh_instance = MagicMock()
        MockGithub.return_value = mock_gh_instance
        c = GitHubClient(token="fake-token")
        c._mock_gh = mock_gh_instance  # Store for test assertions
        yield c


class TestGetPrDiff:
    def test_fetches_diff_with_correct_accept_header(self, client):
        """get_pr_diff must request application/vnd.github.v3.diff."""
        with patch("code_reviewer.integrations.github_client.httpx.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.text = SAMPLE_DIFF
            mock_resp.raise_for_status = MagicMock()
            mock_get.return_value = mock_resp

            result = client.get_pr_diff(REPO_NAME, PR_NUMBER)

            assert result == SAMPLE_DIFF
            call_kwargs = mock_get.call_args
            assert call_kwargs.kwargs["headers"]["Accept"] == "application/vnd.github.v3.diff"
            assert f"/pulls/{PR_NUMBER}" in call_kwargs.args[0]


class TestGetPrFiles:
    def test_returns_list_of_filenames(self, client):
        mock_file = MagicMock()
        mock_file.filename = "src/foo.py"
        mock_pr = MagicMock()
        mock_pr.get_files.return_value = [mock_file]
        client._mock_gh.get_repo.return_value.get_pull.return_value = mock_pr

        result = client.get_pr_files(REPO_NAME, PR_NUMBER)

        assert result == ["src/foo.py"]
        client._mock_gh.get_repo.assert_called_once_with(REPO_NAME)


class TestGetPrHeadSha:
    def test_returns_head_sha(self, client):
        mock_pr = MagicMock()
        mock_pr.head.sha = COMMIT_SHA
        client._mock_gh.get_repo.return_value.get_pull.return_value = mock_pr

        result = client.get_pr_head_sha(REPO_NAME, PR_NUMBER)

        assert result == COMMIT_SHA


class TestPostInlineComment:
    def test_posts_to_reviews_endpoint_with_diff_position(self, client):
        """post_inline_comment must use diff_position, NOT the file line number."""
        with patch("code_reviewer.integrations.github_client.httpx.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.status_code = 201
            mock_post.return_value = mock_resp

            client.post_inline_comment(
                repo_name=REPO_NAME,
                pr_number=PR_NUMBER,
                commit_sha=COMMIT_SHA,
                file_path="src/foo.py",
                diff_position=3,       # diff position, NOT file line 2
                body="**[HIGH]** logic — Unhandled exception",
            )

            mock_post.assert_called_once()
            call_args = mock_post.call_args
            url = call_args.args[0]
            payload = call_args.kwargs["json"]

            assert f"/pulls/{PR_NUMBER}/reviews" in url
            assert payload["comments"][0]["position"] == 3
            assert payload["comments"][0]["path"] == "src/foo.py"
            assert payload["commit_id"] == COMMIT_SHA

    def test_logs_warning_on_non_2xx_response(self, client, caplog):
        with patch("code_reviewer.integrations.github_client.httpx.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.status_code = 422
            mock_resp.text = "Unprocessable Entity"
            mock_post.return_value = mock_resp

            import logging
            with caplog.at_level(logging.WARNING, logger="code_reviewer.integrations.github_client"):
                client.post_inline_comment(
                    repo_name=REPO_NAME,
                    pr_number=PR_NUMBER,
                    commit_sha=COMMIT_SHA,
                    file_path="src/foo.py",
                    diff_position=3,
                    body="test comment",
                )

            assert any("Failed to post inline comment" in r.message for r in caplog.records)


class TestPostPrSummary:
    def test_creates_issue_comment(self, client):
        mock_pr = MagicMock()
        client._mock_gh.get_repo.return_value.get_pull.return_value = mock_pr

        client.post_pr_summary(REPO_NAME, PR_NUMBER, "## Summary\nAll good.")

        mock_pr.create_issue_comment.assert_called_once_with("## Summary\nAll good.")
