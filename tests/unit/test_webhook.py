"""Unit tests for the FastAPI webhook endpoint."""

import hashlib
import hmac
import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

FIXTURE_PATH = Path(__file__).parent.parent / "fixtures" / "pr_opened_payload.json"
TEST_SECRET = "super_secret_test_key"


def _make_client(secret: str = TEST_SECRET) -> TestClient:
    """Return a TestClient with WEBHOOK_SECRET pre-set in the environment."""
    os.environ["WEBHOOK_SECRET"] = secret
    # Re-import app after env is patched so the module sees the right secret.
    from code_reviewer.api.main import app
    return TestClient(app, raise_server_exceptions=False)


def _sign(body: bytes, secret: str = TEST_SECRET) -> str:
    """Return the X-Hub-Signature-256 header value for the given body."""
    digest = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


# ---------------------------------------------------------------------------
# Signature verification tests
# ---------------------------------------------------------------------------

class TestSignatureVerification:
    def test_missing_signature_returns_401(self):
        client = _make_client()
        body = b'{"zen": "Keep it logically awesome."}'
        response = client.post(
            "/webhook",
            content=body,
            headers={"X-GitHub-Event": "ping", "Content-Type": "application/json"},
        )
        assert response.status_code == 401
        assert "X-Hub-Signature-256" in response.json()["detail"]

    def test_malformed_signature_returns_401(self):
        client = _make_client()
        body = b'{"zen": "Keep it logically awesome."}'
        response = client.post(
            "/webhook",
            content=body,
            headers={
                "X-GitHub-Event": "ping",
                "X-Hub-Signature-256": "not-sha256-format",
                "Content-Type": "application/json",
            },
        )
        assert response.status_code == 401

    def test_wrong_signature_returns_401(self):
        client = _make_client()
        body = b'{"zen": "Keep it logically awesome."}'
        response = client.post(
            "/webhook",
            content=body,
            headers={
                "X-GitHub-Event": "ping",
                "X-Hub-Signature-256": "sha256=" + "a" * 64,
                "Content-Type": "application/json",
            },
        )
        assert response.status_code == 401
        assert "Signature verification failed" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Event dispatch tests
# ---------------------------------------------------------------------------

class TestEventDispatch:
    def test_ping_event_returns_200_pong(self):
        client = _make_client()
        body = json.dumps({"zen": "Keep it logically awesome."}).encode()
        response = client.post(
            "/webhook",
            content=body,
            headers={
                "X-GitHub-Event": "ping",
                "X-Hub-Signature-256": _sign(body),
                "Content-Type": "application/json",
            },
        )
        assert response.status_code == 200
        assert response.json() == {"status": "pong"}

    def test_unknown_event_returns_200_ignored(self):
        client = _make_client()
        body = json.dumps({"action": "labeled"}).encode()
        response = client.post(
            "/webhook",
            content=body,
            headers={
                "X-GitHub-Event": "issues",
                "X-Hub-Signature-256": _sign(body),
                "Content-Type": "application/json",
            },
        )
        assert response.status_code == 200
        assert response.json()["status"] == "ignored"

    def test_pr_opened_triggers_review_and_returns_200(self, mocker):
        """PR opened event should schedule a background review and return 200 immediately."""
        # Mock the background task function so it doesn't try to contact GitHub/LLM
        mock_run = mocker.patch(
            "code_reviewer.api.webhook._run_pr_review",
            return_value=None,
        )

        client = _make_client()
        payload = json.loads(FIXTURE_PATH.read_text())
        body = json.dumps(payload).encode()

        response = client.post(
            "/webhook",
            content=body,
            headers={
                "X-GitHub-Event": "pull_request",
                "X-Hub-Signature-256": _sign(body),
                "Content-Type": "application/json",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "review_scheduled"
        assert data["pr"] == 42

    def test_pr_closed_action_is_ignored(self):
        """PR closed action should be acknowledged but not trigger a review."""
        client = _make_client()
        payload = {"action": "closed", "pull_request": {"number": 99}, "repository": {"full_name": "owner/repo"}}
        body = json.dumps(payload).encode()

        response = client.post(
            "/webhook",
            content=body,
            headers={
                "X-GitHub-Event": "pull_request",
                "X-Hub-Signature-256": _sign(body),
                "Content-Type": "application/json",
            },
        )
        assert response.status_code == 200
        assert response.json()["status"] == "ignored"
