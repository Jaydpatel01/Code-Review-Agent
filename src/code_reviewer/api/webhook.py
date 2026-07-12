"""Webhook endpoint for GitHub PR events."""

import hashlib
import hmac
import logging
import os
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Request

logger = logging.getLogger(__name__)

router = APIRouter()

HANDLED_PR_ACTIONS = {"opened", "synchronize", "reopened"}


def _verify_signature(payload_bytes: bytes, signature_header: str | None) -> None:
    """
    Verify the HMAC-SHA256 signature from GitHub.

    GitHub signs every webhook payload with HMAC-SHA256 using the webhook
    secret configured on the repository.  The signature is sent in the
    ``X-Hub-Signature-256`` header as ``sha256=<hex-digest>``.

    Raises:
        HTTPException: 401 if the header is missing or the signature does not
            match.
    """
    secret = os.environ.get("WEBHOOK_SECRET", "")
    if not secret:
        # If no secret is configured we cannot verify — reject all requests.
        raise HTTPException(
            status_code=401,
            detail="WEBHOOK_SECRET is not configured on the server.",
        )

    if not signature_header or not signature_header.startswith("sha256="):
        raise HTTPException(
            status_code=401,
            detail="Missing or malformed X-Hub-Signature-256 header.",
        )

    expected = hmac.new(
        secret.encode(), payload_bytes, hashlib.sha256
    ).hexdigest()
    received = signature_header[len("sha256="):]

    if not hmac.compare_digest(expected, received):
        raise HTTPException(
            status_code=401,
            detail="Signature verification failed.",
        )


async def _run_pr_review(repo_name: str, pr_number: int) -> None:
    """
    Background task: run the full review pipeline on a pull request.

    Intentionally catches all exceptions so a failed review never crashes
    the background worker.
    """
    try:
        # Lazy import to avoid circular dependencies and to keep the import
        # cost out of the hot path for every webhook request.
        from code_reviewer.config import load_settings
        from code_reviewer.core.llm_client import LLMClient
        from code_reviewer.integrations.github_client import GitHubClient
        from code_reviewer.core.pr_reviewer import PRReviewer

        github_token = os.environ.get("GITHUB_TOKEN", "")
        settings = load_settings()
        llm_client = LLMClient(model=settings.model, max_tokens=settings.max_tokens)
        github_client = GitHubClient(github_token)

        reviewer = PRReviewer(
            github_client=github_client,
            llm_client=llm_client,
            settings=settings,
        )
        reviewer.review_pr(repo_name, pr_number)
        logger.info("Review completed for %s#%d", repo_name, pr_number)
    except Exception:
        logger.exception(
            "Review pipeline failed for %s#%d", repo_name, pr_number
        )


@router.post("/webhook")
async def webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_hub_signature_256: str | None = Header(default=None),
    x_github_event: str | None = Header(default=None),
) -> dict[str, Any]:
    """
    Receive and dispatch GitHub webhook events.

    Security: every request is verified against X-Hub-Signature-256 before
    any processing occurs.  Invalid or missing signatures are rejected with
    HTTP 401.

    Supported events:
    - ``ping``         → acknowledge the handshake and return 200.
    - ``pull_request`` (opened / synchronize / reopened) → schedule review.
    All other events are acknowledged and ignored.
    """
    payload_bytes = await request.body()

    # --- Security gate: verify signature FIRST, before reading anything ---
    _verify_signature(payload_bytes, x_hub_signature_256)

    payload: dict[str, Any] = await request.json()

    # --- Ping: GitHub sends this when a webhook is first configured ---
    if x_github_event == "ping":
        logger.info("Received GitHub ping: %s", payload.get("zen", ""))
        return {"status": "pong"}

    # --- Pull request events ---
    if x_github_event == "pull_request":
        action = payload.get("action", "")
        if action in HANDLED_PR_ACTIONS:
            repo_name: str = payload["repository"]["full_name"]
            pr_number: int = payload["pull_request"]["number"]
            logger.info(
                "Scheduling review for %s#%d (action=%s)",
                repo_name,
                pr_number,
                action,
            )
            # Return 200 immediately; review runs asynchronously.
            background_tasks.add_task(_run_pr_review, repo_name, pr_number)
            return {"status": "review_scheduled", "pr": pr_number}

        return {"status": "ignored", "reason": f"unhandled action '{action}'"}

    # --- All other event types ---
    return {"status": "ignored", "event": x_github_event}
