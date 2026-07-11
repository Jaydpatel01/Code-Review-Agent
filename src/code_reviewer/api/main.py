"""FastAPI application entry point."""

import logging

from fastapi import FastAPI

from code_reviewer.api.webhook import router as webhook_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = FastAPI(
    title="AI Code Reviewer",
    description="GitHub PR bot that reviews pull requests using static analysis and LLM.",
    version="0.1.0",
)

app.include_router(webhook_router)


@app.get("/health")
async def health() -> dict[str, str]:
    """Health-check endpoint."""
    return {"status": "ok"}
