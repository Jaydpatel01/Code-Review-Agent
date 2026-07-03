"""Pydantic data models for the code reviewer."""

from datetime import datetime, timezone
from typing import Literal, Optional, List, Tuple
from pydantic import BaseModel, Field


class Finding(BaseModel):
    """Represents a single issue found during code review."""

    file_path: str
    line_number: Optional[int] = None
    severity: Literal["HIGH", "MEDIUM", "LOW", "INFO"]
    category: Literal["security", "performance", "style", "logic", "docs", "complexity"]
    message: str
    suggestion: str
    source: Literal["llm", "static", "ast"]


class ReviewResult(BaseModel):
    """Contains the overall results of a reviewed file."""

    file_path: str
    findings: List[Finding]
    summary: str
    reviewed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    model_used: str
    lines_reviewed: int


class DiffHunk(BaseModel):
    """Represents a structured diff hunk for localized reviews."""

    file_path: str
    start_line: int
    end_line: int
    added_lines: List[Tuple[int, str]]      # (line_number, content)
    removed_lines: List[Tuple[int, str]]    # (line_number, content)
    context_lines: List[Tuple[int, str]]    # (line_number, content)
    raw_hunk: str


class LLMFinding(BaseModel):
    """Internal model for LLM findings generation."""

    line_number: Optional[int] = None
    severity: Literal["HIGH", "MEDIUM", "LOW", "INFO"]
    category: Literal["security", "performance", "style", "logic", "docs", "complexity"]
    message: str
    suggestion: str


class LLMReviewResponse(BaseModel):
    """Internal model for LLM review response packaging."""

    findings: List[LLMFinding]
    summary: str

