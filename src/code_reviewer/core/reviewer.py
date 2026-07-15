"""Module for reviewing files using LLM clients and static analysis."""

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import List, Set, Tuple

from code_reviewer.core.llm_client import LLMClient
from code_reviewer.core.models import Finding, ReviewResult, LLMReviewResponse, DiffHunk
from code_reviewer.core.prompts import (
    SYSTEM_REVIEW_PROMPT,
    USER_REVIEW_PROMPT_TEMPLATE,
)
from code_reviewer.config import Settings
from code_reviewer.analyzers.static_checks import StaticAnalyzer
from code_reviewer.agents.graph import run_agent_review

logger = logging.getLogger(__name__)

SEVERITY_ORDER = {
    "INFO": 1,
    "LOW": 2,
    "MEDIUM": 3,
    "HIGH": 4
}


def combine_findings(static: List[Finding], llm: List[Finding]) -> List[Finding]:
    """
    Combines static and LLM findings, deduplicating them.
    If both flag the same line for the same category, keep the static finding
    and discard the LLM finding.
    """
    combined: List[Finding] = []

    # Track static findings by (line_number, category)
    static_keys: Set[Tuple[int, str]] = {(f.line_number, f.category) for f in static}

    # Add all static findings unconditionally
    combined.extend(static)

    # Add LLM findings only if they don't overlap with static findings
    for llm_finding in llm:
        if (llm_finding.line_number, llm_finding.category) not in static_keys:
            combined.append(llm_finding)

    return combined


def _apply_filters(findings: List[Finding], settings: Settings) -> List[Finding]:
    """
    Filter a list of findings by severity threshold and rule-enabled flags.

    Findings whose severity is below the configured threshold are dropped.
    Findings whose category has been disabled in settings.rules are also dropped.

    Args:
        findings: Raw list of Finding objects to filter.
        settings: Project settings carrying severity_threshold and rules config.

    Returns:
        Filtered list of Finding objects that pass all active rules.
    """
    min_severity = SEVERITY_ORDER.get(settings.severity_threshold, 3)
    rules = settings.rules
    if rules is None:
        _category_enabled = {
            "complexity": True,
            "security": True,
            "style": True,
            "docs": True,
        }
    else:
        _category_enabled = {
            "complexity": rules.complexity.enabled if rules.complexity is not None else True,
            "security": rules.security.enabled if rules.security is not None else True,
            "style": rules.style.enabled if rules.style is not None else True,
            "docs": rules.docs.enabled if rules.docs is not None else True,
        }

    result: List[Finding] = []
    for finding in findings:
        if SEVERITY_ORDER.get(finding.severity, 1) < min_severity:
            continue
        if not _category_enabled.get(finding.category, True):
            continue
        result.append(finding)
    return result


class FileReviewer:
    """Performs code review on single files using static checks and LiteLLM."""

    def __init__(self, llm_client: LLMClient, settings: Settings):
        self.llm_client = llm_client
        self.settings = settings
        self.static_analyzer = StaticAnalyzer(settings)

    def review_file(self, file_path: str) -> ReviewResult:
        """
        Review a single file using static analysis and the LLM.

        Args:
            file_path: Absolute or relative path to the Python file.

        Returns:
            A ReviewResult containing merged, filtered findings.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file cannot be read.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            raise ValueError(f"Failed to read file {file_path}: {e}") from e

        lines_reviewed = len(content.splitlines())

        # 1. Run Static Analysis
        static_findings = self.static_analyzer.analyze_file(file_path, content)

        # 2. Run LLM Analysis
        messages = [
            {"role": "system", "content": SYSTEM_REVIEW_PROMPT},
            {"role": "user", "content": USER_REVIEW_PROMPT_TEMPLATE.format(
                file_path=file_path,
                file_content=content
            )}
        ]
        response: LLMReviewResponse = self.llm_client.generate_completion(
            messages=messages,
            response_format=LLMReviewResponse
        )

        # 3. Filter and combine
        llm_findings = _apply_filters(
            [Finding(
                file_path=file_path,
                line_number=f.line_number,
                severity=f.severity,
                category=f.category,
                message=f.message,
                suggestion=f.suggestion,
                source="llm",
            ) for f in response.findings],
            self.settings,
        )
        final_findings = combine_findings(static_findings, llm_findings)

        return ReviewResult(
            file_path=file_path,
            findings=final_findings,
            summary=response.summary,
            reviewed_at=datetime.now(timezone.utc),
            model_used=self.llm_client.model,
            lines_reviewed=lines_reviewed,
        )


class DiffReviewer:
    """Performs code review on diff hunks using static checks and LiteLLM."""

    def __init__(self, llm_client: LLMClient, settings: Settings):
        self.llm_client = llm_client
        self.settings = settings
        self.static_analyzer = StaticAnalyzer(settings)

    def review_hunks(self, file_path: str, hunks: List[DiffHunk]) -> ReviewResult:
        """Review a set of diff hunks using static analysis and the multi-agent graph.

        Flow:
          1. Run StaticAnalyzer on the full file (if it exists on disk).
          2. Run the five-agent LangGraph pipeline in parallel via asyncio.run().
          3. Combine both finding sets with combine_findings() (static wins on
             same line+category).
          4. Apply severity_threshold filter.

        Args:
            file_path: Path to the file being reviewed.
            hunks:     Parsed diff hunks for this file.

        Returns:
            A ReviewResult containing the merged, filtered findings.
        """
        lines_reviewed = sum(len(h.added_lines) for h in hunks)

        # 1. Static analysis — requires full file content
        static_findings: List[Finding] = []
        try:
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                static_findings = self.static_analyzer.analyze_file(file_path, content)
        except Exception:
            logger.exception(
                "Static analysis skipped for %s — file unreadable or deleted.", file_path
            )

        # 2. Multi-agent LLM review (parallel fan-out via LangGraph)
        _agent_err: str | None = None
        try:
            raw_llm_findings: List[Finding] = asyncio.run(
                run_agent_review(hunks, self.settings.model, self.settings)
            )
        except Exception as exc:
            raw_llm_findings = []
            _agent_err = f"Agent review failed: {exc}"

        # 3. Apply severity threshold and rule-enabled filters to LLM findings
        llm_findings = _apply_filters(raw_llm_findings, self.settings)

        # 4. Combine static + LLM findings (static wins on duplicates)
        final_findings = combine_findings(static_findings, llm_findings)

        final_summary = (
            _agent_err
            if _agent_err
            else "No issues found in the provided diffs."
        )

        return ReviewResult(
            file_path=file_path,
            findings=final_findings,
            summary=final_summary,
            reviewed_at=datetime.now(timezone.utc),
            model_used=self.llm_client.model,
            lines_reviewed=lines_reviewed,
        )
