"""Module for reviewing files using LLM clients and static analysis."""

import asyncio
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


class FileReviewer:
    """Performs code review on single files using static checks and LiteLLM."""

    def __init__(self, llm_client: LLMClient, settings: Settings):
        self.llm_client = llm_client
        self.settings = settings
        self.static_analyzer = StaticAnalyzer(settings)

    def review_file(self, file_path: str) -> ReviewResult:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            raise ValueError(f"Failed to read file {file_path}: {e}") from e

        lines = content.splitlines()
        lines_reviewed = len(lines)
        
        # 1. Run Static Analysis First
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

        llm_findings = []
        min_severity_level = SEVERITY_ORDER.get(self.settings.severity_threshold, 3)

        for llm_finding in response.findings:
            severity = llm_finding.severity
            category = llm_finding.category

            if SEVERITY_ORDER.get(severity, 1) < min_severity_level:
                continue

            rule_enabled = True
            if category == "complexity" and not self.settings.rules.complexity.enabled:
                rule_enabled = False
            elif category == "security" and not self.settings.rules.security.enabled:
                rule_enabled = False
            elif category == "style" and not self.settings.rules.style.enabled:
                rule_enabled = False
            elif category == "docs" and not self.settings.rules.docs.enabled:
                rule_enabled = False

            if not rule_enabled:
                continue

            llm_findings.append(
                Finding(
                    file_path=file_path,
                    line_number=llm_finding.line_number,
                    severity=severity,
                    category=category,
                    message=llm_finding.message,
                    suggestion=llm_finding.suggestion,
                    source="llm"
                )
            )

        # 3. Combine and Deduplicate
        final_findings = combine_findings(static_findings, llm_findings)

        return ReviewResult(
            file_path=file_path,
            findings=final_findings,
            summary=response.summary,
            reviewed_at=datetime.now(timezone.utc),
            model_used=self.llm_client.model,
            lines_reviewed=lines_reviewed
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
        min_severity_level = SEVERITY_ORDER.get(self.settings.severity_threshold, 3)

        # 1. Static analysis — requires full file content
        static_findings: List[Finding] = []
        try:
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                static_findings = self.static_analyzer.analyze_file(file_path, content)
        except Exception:
            pass  # File deleted or unreadable — static analysis skipped.

        # 2. Multi-agent LLM review (parallel fan-out via LangGraph)
        try:
            raw_llm_findings: List[Finding] = asyncio.run(
                run_agent_review(hunks, self.settings.model, self.settings)
            )
        except Exception as exc:
            raw_llm_findings = []
            # Surface the error in the summary but don't crash.
            _agent_err = f"Agent review failed: {exc}"
        else:
            _agent_err = None

        # 3. Apply severity threshold and rule-enabled filters to LLM findings
        llm_findings: List[Finding] = []
        for finding in raw_llm_findings:
            if SEVERITY_ORDER.get(finding.severity, 1) < min_severity_level:
                continue

            rule_enabled = True
            cat = finding.category
            if cat == "complexity" and not self.settings.rules.complexity.enabled:
                rule_enabled = False
            elif cat == "security" and not self.settings.rules.security.enabled:
                rule_enabled = False
            elif cat == "style" and not self.settings.rules.style.enabled:
                rule_enabled = False
            elif cat == "docs" and not self.settings.rules.docs.enabled:
                rule_enabled = False

            if not rule_enabled:
                continue

            llm_findings.append(finding)

        # 4. Combine static + LLM findings (static wins on duplicates)
        final_findings = combine_findings(static_findings, llm_findings)

        summary_parts = []
        if _agent_err:
            summary_parts.append(_agent_err)
        final_summary = (
            "\n\n".join(summary_parts)
            if summary_parts
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
