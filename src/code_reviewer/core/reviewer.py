"""Module for reviewing files using LLM clients."""

import os
from datetime import datetime, timezone
from typing import List
from code_reviewer.core.llm_client import LLMClient
from code_reviewer.core.models import Finding, ReviewResult, LLMReviewResponse, DiffHunk
from code_reviewer.core.prompts import (
    SYSTEM_REVIEW_PROMPT,
    USER_REVIEW_PROMPT_TEMPLATE,
    SYSTEM_DIFF_REVIEW_PROMPT,
    USER_DIFF_REVIEW_PROMPT_TEMPLATE
)
from code_reviewer.config import Settings

SEVERITY_ORDER = {
    "INFO": 1,
    "LOW": 2,
    "MEDIUM": 3,
    "HIGH": 4
}


class FileReviewer:
    """Performs code review on single files using LiteLLM."""

    def __init__(self, llm_client: LLMClient, settings: Settings):
        """
        Initialize the FileReviewer.

        Args:
            llm_client: The LLM client to use for reviews.
            settings: Configuration settings for filtering and rules.
        """
        self.llm_client = llm_client
        self.settings = settings

    def review_file(self, file_path: str) -> ReviewResult:
        """
        Review a single file, filtering results based on config settings.

        Args:
            file_path: Absolute or relative path to the file to review.

        Returns:
            ReviewResult: The filtered and completed review results.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is not a text file or cannot be read.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            raise ValueError(f"Failed to read file {file_path}: {e}") from e

        lines = content.splitlines()
        lines_reviewed = len(lines)

        messages = [
            {"role": "system", "content": SYSTEM_REVIEW_PROMPT},
            {"role": "user", "content": USER_REVIEW_PROMPT_TEMPLATE.format(
                file_path=file_path,
                file_content=content
            )}
        ]

        # Call the LLM to get structured findings
        response: LLMReviewResponse = self.llm_client.generate_completion(
            messages=messages,
            response_format=LLMReviewResponse
        )

        # Filter and map findings
        filtered_findings: List[Finding] = []
        min_severity_level = SEVERITY_ORDER.get(self.settings.severity_threshold, 3)

        for llm_finding in response.findings:
            severity = llm_finding.severity
            category = llm_finding.category

            # 1. Filter by severity threshold
            if SEVERITY_ORDER.get(severity, 1) < min_severity_level:
                continue

            # 2. Filter by rule enabled state
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

            # Add mapped finding
            filtered_findings.append(
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

        return ReviewResult(
            file_path=file_path,
            findings=filtered_findings,
            summary=response.summary,
            reviewed_at=datetime.now(timezone.utc),
            model_used=self.llm_client.model,
            lines_reviewed=lines_reviewed
        )

class DiffReviewer:
    """Performs code review on diff hunks using LiteLLM."""

    def __init__(self, llm_client: LLMClient, settings: Settings):
        """
        Initialize the DiffReviewer.

        Args:
            llm_client: The LLM client to use for reviews.
            settings: Configuration settings for filtering and rules.
        """
        self.llm_client = llm_client
        self.settings = settings

    def review_hunks(self, file_path: str, hunks: List[DiffHunk]) -> ReviewResult:
        """
        Review a list of diff hunks for a single file.

        Args:
            file_path: The path of the file being reviewed.
            hunks: List of DiffHunk instances for this file.

        Returns:
            ReviewResult: The aggregated review results for all hunks.
        """
        all_findings: List[Finding] = []
        summaries: List[str] = []
        lines_reviewed = 0
        min_severity_level = SEVERITY_ORDER.get(self.settings.severity_threshold, 3)

        for hunk in hunks:
            if not hunk.added_lines:
                continue

            lines_reviewed += len(hunk.added_lines)
            
            added_text = "\n".join(f"{line_num}: {content}" for line_num, content in hunk.added_lines)
            context_text = "\n".join(f"{line_num}: {content}" for line_num, content in hunk.context_lines)

            messages = [
                {"role": "system", "content": SYSTEM_DIFF_REVIEW_PROMPT},
                {"role": "user", "content": USER_DIFF_REVIEW_PROMPT_TEMPLATE.format(
                    file_path=file_path,
                    start_line=hunk.start_line,
                    end_line=hunk.end_line,
                    added_lines=added_text,
                    context_lines=context_text
                )}
            ]

            response: LLMReviewResponse = self.llm_client.generate_completion(
                messages=messages,
                response_format=LLMReviewResponse
            )

            if response.summary:
                summaries.append(response.summary)

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

                # Ensure line_number is restricted to added lines if LLM hallucinated
                valid_line_nums = {ln for ln, _ in hunk.added_lines}
                reported_line = llm_finding.line_number
                if reported_line not in valid_line_nums:
                    # If it doesn't match an added line, default to the first added line
                    # or null. We'll default to the first added line to map it somewhere in the diff.
                    reported_line = hunk.added_lines[0][0]

                all_findings.append(
                    Finding(
                        file_path=file_path,
                        line_number=reported_line,
                        severity=severity,
                        category=category,
                        message=llm_finding.message,
                        suggestion=llm_finding.suggestion,
                        source="llm"
                    )
                )

        final_summary = "\n\n".join(summaries) if summaries else "No issues found in the provided diffs."

        return ReviewResult(
            file_path=file_path,
            findings=all_findings,
            summary=final_summary,
            reviewed_at=datetime.now(timezone.utc),
            model_used=self.llm_client.model,
            lines_reviewed=lines_reviewed
        )
