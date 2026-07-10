"""Module for reviewing files using LLM clients and static analysis."""

import os
from datetime import datetime, timezone
from typing import List, Set, Tuple
from code_reviewer.core.llm_client import LLMClient
from code_reviewer.core.models import Finding, ReviewResult, LLMReviewResponse, DiffHunk
from code_reviewer.core.prompts import (
    SYSTEM_REVIEW_PROMPT,
    USER_REVIEW_PROMPT_TEMPLATE,
    SYSTEM_DIFF_REVIEW_PROMPT,
    USER_DIFF_REVIEW_PROMPT_TEMPLATE
)
from code_reviewer.config import Settings
from code_reviewer.analyzers.static_checks import StaticAnalyzer

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
        all_findings: List[Finding] = []
        summaries: List[str] = []
        lines_reviewed = 0
        min_severity_level = SEVERITY_ORDER.get(self.settings.severity_threshold, 3)

        # 1. Run Static Analysis (requires full file content if possible)
        static_findings = []
        try:
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                static_findings = self.static_analyzer.analyze_file(file_path, content)
        except Exception:
            pass # If file doesn't exist on disk (deleted), static analysis is skipped.

        for hunk in hunks:
            if not hunk.added_lines:
                continue

            lines_reviewed += len(hunk.added_lines)
            valid_line_nums = {ln for ln, _ in hunk.added_lines}
            
            # Filter static findings to this specific hunk
            hunk_static_findings = [f for f in static_findings if f.line_number in valid_line_nums]
            
            # Check if AST fully covers this hunk (every added line has an AST finding)
            is_fully_covered = len(hunk_static_findings) >= len(hunk.added_lines) and all(
                ln in [f.line_number for f in hunk_static_findings] for ln in valid_line_nums
            )

            llm_findings = []
            
            # Only send to LLM if AST didn't already fully cover it
            if not is_fully_covered:
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

                try:
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

                        reported_line = llm_finding.line_number
                        if reported_line not in valid_line_nums:
                            reported_line = hunk.added_lines[0][0]

                        llm_findings.append(
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
                except Exception as e:
                    # Log error but continue processing other hunks
                    summaries.append(f"LLM review failed for hunk starting at line {hunk.start_line}: {str(e)}")

            # Combine and deduplicate
            hunk_combined = combine_findings(hunk_static_findings, llm_findings)
            all_findings.extend(hunk_combined)

        final_summary = "\n\n".join(summaries) if summaries else "No issues found in the provided diffs."

        return ReviewResult(
            file_path=file_path,
            findings=all_findings,
            summary=final_summary,
            reviewed_at=datetime.now(timezone.utc),
            model_used=self.llm_client.model,
            lines_reviewed=lines_reviewed
        )
