"""PR review orchestrator — ties together diff parsing, static analysis, LLM review, and GitHub comments."""

import logging
from collections import defaultdict
from typing import Optional

from code_reviewer.analyzers.diff_parser import parse_diff
from code_reviewer.analyzers.static_checks import StaticAnalyzer
from code_reviewer.config import Settings
from code_reviewer.core.llm_client import LLMClient
from code_reviewer.core.models import Finding, ReviewResult
from code_reviewer.core.reviewer import DiffReviewer, combine_findings
from code_reviewer.integrations.github_client import GitHubClient

logger = logging.getLogger(__name__)

SEVERITY_ORDER = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "INFO": 3}
SEVERITY_EMOJI = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢", "INFO": "ℹ️"}


class PRReviewer:
    """
    Orchestrates the full pull-request review pipeline.

    Flow:
      1. Fetch raw unified diff via GitHubClient
      2. Parse diff → List[DiffHunk]
      3. Build diff-position map (line_number → diff position per file)
      4. Run StaticAnalyzer on changed .py files
      5. Run DiffReviewer (LLM) per file
      6. combine_findings() — static takes precedence
      7. Post inline comments for findings that land on + lines
      8. Post a formatted summary comment
    """

    def __init__(
        self,
        github_client: GitHubClient,
        llm_client: LLMClient,
        settings: Settings,
    ) -> None:
        self.github_client = github_client
        self.llm_client = llm_client
        self.settings = settings
        self._static_analyzer = StaticAnalyzer(settings)
        self._diff_reviewer = DiffReviewer(llm_client, settings)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def review_pr(self, repo_name: str, pr_number: int) -> None:
        """
        Run the full review pipeline for a pull request and post results.

        Args:
            repo_name:  Full repository name, e.g. ``"owner/repo"``.
            pr_number:  Pull-request number.
        """
        logger.info("Starting review for %s#%d", repo_name, pr_number)

        raw_diff = self.github_client.get_pr_diff(repo_name, pr_number)
        if not raw_diff.strip():
            logger.info("Empty diff for %s#%d — nothing to review.", repo_name, pr_number)
            return

        commit_sha = self.github_client.get_pr_head_sha(repo_name, pr_number)

        # Build the position map from the raw diff *before* parsing hunks
        # so both analyses share the same authoritative mapping.
        position_map = self._build_diff_position_map(raw_diff)

        # Parse diff into structured hunks grouped by file
        hunks = parse_diff(raw_diff)
        hunks_by_file: dict[str, list] = defaultdict(list)
        for hunk in hunks:
            hunks_by_file[hunk.file_path].append(hunk)

        all_results: list[ReviewResult] = []
        for file_path, file_hunks in hunks_by_file.items():
            logger.debug("Reviewing file: %s (%d hunks)", file_path, len(file_hunks))
            try:
                result = self._diff_reviewer.review_hunks(file_path, file_hunks)
                all_results.append(result)
            except Exception:
                logger.exception("Review failed for file %s", file_path)

        # Flatten all findings
        all_findings: list[Finding] = [f for r in all_results for f in r.findings]

        # Post inline comments
        inline_count = 0
        for finding in all_findings:
            file_pos_map = position_map.get(finding.file_path, {})
            if finding.line_number is None:
                continue
            diff_position = file_pos_map.get(finding.line_number)
            if diff_position is None:
                continue  # Finding is not on an added line — goes to summary only
            try:
                comment_body = self._format_inline_comment(finding)
                self.github_client.post_inline_comment(
                    repo_name=repo_name,
                    pr_number=pr_number,
                    commit_sha=commit_sha,
                    file_path=finding.file_path,
                    diff_position=diff_position,
                    body=comment_body,
                )
                inline_count += 1
            except Exception:
                logger.exception(
                    "Failed to post inline comment for %s:%s",
                    finding.file_path,
                    finding.line_number,
                )

        # Post summary
        summary_body = self._format_pr_summary(
            files_reviewed=list(hunks_by_file.keys()),
            findings=all_findings,
        )
        try:
            self.github_client.post_pr_summary(repo_name, pr_number, summary_body)
        except Exception:
            logger.exception("Failed to post PR summary for %s#%d", repo_name, pr_number)

        logger.info(
            "Review complete for %s#%d: %d findings, %d inline comments posted.",
            repo_name,
            pr_number,
            len(all_findings),
            inline_count,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_diff_position_map(
        self, raw_diff: str
    ) -> dict[str, dict[int, int]]:
        """
        Build a mapping of ``file_path → {line_number: diff_position}``.

        GitHub's PR review API requires a ``position`` parameter that is the
        line's 1-indexed offset *within the file's diff block* (not the file
        line number).  The rules are:

        - Position resets to 0 at the start of each new file.
        - The ``@@ … @@`` hunk header line counts as position 1.
        - Every subsequent line (context, added ``+``, or removed ``-``)
          increments the position counter.
        - Only ``+`` (added) lines are inserted into the output map, because
          only they can receive inline comments via the GitHub API.
        - ``-`` (removed) lines increment the counter but are NOT mapped.

        Args:
            raw_diff: Raw unified diff string as returned by the GitHub API.

        Returns:
            Dict keyed by file path; each value maps file line numbers to
            their diff position.
        """
        result: dict[str, dict[int, int]] = {}

        current_file: Optional[str] = None
        position = 0
        current_new_line = 0

        import re
        hunk_header_re = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")

        for line in raw_diff.splitlines():
            if line.startswith("diff --git"):
                current_file = None
                position = 0
                continue

            if line.startswith("+++ b/"):
                current_file = line[6:]
                result.setdefault(current_file, {})
                position = 0
                continue

            if line.startswith("+++ /dev/null") or line.startswith("--- "):
                continue

            if current_file is None:
                continue

            m = hunk_header_re.match(line)
            if m:
                # Hunk header is always position 1 (within this hunk's block).
                # Per the GitHub docs, position is reset per-file, not per-hunk.
                position += 1
                current_new_line = int(m.group(1))
                continue

            if line.startswith("+"):
                position += 1
                result[current_file][current_new_line] = position
                current_new_line += 1
            elif line.startswith("-"):
                position += 1  # counter increments but we do NOT map it
            elif line.startswith(" "):
                position += 1
                current_new_line += 1
            # Lines like `\ No newline at end of file` — ignore

        return result

    @staticmethod
    def _format_inline_comment(finding: Finding) -> str:
        """
        Format a single Finding as a Markdown inline comment body.

        Args:
            finding: The finding to format.

        Returns:
            Markdown string for the GitHub comment body.
        """
        emoji = SEVERITY_EMOJI.get(finding.severity, "•")
        lines = [
            f"{emoji} **[{finding.severity}]** `{finding.category}` — {finding.message}",
        ]
        if finding.suggestion:
            lines.append(f"> 💡 {finding.suggestion}")
        lines.append(f"*Source: `{finding.source}`*")
        return "\n\n".join(lines)

    @staticmethod
    def _format_pr_summary(
        files_reviewed: list[str],
        findings: list[Finding],
    ) -> str:
        """
        Build the formatted PR summary comment.

        Args:
            files_reviewed: List of file paths that were reviewed.
            findings:       All findings across all files.

        Returns:
            Markdown string for the GitHub PR summary comment.
        """
        high = [f for f in findings if f.severity == "HIGH"]
        medium = [f for f in findings if f.severity == "MEDIUM"]
        low_info = [f for f in findings if f.severity in ("LOW", "INFO")]

        lines = [
            "## AI Code Review Summary",
            "",
            f"**Files reviewed:** {len(files_reviewed)}  ",
            f"**Total findings:** {len(findings)} "
            f"({len(high)} high, {len(medium)} medium, {len(low_info)} low/info)",
            "",
        ]

        def _section(title: str, emoji: str, section_findings: list[Finding]) -> list[str]:
            if not section_findings:
                return []
            block = [f"### {emoji} {title}", ""]
            for f in sorted(section_findings, key=lambda x: (x.file_path, x.line_number or 0)):
                loc = f"`{f.file_path}:{f.line_number}`" if f.line_number else f"`{f.file_path}`"
                block.append(f"- {loc} · `{f.category}` · {f.message}")
            block.append("")
            return block

        lines += _section("High Severity", "🔴", high)
        lines += _section("Medium Severity", "🟡", medium)
        lines += _section("Low / Info", "🟢", low_info)

        if not findings:
            lines.append("✅ No issues found — looks good!")
            lines.append("")

        lines.append("> Reviewed by AI Code Reviewer · source: static+llm")
        return "\n".join(lines)
