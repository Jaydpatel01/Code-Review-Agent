"""BaseReviewAgent — abstract base class for all LangGraph review agent nodes.

Each concrete agent subclass must define:
  - name       (str property)   — key used in AgentState (e.g. "security")
  - category   (str property)   — Finding.category literal
  - system_prompt (str property) — the system instruction sent to the LLM

The __call__ method makes every agent a valid LangGraph node callable.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod

import litellm

from code_reviewer.agents.state import AgentState
from code_reviewer.core.models import DiffHunk, Finding

logger = logging.getLogger(__name__)


class BaseReviewAgent(ABC):
    """Abstract base class for specialised LangGraph review agent nodes.

    Subclasses implement three properties (name, category, system_prompt)
    and inherit the full message-building, response-parsing, and
    LangGraph node protocol from this class.
    """

    # ------------------------------------------------------------------
    # Abstract interface — subclasses must implement
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique agent name used as the AgentState key prefix.

        Example: "security" → writes to state["security_findings"].
        """

    @property
    @abstractmethod
    def category(self) -> str:
        """Finding.category literal this agent is responsible for.

        Must be one of: security | performance | style | logic | docs | complexity.
        """

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Full system prompt sent as the first message to the LLM."""

    # ------------------------------------------------------------------
    # Concrete helpers
    # ------------------------------------------------------------------

    def build_user_message(self, hunks: list[DiffHunk], codebase_context: str = "") -> str:
        """Format hunks into the user message sent to the LLM.

        Includes added lines (prefix '+') and context lines (prefix ' ').
        Removed lines are intentionally excluded — agents review what
        the code *becomes*, not what was deleted.

        Args:
            hunks: The diff hunks to format.
            codebase_context: Optional context from semantic search.

        Returns:
            A structured multi-hunk text block ready to send as user content.
        """
        parts: list[str] = []
        for hunk in hunks:
            header = (
                f"File: {hunk.file_path}\n"
                f"@@ Lines {hunk.start_line}–{hunk.end_line} @@"
            )
            lines: list[str] = []

            # Merge added and context lines, sorted by line number so the
            # LLM sees them in the natural reading order.
            combined = (
                [(ln, content, "+") for ln, content in hunk.added_lines]
                + [(ln, content, " ") for ln, content in hunk.context_lines]
            )
            combined.sort(key=lambda t: t[0])

            for ln, content, prefix in combined:
                lines.append(f"{prefix}{content}")

            parts.append(header + "\n" + "\n".join(lines))

        message = "\n\n".join(parts)
        
        # Append codebase context if provided
        if codebase_context:
            message += "\n\n## Codebase Context\n(For reference only — do not review the functions below)\n\n" + codebase_context
        
        return message

    def _extract_json_payload(self, raw: str) -> list[dict]:
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            # Drop first line (```json or ```) and last line (```)
            text = "\n".join(lines[1:-1]).strip()

        try:
            payload = json.loads(text)
            return payload.get("findings", [])
        except (json.JSONDecodeError, AttributeError, TypeError):
            logger.warning(
                "%s: failed to parse LLM JSON response — returning empty list.",
                self.name,
            )
            return []

    def _build_valid_lines(self, hunks: list[DiffHunk]) -> dict[int, str]:
        valid_lines: dict[int, str] = {}
        for hunk in hunks:
            for ln, _ in hunk.added_lines:
                valid_lines[ln] = hunk.file_path
        return valid_lines

    def _parse_single_finding(
        self, item: dict, valid_lines: dict[int, str], hunks: list[DiffHunk]
    ) -> Finding | None:
        try:
            line_val = item.get("line_number")
            line_number = int(line_val) if line_val is not None else None
        except (ValueError, TypeError):
            line_number = None

        # Hallucination guard: drop findings not on an added line
        if line_number is not None and line_number not in valid_lines:
            logger.debug(
                "%s: dropping hallucinated line_number=%s (not in added lines).",
                self.name,
                line_number,
            )
            return None

        file_path = (
            valid_lines[line_number]
            if line_number in valid_lines
            else (hunks[0].file_path if hunks else "unknown")
        )

        try:
            return Finding(
                file_path=file_path,
                line_number=line_number,
                severity=item["severity"],
                category=self.category,   # type: ignore[arg-type]
                message=item["message"],
                suggestion=item["suggestion"],
                source="llm",
            )
        except (KeyError, ValueError, TypeError) as exc:
            logger.debug("%s: skipping malformed finding item: %s", self.name, exc)
            return None

    def parse_response(self, raw: str, hunks: list[DiffHunk]) -> list[Finding]:
        """Parse the LLM's JSON response into validated Finding objects.

        Expected LLM output schema::

            {
                "findings": [
                    {
                        "line_number": 42,
                        "severity": "HIGH",
                        "category": "security",
                        "message": "...",
                        "suggestion": "..."
                    }
                ]
            }

        Rules applied during parsing:
        - source is always set to "llm".
        - category is always overridden to self.category (prevents
          the LLM from hallucinating categories outside its scope).
        - file_path is taken from the first added_line's hunk when
          line_number matches, otherwise the first hunk's file_path.
        - Any finding whose line_number is not present in any hunk's
          added_lines is silently dropped (hallucination guard).
        - If JSON parsing fails for any reason, returns [] (never raises).

        Args:
            raw:   Raw string content from the LLM response.
            hunks: The diff hunks that were reviewed.

        Returns:
            A list of validated Finding objects.
        """
        valid_lines = self._build_valid_lines(hunks)
        raw_findings = self._extract_json_payload(raw)

        findings: list[Finding] = []
        for item in raw_findings:
            if not isinstance(item, dict):
                continue
            finding = self._parse_single_finding(item, valid_lines, hunks)
            if finding:
                findings.append(finding)

        return findings

    # ------------------------------------------------------------------
    # LangGraph node protocol
    # ------------------------------------------------------------------

    def __call__(self, state: AgentState) -> dict:
        """LangGraph node callable — runs this agent on the shared state.

        Flow:
          1. Build [system, user] message pair.
          2. Call litellm.completion() directly (not LangChain wrappers).
          3. Parse findings with parse_response().
          4. Return a partial-state dict keyed by "{name}_findings".

        On any exception the agent logs the error, returns an empty
        findings list so downstream nodes can continue, and sets the
        "error" key so the aggregator can surface a warning.

        Args:
            state: The current shared AgentState.

        Returns:
            A partial-state dict, e.g. {"security_findings": [...], "error": None}.
        """
        state_key = f"{self.name}_findings"
        try:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user",   "content": self.build_user_message(
                    state["hunks"], 
                    state.get("codebase_context", "")
                )},
            ]

            response = litellm.completion(
                model=state["model"],
                messages=messages,
            )
            raw = response.choices[0].message.content or ""
            findings = self.parse_response(raw, state["hunks"])

            logger.info(
                "%s: completed — %d finding(s) returned.", self.name, len(findings)
            )
            return {state_key: findings, "error": None}

        except Exception as exc:  # noqa: BLE001
            logger.error(
                "%s: unhandled exception during review: %s", self.name, exc, exc_info=True
            )
            return {state_key: [], "error": f"{self.name}: {exc}"}
