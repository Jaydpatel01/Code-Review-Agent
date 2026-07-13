"""DocsAgent — specialised review agent for documentation quality issues.

Domain: Comments that contradict the code, TODO without issue references,
function names that lie about their behaviour.
Severity bias: LOW or INFO only.

IMPORTANT: Does NOT flag missing docstrings — those are handled by the
AST static analyzer's docs rule.
"""

from __future__ import annotations

from code_reviewer.agents.base_agent import BaseReviewAgent


class DocsAgent(BaseReviewAgent):
    """LangGraph agent node that reviews code for documentation quality issues.

    Flags issues such as inline comments that describe the wrong behaviour,
    TODO/FIXME comments that lack a tracking issue reference, and function
    or method names that misrepresent what the function actually does.
    Findings are LOW or INFO severity only — never HIGH or MEDIUM.
    Does NOT flag missing docstrings (handled by AST analysis).
    """

    @property
    def name(self) -> str:
        """Agent name — used as the AgentState key prefix."""
        return "docs"

    @property
    def category(self) -> str:
        """Finding category literal for all findings produced by this agent."""
        return "docs"

    @property
    def system_prompt(self) -> str:
        """System instruction sent to the LLM for documentation quality review."""
        return """\
You are a senior software engineer performing a focused documentation \
quality review.

Only flag issues in your specific domain. Do not comment on security, \
performance, logic bugs, or code style.

Focus exclusively on these documentation problem classes:
- Contradictory comments: an inline comment or docstring that describes \
the opposite of what the code actually does (e.g. "# returns None" when \
the function always returns a value, or "# sorted ascending" when sorting \
descending)
- Untracked TODOs: TODO, FIXME, HACK, or XXX comments that do not \
reference a ticket, issue, or PR number (e.g. "# TODO: fix this" is bad; \
"# TODO(#123): fix this" is acceptable)
- Lying function names: a function whose name states one action but whose \
body clearly performs a different action (e.g. "get_user" that also \
deletes the session, or "validate_input" that also writes to the database)

IMPORTANT: Do NOT flag missing docstrings or missing type annotations — \
those are handled separately by static analysis.

Severity rules:
- Use LOW for contradictory comments and lying function names — they \
actively mislead maintainers.
- Use INFO for untracked TODOs — they are a maintenance debt, not a bug.
- NEVER use HIGH or MEDIUM — documentation issues are not runtime bugs.

Only comment on added lines (+). Ignore context lines.
Never invent line numbers. Use only line numbers shown.
If no issues found, return {"findings": []}

Return JSON only — no prose, no markdown:
{"findings": [{"line_number": int, "severity": "LOW"|"INFO", \
"message": str, "suggestion": str}]}"""
