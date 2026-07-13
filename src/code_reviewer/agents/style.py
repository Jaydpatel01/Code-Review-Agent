"""StyleAgent — specialised review agent for code style issues.

Domain: Misleading names, single-letter variables outside loops, functions
doing more than one thing (SRP violation), dead/commented-out code.
Severity bias: LOW or INFO only. Never HIGH.

IMPORTANT: Does NOT flag cyclomatic complexity or nesting depth —
those are handled by the AST static analyzer.
"""

from __future__ import annotations

from code_reviewer.agents.base_agent import BaseReviewAgent


class StyleAgent(BaseReviewAgent):
    """LangGraph agent node that reviews code for style issues.

    Flags issues such as misleading or uninformative variable/function
    names, single-letter variables used outside of loop iteration,
    functions that clearly do more than one thing (Single Responsibility
    Principle violations), and commented-out or dead code left in the
    codebase.
    Findings are LOW or INFO severity only — never HIGH or MEDIUM.
    Does NOT flag complexity or nesting (handled by AST analysis).
    """

    @property
    def name(self) -> str:
        """Agent name — used as the AgentState key prefix."""
        return "style"

    @property
    def category(self) -> str:
        """Finding category literal for all findings produced by this agent."""
        return "style"

    @property
    def system_prompt(self) -> str:
        """System instruction sent to the LLM for style review."""
        return """\
You are a senior software engineer performing a focused code style \
and readability review.

Only flag issues in your specific domain. Do not comment on security, \
performance, logic bugs, or documentation content.

Focus exclusively on these style problem classes:
- Misleading names: variable or function names that imply the opposite \
of what they do (e.g. "is_active" that returns a count, "data" used for \
a list of users)
- Single-letter variable names outside of loop iterators: variables named \
a, b, x, y, z used as meaningful state outside of a for/while loop
- Single Responsibility Principle violations: functions that clearly do \
more than one distinct thing (e.g. fetch data AND render output AND write \
to a file in the same function body)
- Dead or commented-out code: blocks of code commented out with # or \
inside triple-quoted strings that serve no documentation purpose, \
TODO blocks that have been there without an issue reference

IMPORTANT: Do NOT flag cyclomatic complexity, nesting depth, or function \
length — those are handled separately by static analysis.

Severity rules:
- Use LOW for issues that reduce readability meaningfully.
- Use INFO for minor style preferences that are suggestive only.
- NEVER use HIGH or MEDIUM — style issues are not bugs.

Only comment on added lines (+). Ignore context lines.
Never invent line numbers. Use only line numbers shown.
If no issues found, return {"findings": []}

Return JSON only — no prose, no markdown:
{"findings": [{"line_number": int, "severity": "LOW"|"INFO", \
"message": str, "suggestion": str}]}"""
