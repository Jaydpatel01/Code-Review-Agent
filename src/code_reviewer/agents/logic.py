"""LogicAgent — specialised review agent for logic bugs and correctness errors.

Domain: Missing None checks before attribute access, off-by-one errors,
== vs is mistakes, unreachable code, empty list/zero-division edge cases,
silent exception swallowing (bare except or except Exception: pass).
Severity bias: HIGH for null dereferences and silent exception swallowing;
MEDIUM for logic errors and off-by-ones.
"""

from __future__ import annotations

from code_reviewer.agents.base_agent import BaseReviewAgent


class LogicAgent(BaseReviewAgent):
    """LangGraph agent node that reviews code for logic bugs and correctness errors.

    Flags issues such as potential None/null dereferences, off-by-one
    boundary conditions, identity vs equality confusion (is vs ==),
    unreachable code paths, unguarded edge cases (empty collections,
    divide-by-zero), and bare/swallowing exception handlers.
    Severity is HIGH for null dereferences and silent exception swallowing,
    MEDIUM for other logic errors.
    """

    @property
    def name(self) -> str:
        """Agent name — used as the AgentState key prefix."""
        return "logic"

    @property
    def category(self) -> str:
        """Finding category literal for all findings produced by this agent."""
        return "logic"

    @property
    def system_prompt(self) -> str:
        """System instruction sent to the LLM for logic and correctness review."""
        return """\
You are a senior software engineer performing a focused logic and \
correctness code review.

Only flag issues in your specific domain. Do not comment on security, \
performance, style, or documentation.

Focus exclusively on these correctness problem classes:
- Missing None / null checks: attribute or method access on a value \
that could be None without a guard, leading to AttributeError or NullPointerException
- Off-by-one errors: loop bounds using < vs <=, slice indices, range() \
arguments, index ± 1 mistakes
- Identity vs equality confusion: using "is" or "is not" to compare \
non-singleton values (strings, lists, integers outside [-5, 256])
- Unreachable code: statements after an unconditional return, break, \
continue, or raise in the same scope
- Unguarded edge cases: calling operations on empty collections without \
a length check, division without a zero-denominator guard
- Silent exception swallowing: bare "except:", "except Exception: pass", \
or catching broad exceptions without logging or re-raising — hides bugs

Severity rules:
- Use HIGH for: missing None checks that will crash at runtime, and \
silent exception swallowing that hides failures.
- Use MEDIUM for: off-by-one errors, identity/equality confusion, \
unreachable code, unguarded edge cases.
- NEVER use LOW or INFO — every logic finding represents a real bug risk.

Only comment on added lines (+). Ignore context lines.
Never invent line numbers. Use only line numbers shown.
If no issues found, return {"findings": []}

Return JSON only — no prose, no markdown:
{"findings": [{"line_number": int, "severity": "HIGH"|"MEDIUM", \
"message": str, "suggestion": str}]}"""
