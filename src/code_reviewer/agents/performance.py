"""PerformanceAgent — specialised review agent for performance issues.

Domain: DB calls inside loops (N+1), O(n²) algorithms, repeated
identical function calls (missing memoization), sync I/O in async
context, unnecessary list materializations.
Severity bias: MEDIUM or LOW.
"""

from __future__ import annotations

from code_reviewer.agents.base_agent import BaseReviewAgent


class PerformanceAgent(BaseReviewAgent):
    """LangGraph agent node that reviews code for performance issues.

    Flags issues such as N+1 database query patterns, quadratic-time
    algorithms, redundant repeated calls, blocking synchronous I/O inside
    async functions, and unnecessary eager list materializations.
    Findings are MEDIUM or LOW severity — never HIGH.
    """

    @property
    def name(self) -> str:
        """Agent name — used as the AgentState key prefix."""
        return "performance"

    @property
    def category(self) -> str:
        """Finding category literal for all findings produced by this agent."""
        return "performance"

    @property
    def system_prompt(self) -> str:
        """System instruction sent to the LLM for performance review."""
        return """\
You are a senior performance engineer performing a focused performance \
code review.

Only flag issues in your specific domain. Do not comment on security, \
style, documentation, or correctness bugs.

Focus exclusively on these performance problem classes:
- N+1 query patterns: database or API calls placed inside loops where \
a single batched call would suffice
- O(n²) or worse algorithms: nested loops over the same collection, \
repeated linear searches in inner loops
- Missing memoization: identical expensive function calls (DB, network, \
heavy computation) repeated with the same arguments without caching
- Synchronous / blocking I/O inside an async context: time.sleep(), \
open(), requests.get() called with await-able alternatives available
- Unnecessary list materializations: wrapping generators in list() when \
only iteration is needed, or using list comprehensions where a generator \
expression would do

Severity rules:
- Use MEDIUM for patterns that cause measurable latency or resource \
waste in production (N+1, sync I/O in async, O(n²)).
- Use LOW for stylistic inefficiencies that rarely matter at scale \
(unnecessary list(), minor redundant calls).
- NEVER use HIGH or INFO.

Only comment on added lines (+). Ignore context lines.
Never invent line numbers. Use only line numbers shown.
If no issues found, return {"findings": []}

Return JSON only — no prose, no markdown:
{"findings": [{"line_number": int, "severity": "MEDIUM"|"LOW", \
"message": str, "suggestion": str}]}"""
