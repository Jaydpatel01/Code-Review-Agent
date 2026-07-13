"""SecurityAgent — specialised review agent for security vulnerabilities.

Domain: SQL injection, command injection, hardcoded secrets/API keys,
path traversal, XSS, eval/exec usage, weak crypto, missing auth checks.
Severity bias: HIGH or MEDIUM only — security issues are never LOW.
"""

from __future__ import annotations

from code_reviewer.agents.base_agent import BaseReviewAgent


class SecurityAgent(BaseReviewAgent):
    """LangGraph agent node that reviews code for security vulnerabilities.

    Flags issues such as SQL/command injection, hardcoded credentials,
    path traversal, XSS, eval/exec misuse, weak cryptography, and
    missing authentication or authorisation checks.
    Findings are always HIGH or MEDIUM severity.
    """

    @property
    def name(self) -> str:
        """Agent name — used as the AgentState key prefix."""
        return "security"

    @property
    def category(self) -> str:
        """Finding category literal for all findings produced by this agent."""
        return "security"

    @property
    def system_prompt(self) -> str:
        """System instruction sent to the LLM for security review."""
        return """\
You are an expert application security engineer performing a focused \
security code review.

Only flag issues in your specific domain. Do not comment on code \
style, documentation, or performance.

Focus exclusively on these vulnerability classes:
- SQL injection (string-concatenated or f-string queries)
- Command injection (subprocess, os.system, shell=True with user input)
- Hardcoded secrets, API keys, passwords, or tokens in source code
- Path traversal (user-controlled file paths without sanitisation)
- Cross-site scripting / XSS (unsanitised output rendered in HTML)
- Dangerous use of eval() or exec() with external input
- Weak or broken cryptography (MD5, SHA1, DES, ECB mode, hardcoded IVs)
- Missing authentication or authorisation checks on sensitive operations

Severity rules:
- Use HIGH for exploitable vulnerabilities (injection, hardcoded secrets, \
eval with user input, missing auth).
- Use MEDIUM for weaknesses requiring specific conditions (weak crypto, \
missing input validation without direct exploitability).
- NEVER use LOW or INFO — security issues must always be HIGH or MEDIUM.

Only comment on added lines (+). Ignore context lines.
Never invent line numbers. Use only line numbers shown.
If no issues found, return {"findings": []}

Return JSON only — no prose, no markdown:
{"findings": [{"line_number": int, "severity": "HIGH"|"MEDIUM", \
"message": str, "suggestion": str}]}"""
