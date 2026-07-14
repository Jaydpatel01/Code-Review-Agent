"""AST-based file risk scorer for repo-level review triage.

Classifies each file into HIGH / MEDIUM / LOW risk by running the existing
ASTAnalyzer and inspecting the severity of its findings.  No LLM calls are
made here — this is a pure static analysis tier used to decide which files
need deeper LLM attention in --smart mode.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from code_reviewer.analyzers.ast_analyzer import ASTAnalyzer
from code_reviewer.core.models import Finding

if TYPE_CHECKING:
    from code_reviewer.config import Settings


def score_file_risk(
    file_path: Path,
    settings: "Settings",
) -> tuple[str, list[Finding]]:
    """Score a file's risk tier using AST analysis only.

    Reads the file, runs ASTAnalyzer, and classifies the result:
      - ``"HIGH"``   — at least one HIGH severity finding.
      - ``"MEDIUM"`` — MEDIUM findings present, but no HIGH.
      - ``"LOW"``    — only LOW/INFO findings, or the file is clean.

    On any error (SyntaxError, UnicodeDecodeError, I/O problem, etc.) the
    function returns ``("LOW", [])`` so a single bad file never crashes the
    whole-repo walk.

    Args:
        file_path: Path to the Python source file to analyse.
        settings:  Project Settings (passed to ASTAnalyzer for rule config).

    Returns:
        A ``(risk_tier, findings)`` tuple where ``risk_tier`` is one of
        ``"HIGH"``, ``"MEDIUM"``, or ``"LOW"``.
    """
    try:
        source = file_path.read_text(encoding="utf-8", errors="ignore")
        analyzer = ASTAnalyzer(str(file_path), settings)
        findings = analyzer.analyze(source)

        if any(f.severity == "HIGH" for f in findings):
            return "HIGH", findings
        if any(f.severity == "MEDIUM" for f in findings):
            return "MEDIUM", findings
        return "LOW", findings

    except Exception:
        # Graceful degradation: unknown files are treated as low risk.
        return "LOW", []
