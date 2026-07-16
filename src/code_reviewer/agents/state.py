"""AgentState — shared state dictionary for the LangGraph review pipeline.

All agents read from and write to this state. Inputs (hunks, model,
context_lines_count) are set once by the graph entrypoint and are
treated as read-only by individual agent nodes.
"""

from __future__ import annotations

from typing import Optional
from typing_extensions import TypedDict, Annotated

from code_reviewer.core.models import DiffHunk, Finding


def _keep_last_error(current: Optional[str], update: Optional[str]) -> Optional[str]:
    """Reducer for the 'error' field in AgentState.

    When five agents run in parallel they may all write to 'error'.
    LangGraph requires a reducer for any key written by concurrent nodes.
    This reducer returns the update value if it is non-None, otherwise
    keeps the current value — i.e. the first real error wins.

    Args:
        current: The existing error value in the channel.
        update:  The value being written by the current node.

    Returns:
        The update if non-None, else current.
    """
    if update is not None:
        return update
    return current


class AgentState(TypedDict):
    """Shared state passed between all nodes in the review graph.

    Input fields (set by the graph entrypoint, never modified by agents):
        hunks:               The parsed diff hunks to review.
        model:               The litellm model string (e.g. 'gemini/...').
        context_lines_count: How many context lines were included per hunk.

    Per-agent output fields (each agent writes only its own slice):
        security_findings:    Findings from the SecurityAgent.
        performance_findings: Findings from the PerformanceAgent.
        logic_findings:       Findings from the LogicAgent.
        style_findings:       Findings from the StyleAgent.
        docs_findings:        Findings from the DocsAgent.

    Aggregator output:
        final_findings: Deduplicated, merged findings from all agents.

    Error tracking:
        error: Human-readable error string if any node failed, else None.
    """

    # ---- inputs --------------------------------------------------------
    hunks: list[DiffHunk]
    model: str
    context_lines_count: int
    codebase_context: str  # Additional context from semantic search

    # ---- per-agent outputs ---------------------------------------------
    security_findings: list[Finding]
    performance_findings: list[Finding]
    logic_findings: list[Finding]
    style_findings: list[Finding]
    docs_findings: list[Finding]

    # ---- aggregator output ---------------------------------------------
    final_findings: list[Finding]

    # ---- error tracking ------------------------------------------------
    error: Annotated[Optional[str], _keep_last_error]
