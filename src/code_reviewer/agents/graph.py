"""LangGraph orchestrator graph for the multi-agent review pipeline.

Topology (parallel fan-out → aggregate fan-in):

  START ──┬──► security  ──┐
          ├──► performance─┤
          ├──► logic     ──┼──► aggregate ──► END
          ├──► style     ──┤
          └──► docs      ──┘

All five agents run in parallel (START → each agent). Their results are
collected by a single aggregate node that deduplicates across agents.

Public API:
  review_graph   — compiled StateGraph, usable by tests and the wiring layer
  run_agent_review() — async helper that builds state and returns final findings
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langgraph.graph import StateGraph, START, END

from code_reviewer.agents.state import AgentState
from code_reviewer.agents.security import SecurityAgent
from code_reviewer.agents.performance import PerformanceAgent
from code_reviewer.agents.logic import LogicAgent
from code_reviewer.agents.style import StyleAgent
from code_reviewer.agents.docs import DocsAgent
from code_reviewer.core.models import DiffHunk, Finding

if TYPE_CHECKING:
    from code_reviewer.config import Settings

logger = logging.getLogger(__name__)

# Severity ranking used for deduplication — higher number wins.
_SEVERITY_ORDER: dict[str, int] = {
    "INFO": 1,
    "LOW": 2,
    "MEDIUM": 3,
    "HIGH": 4,
}


# ---------------------------------------------------------------------------
# Aggregate node
# ---------------------------------------------------------------------------

def aggregate_findings(state: AgentState) -> dict:
    """Merge and deduplicate findings from all five review agents.

    Collects all per-agent finding lists from the shared state, then
    deduplicates on the key (file_path, line_number, category). When
    duplicates exist the finding with the higher severity is kept.

    Args:
        state: The fully-populated AgentState after all agents have run.

    Returns:
        A partial-state dict: {"final_findings": deduplicated_list}.
    """
    all_findings: list[Finding] = (
        state.get("security_findings", [])
        + state.get("performance_findings", [])
        + state.get("logic_findings", [])
        + state.get("style_findings", [])
        + state.get("docs_findings", [])
    )

    # Deduplicate: same (file_path, line_number, category) → keep highest severity
    best: dict[tuple, Finding] = {}
    for finding in all_findings:
        key = (finding.file_path, finding.line_number, finding.category)
        existing = best.get(key)
        if existing is None:
            best[key] = finding
        else:
            if _SEVERITY_ORDER.get(finding.severity, 0) > _SEVERITY_ORDER.get(existing.severity, 0):
                best[key] = finding

    final = list(best.values())
    logger.info("aggregate_findings: %d raw → %d after dedup", len(all_findings), len(final))
    return {"final_findings": final}


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

# Instantiate agents once at module level — they are stateless and reusable.
_security_agent = SecurityAgent()
_performance_agent = PerformanceAgent()
_logic_agent = LogicAgent()
_style_agent = StyleAgent()
_docs_agent = DocsAgent()

_builder = StateGraph(AgentState)

# Register nodes
_builder.add_node("security", _security_agent)
_builder.add_node("performance", _performance_agent)
_builder.add_node("logic", _logic_agent)
_builder.add_node("style", _style_agent)
_builder.add_node("docs", _docs_agent)
_builder.add_node("aggregate", aggregate_findings)

# Fan-out: START → all agents in parallel
_builder.add_edge(START, "security")
_builder.add_edge(START, "performance")
_builder.add_edge(START, "logic")
_builder.add_edge(START, "style")
_builder.add_edge(START, "docs")

# Fan-in: all agents → aggregate
_builder.add_edge("security", "aggregate")
_builder.add_edge("performance", "aggregate")
_builder.add_edge("logic", "aggregate")
_builder.add_edge("style", "aggregate")
_builder.add_edge("docs", "aggregate")

# Terminate after aggregation
_builder.add_edge("aggregate", END)

# Compiled graph — exported for use by the wiring layer and tests.
review_graph = _builder.compile()


# ---------------------------------------------------------------------------
# Public async entry point
# ---------------------------------------------------------------------------

async def run_agent_review(
    hunks: list[DiffHunk],
    model: str,
    settings: "Settings",
    codebase_context: str = "",
) -> list[Finding]:
    """Run the full multi-agent review pipeline on a set of diff hunks.

    Builds the initial AgentState, invokes the compiled graph asynchronously,
    and returns the final deduplicated findings list.

    Args:
        hunks:    The parsed diff hunks to review.
        model:    The litellm model string (e.g. 'gemini/gemini-3.1-flash-lite').
        settings: The project Settings object (used for future filtering hooks).
        codebase_context: Optional context from semantic search to enrich reviews.

    Returns:
        A deduplicated list of Finding objects from all five agents combined.
    """
    initial_state = AgentState(
        hunks=hunks,
        model=model,
        context_lines_count=5,
        codebase_context=codebase_context,
        security_findings=[],
        performance_findings=[],
        logic_findings=[],
        style_findings=[],
        docs_findings=[],
        final_findings=[],
        error=None,
    )

    result = await review_graph.ainvoke(initial_state)

    if result.get("error"):
        logger.warning("Agent pipeline completed with error: %s", result["error"])

    return result.get("final_findings", [])
