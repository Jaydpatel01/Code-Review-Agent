"""Typer CLI entrypoint for the AI Code Reviewer."""

import asyncio
import json
import re
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer
from rich import box
from rich.console import Console
from rich.panel import Panel

from code_reviewer.config import Settings, load_settings
from code_reviewer.core.llm_client import LLMClient, LLMClientError
from code_reviewer.core.models import Finding, ReviewResult
from code_reviewer.core.reviewer import DiffReviewer, FileReviewer, combine_findings
from code_reviewer.analyzers.diff_parser import parse_diff

# ---------------------------------------------------------------------------
# Severity ordering used for threshold filtering
# ---------------------------------------------------------------------------
_SEVERITY_ORDER: dict[str, int] = {"HIGH": 3, "MEDIUM": 2, "LOW": 1, "INFO": 0}
_VALID_SEVERITIES = list(_SEVERITY_ORDER.keys())
_VALID_FORMATS = ["pretty", "json", "github"]

# ---------------------------------------------------------------------------
# Git ref validation — allowlist pattern to prevent injection via target arg
# ---------------------------------------------------------------------------
_VALID_GIT_REF = re.compile(
    r'^[a-zA-Z0-9._/~^:@{}\[\]\\-]+(\.\.[\ a-zA-Z0-9._/~^:@{}\[\]\\-]+)?$'
)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = typer.Typer(name="code-reviewer", help="AI Code Reviewer CLI")
review_app = typer.Typer(help="Review code changes")
app.add_typer(review_app, name="review")

console = Console()


# ===========================================================================
# Shared helpers
# ===========================================================================

def _sanitize_annotation(value: str) -> str:
    """Sanitize a string for safe use in GitHub Actions annotation syntax."""
    return (
        value
        .replace('%', '%25')
        .replace('\r', '%0D')
        .replace('\n', '%0A')
        .replace(':', '%3A')
        .replace(',', '%2C')
    )


def _validate_severity(severity: str) -> str:
    """Return the uppercased severity if valid, raise Exit(1) otherwise."""
    sev_upper = severity.upper()
    if sev_upper not in _VALID_SEVERITIES:
        console.print(
            f"[red]Error: Invalid severity '{severity}'. "
            f"Must be one of: {', '.join(_VALID_SEVERITIES)}.[/red]"
        )
        raise typer.Exit(code=1)
    return sev_upper


def _validate_output_format(output: str) -> str:
    """Return the lowercased output format if valid, raise Exit(1) otherwise."""
    out_lower = output.lower()
    if out_lower not in _VALID_FORMATS:
        console.print(
            f"[red]Error: Invalid output format '{output}'. "
            f"Must be one of: {', '.join(_VALID_FORMATS)}.[/red]"
        )
        raise typer.Exit(code=1)
    return out_lower


def _apply_cli_overrides(
    settings: Settings,
    severity: Optional[str],
    output: Optional[str],
) -> Settings:
    """Apply CLI flag overrides to a Settings object and return it.

    Validates severity and output format values; raises Exit(1) on bad input.
    """
    if severity:
        settings.severity_threshold = _validate_severity(severity)
    if output:
        settings.output.format = _validate_output_format(output)
    return settings


def _filter_by_severity(
    findings: list[Finding],
    threshold: str,
) -> list[Finding]:
    """Return only findings at or above the given severity threshold.

    Severity order: HIGH > MEDIUM > LOW > INFO.
    """
    min_rank = _SEVERITY_ORDER.get(threshold.upper(), 0)
    return [f for f in findings if f is not None and _SEVERITY_ORDER.get(f.severity, 0) >= min_rank]


def _severity_to_github_level(severity: str) -> str:
    """Map a Finding severity to a GitHub Actions workflow command level.

    HIGH   → error
    MEDIUM → warning
    LOW    → notice
    INFO   → notice
    """
    if severity == "HIGH":
        return "error"
    if severity == "MEDIUM":
        return "warning"
    return "notice"


def _format_finding_github(finding: Finding) -> str:
    """Render a single Finding as a GitHub Actions annotation line."""
    line_str = f"line={finding.line_number}" if finding.line_number else ""
    level = _severity_to_github_level(finding.severity)
    title = _sanitize_annotation(finding.category)
    safe_file = _sanitize_annotation(finding.file_path)
    msg = _sanitize_annotation(finding.message)
    if finding.suggestion:
        msg += f" {_sanitize_annotation(finding.suggestion)}"
    return f"::{level} file={safe_file},{line_str},title={title}::{msg}"


def _format_finding_pretty_lines(finding: Finding, show_suggestion: bool) -> list[str]:
    """Return rich-formatted lines for a single Finding in pretty mode."""
    sev = finding.severity
    sev_color = (
        "red" if sev == "HIGH"
        else "yellow" if sev == "MEDIUM"
        else "blue" if sev == "LOW"
        else "dim"
    )
    line_text = f"Line {finding.line_number}" if finding.line_number else "File Scope"
    lines: list[str] = [
        f"[{sev_color}][{sev}][/{sev_color}]   {line_text} · {finding.category}",
        finding.message,
    ]
    if show_suggestion and finding.suggestion:
        lines.append(f"[green]→ {finding.suggestion}[/green]")
    return lines


def _print_github_findings(findings: list[Finding]) -> None:
    """Print all non-None findings as GitHub Actions annotation lines."""
    for finding in findings:
        if finding is None:
            continue
        print(_format_finding_github(finding))


def _print_pretty_findings(
    findings: list[Finding],
    title: str,
    show_suggestion: bool,
    footer: str,
) -> None:
    """Print findings as a rich Panel, then a footer line."""
    lines: list[str] = []
    for i, finding in enumerate(findings):
        if finding is None:
            continue
        if i > 0:
            lines.append("")
        lines.extend(_format_finding_pretty_lines(finding, show_suggestion))

    panel = Panel(
        "\n".join(lines),
        title=title,
        title_align="left",
        box=box.SQUARE,
        width=60,
    )
    console.print(panel)
    console.print(footer)


def _output_result(
    result: ReviewResult,
    output_format: str,
    settings: Settings,
    elapsed: float,
) -> None:
    """Dispatch a ReviewResult to the correct output formatter.

    Supported formats: pretty, json, github.
    """
    findings = [f for f in result.findings if f is not None]

    if output_format == "json":
        print(result.model_dump_json(indent=2))
    elif output_format == "github":
        _print_github_findings(findings)
        console.print(f"Reviewed {result.file_path}: {len(findings)} findings.")
    else:  # pretty
        footer = (
            f"{len(findings)} findings · reviewed in {elapsed:.1f}s · {result.model_used}"
        )
        _print_pretty_findings(
            findings,
            title=result.file_path,
            show_suggestion=settings.output.show_suggestions,
            footer=footer,
        )


# ===========================================================================
# review file helpers
# ===========================================================================

def _run_file_review(file_path: str, settings: Settings) -> ReviewResult:
    """Instantiate FileReviewer and run the review on a single file.

    Raises typer.Exit(code=1) on any review error.
    """
    llm_client = LLMClient(model=settings.model, max_tokens=settings.max_tokens)
    reviewer = FileReviewer(llm_client=llm_client, settings=settings)
    try:
        with console.status("[bold green]Analyzing file..."):
            return reviewer.review_file(file_path)
    except FileNotFoundError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1)
    except LLMClientError as e:
        console.print(f"[red]LLM Review Failed: {e}[/red]")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[red]Unexpected Error: {e}[/red]")
        raise typer.Exit(code=1)


# ===========================================================================
# review diff helpers
# ===========================================================================

def _build_git_diff_command(target: Optional[str], staged: bool) -> list[str]:
    """Return the git diff subprocess command as a list of arguments.

    staged=True  → ["git", "diff", "--cached"]
    target set   → ["git", "diff", target]
    neither      → ["git", "diff"]
    """
    cmd = ["git", "diff"]
    if staged:
        cmd.append("--cached")
    elif target:
        cmd.append(target)
    return cmd


def _validate_git_target(target: str) -> bool:
    """Validate a git reference against a strict allowlist pattern.

    Allows: branch names, commit SHAs, HEAD~N, range syntax (a..b).
    Rejects: shell metacharacters, semicolons, pipes, backticks, $(), etc.
    """
    return bool(_VALID_GIT_REF.match(target))


def _run_git_diff(target: Optional[str], staged: bool) -> Optional[str]:
    """Run git diff and return the raw diff text, or None on failure.

    Validates target against the allowlist before passing to subprocess.
    Prints an error message and returns None on invalid ref or CalledProcessError.
    subprocess.run() is always called with a list (shell=False) for safety.
    """
    if target and not staged and not _validate_git_target(target):
        typer.echo(
            f"[ERROR] Invalid git reference: '{target}'. "
            "Only valid git refs and ranges are allowed.",
            err=True,
        )
        return None
    cmd = _build_git_diff_command(target, staged)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return proc.stdout
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Error running git diff: {e.stderr}[/red]")
        return None


def _validate_diff_not_empty(diff_text: str, target: Optional[str]) -> bool:
    """Return False and print a helpful message when the diff is empty.

    Returns True when diff_text contains actual content to review.
    """
    if not diff_text.strip():
        label = target or "current changes"
        console.print(f"[yellow]No diff output for '{label}'. Nothing to review.[/yellow]")
        return False
    return True


async def _review_all_files_async(
    hunks_by_file: dict,
    settings: Settings,
) -> list[ReviewResult]:
    """Run per-file LLM reviews in parallel using asyncio.gather.

    Each file's hunks are sent through the full multi-agent graph
    (run_agent_review) concurrently. Failed files are skipped and
    a warning is printed; they do not block the other files.
    """
    from code_reviewer.agents.graph import run_agent_review

    async def _review_one(
        file_path: str,
        hunks: list,
    ) -> ReviewResult | None:
        """Review a single file's hunks; return None on failure."""
        try:
            findings = await run_agent_review(hunks, settings.model, settings)
            return ReviewResult(
                file_path=file_path,
                findings=findings,
                summary=f"Reviewed {len(hunks)} hunks via multi-agent graph",
                reviewed_at=datetime.now(tz=timezone.utc),
                model_used=settings.model,
                lines_reviewed=sum(len(h.added_lines) for h in hunks),
            )
        except Exception as e:
            typer.echo(f"[WARN] Failed to review {file_path}: {e}", err=True)
            return None

    results = await asyncio.gather(
        *[_review_one(fp, hunks) for fp, hunks in hunks_by_file.items()]
    )
    return [r for r in results if r is not None]


def _run_diff_review(diff_text: str, settings: Settings) -> list[ReviewResult]:
    """Parse the diff text, group hunks by file, and review all files in parallel.

    Uses asyncio.gather to fan-out per-file LLM reviews concurrently via the
    multi-agent graph. Returns a list of ReviewResult objects (one per file).
    Files that fail are skipped; they do not block the other files.
    """
    hunks = parse_diff(diff_text)
    if not hunks:
        console.print("[yellow]No valid diff hunks parsed.[/yellow]")
        return []

    hunks_by_file: dict = defaultdict(list)
    for hunk in hunks:
        hunks_by_file[hunk.file_path].append(hunk)

    return asyncio.run(_review_all_files_async(hunks_by_file, settings))


def _print_diff_json(results: list[ReviewResult], **_: object) -> None:
    """Print all diff ReviewResults as a JSON array."""
    print(json.dumps([r.model_dump(mode="json") for r in results], indent=2))


def _print_diff_github(results: list[ReviewResult], **_: object) -> None:
    """Print all diff ReviewResults as GitHub Actions annotation lines."""
    total = 0
    for result in results:
        findings = [f for f in result.findings if f is not None]
        total += len(findings)
        _print_github_findings(findings)
    console.print(f"Reviewed {len(results)} files: {total} findings.")


def _print_diff_pretty(
    results: list[ReviewResult],
    settings: Settings,
    elapsed: float,
    **_: object,
) -> None:
    """Print all diff ReviewResults as rich Panels (pretty mode)."""
    total = 0
    for result in results:
        findings = [f for f in result.findings if f is not None]
        total += len(findings)
        if not findings:
            continue
        footer = f"{len(findings)} findings · reviewed in {elapsed:.1f}s · {settings.model}"
        _print_pretty_findings(
            findings,
            title=result.file_path,
            show_suggestion=settings.output.show_suggestions,
            footer=footer,
        )
    console.print(f"Total: {total} findings in {len(results)} files.")


_DIFF_FORMATTERS = {
    "json":   _print_diff_json,
    "github": _print_diff_github,
    "pretty": _print_diff_pretty,
}


def _output_diff_results(
    results: list[ReviewResult],
    output_format: str,
    settings: Settings,
    elapsed: float,
) -> None:
    """Dispatch a list of diff ReviewResults to the correct output formatter.

    Uses a dispatch dictionary to select the formatter by name, keeping CC < 5.
    Raises typer.Exit(1) if an unknown format is requested.
    """
    formatter = _DIFF_FORMATTERS.get(output_format)
    if formatter is None:
        typer.echo(f"[ERROR] Unknown output format: '{output_format}'", err=True)
        raise typer.Exit(code=1)
    formatter(results, settings=settings, elapsed=elapsed)


# ===========================================================================
# CLI commands
# ===========================================================================

@review_app.command("file")
def review_file_cmd(
    file_path: str = typer.Argument(..., help="Path to the file to review"),
    severity: Optional[str] = typer.Option(
        None, "--severity", "-s", help="Severity threshold override (HIGH|MEDIUM|LOW|INFO)"
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output format override (pretty|json|github)"
    ),
):
    """Review a single file and output findings."""
    settings = _apply_cli_overrides(load_settings(), severity, output)
    start_time = time.time()
    result = _run_file_review(file_path, settings)
    elapsed = time.time() - start_time
    _output_result(result, settings.output.format, settings, elapsed)


@review_app.command("diff")
def review_diff_cmd(
    target: Optional[str] = typer.Argument(
        "HEAD", help="The git target to diff against (e.g., HEAD~1, main..feature-branch)"
    ),
    staged: bool = typer.Option(
        False, "--staged", help="Review staged changes (runs git diff --cached)"
    ),
    severity: Optional[str] = typer.Option(
        None, "--severity", "-s", help="Severity threshold override (HIGH|MEDIUM|LOW|INFO)"
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output format override (pretty|json|github)"
    ),
):
    """Review git diff changes and output findings."""
    settings = _apply_cli_overrides(load_settings(), severity, output)

    diff_text = _run_git_diff(target, staged)
    if diff_text is None:
        raise typer.Exit(code=1)
    if not _validate_diff_not_empty(diff_text, target):
        raise typer.Exit(code=0)

    start_time = time.time()
    results = _run_diff_review(diff_text, settings)
    elapsed = time.time() - start_time

    if not results:
        typer.echo("No files could be reviewed. Exiting.", err=True)
        raise typer.Exit(code=0)

    _output_diff_results(results, settings.output.format, settings, elapsed)


# ===========================================================================
# review repo helpers
# ===========================================================================

_VALID_REPO_MODES = ["smart", "thorough", "static-only"]


def _run_repo_review(
    root: "Path",
    mode: str,
    include_patterns: list[str],
    exclude_patterns: list[str],
    settings: Settings,
    max_files: int,
) -> dict[str, list[Finding]]:
    """Core logic for repo-level review across all three modes.

    Returns a mapping of file_path → findings (AST-only in static-only mode;
    AST + LLM in smart/thorough).  On KeyboardInterrupt, prints partial results
    collected so far and returns them instead of raising.
    """
    from code_reviewer.indexer.file_walker import FileWalker
    from code_reviewer.indexer.risk_scorer import score_file_risk
    from pathlib import Path as _Path

    walker = FileWalker(root, include=include_patterns, exclude=exclude_patterns, max_files=max_files)
    total = walker.count()
    typer.echo(f"Found {total} files to review.")
    if max_files > 0 and total > max_files:
        typer.echo(
            f"[WARN] Reviewing first {max_files} only. "
            f"Use --max-files 0 for all {total}."
        )

    # Step 2: Walk + AST score (all modes)
    all_findings: dict[str, list[Finding]] = {}
    risk_tiers: dict[str, str] = {}

    try:
        for i, file_path in enumerate(walker.walk(), 1):
            typer.echo(f"  [{i}] {file_path} ", nl=False)
            tier, findings = score_file_risk(file_path, settings)
            risk_tiers[str(file_path)] = tier
            all_findings[str(file_path)] = findings
            tier_color = "HIGH" if tier == "HIGH" else ("MEDIUM" if tier == "MEDIUM" else "LOW")
            typer.echo(tier_color)
    except KeyboardInterrupt:
        typer.echo("\n[Interrupted] Returning partial AST results.", err=True)
        return all_findings

    # Step 3: LLM review (smart + thorough only)
    if mode == "static-only":
        return all_findings

    if mode == "smart":
        high_medium = [fp for fp, tier in risk_tiers.items() if tier in ("HIGH", "MEDIUM")]
        skipped = [fp for fp, tier in risk_tiers.items() if tier == "LOW"]
        typer.echo(
            f"\nRunning LLM review on {len(high_medium)} HIGH/MEDIUM risk files "
            f"(skipping {len(skipped)} LOW files)..."
        )
        target_files = high_medium
    else:  # thorough
        typer.echo(f"\nRunning LLM review on all {len(all_findings)} files...")
        target_files = list(all_findings.keys())

    try:
        for fp in target_files:
            typer.echo(f"  LLM -> {fp}...")
            try:
                llm_result = _run_file_review(str(fp), settings)
                all_findings[fp] = combine_findings(
                    all_findings[fp],
                    llm_result.findings,
                )
            except Exception as e:
                typer.echo(f"  [WARN] LLM failed for {fp}: {e}", err=True)
    except KeyboardInterrupt:
        typer.echo("\n[Interrupted] Returning partial LLM results.", err=True)

    return all_findings


# ===========================================================================
# review repo CLI command
# ===========================================================================

@review_app.command("repo")
def review_repo_cmd(
    path: str = typer.Argument(".", help="Directory to review (default: current directory)"),
    mode: str = typer.Option(
        "smart",
        "--mode",
        help="Review mode: smart | thorough | static-only",
    ),
    include: str = typer.Option(
        "*.py",
        "--include",
        help="Comma-separated glob patterns to include (e.g. '*.py,*.js')",
    ),
    exclude: str = typer.Option(
        "",
        "--exclude",
        help="Comma-separated directory names to skip (merged with defaults)",
    ),
    severity: Optional[str] = typer.Option(
        None, "--severity", "-s",
        help="Severity threshold override (HIGH|MEDIUM|LOW|INFO)",
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o",
        help="Output format override (pretty|json|github)",
    ),
    max_files: int = typer.Option(
        50,
        "--max-files",
        help="Max files to review (0 = unlimited)",
    ),
):
    """Review all source files in a directory (Phase 6 Layer 1).

    Three modes:
      static-only  AST analysis only — instant, zero API calls.
      smart        AST on all files, LLM only on HIGH/MEDIUM risk files.
      thorough     AST + LLM on every file.
    """
    from pathlib import Path

    # Validate mode
    mode = mode.lower()
    if mode not in _VALID_REPO_MODES:
        typer.echo(
            f"[ERROR] Invalid mode '{mode}'. Must be one of: {', '.join(_VALID_REPO_MODES)}",
            err=True,
        )
        raise typer.Exit(code=1)

    settings = _apply_cli_overrides(load_settings(), severity, output)

    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        typer.echo(f"[ERROR] '{path}' is not a directory.", err=True)
        raise typer.Exit(code=1)

    include_patterns = [p.strip() for p in include.split(",") if p.strip()]
    exclude_patterns: list[str] = [p.strip() for p in exclude.split(",") if p.strip()]

    typer.echo(f"Reviewing [{mode.upper()}] {root}")
    start = time.perf_counter()

    all_findings = _run_repo_review(
        root, mode, include_patterns, exclude_patterns, settings, max_files
    )

    elapsed = time.perf_counter() - start

    # Step 4: Filter and output
    threshold = settings.severity_threshold
    output_format = settings.output.format

    high_count = medium_count = low_count = 0
    total_findings = 0
    files_with_findings = 0

    if output_format == "json":
        import json as _json
        out: list[dict] = []
        for fp, findings in all_findings.items():
            filtered = _filter_by_severity(findings, threshold)
            if filtered:
                out.append({"file": fp, "findings": [f.model_dump(mode="json") for f in filtered]})
        print(_json.dumps(out, indent=2))
    else:
        for fp, findings in all_findings.items():
            filtered = _filter_by_severity(findings, threshold)
            if not filtered:
                continue
            files_with_findings += 1
            total_findings += len(filtered)
            high_count += sum(1 for f in filtered if f.severity == "HIGH")
            medium_count += sum(1 for f in filtered if f.severity == "MEDIUM")
            low_count += sum(1 for f in filtered if f.severity in ("LOW", "INFO"))

            if output_format == "github":
                _print_github_findings(filtered)
            else:  # pretty
                footer = f"{len(filtered)} findings | {settings.model}"
                _print_pretty_findings(
                    filtered,
                    title=fp,
                    show_suggestion=settings.output.show_suggestions,
                    footer=footer,
                )

    typer.echo(
        f"\nReviewed {len(all_findings)} files | {total_findings} findings "
        f"({high_count} high, {medium_count} medium, {low_count} low) | "
        f"{elapsed:.1f}s"
    )


@app.command("serve")
def serve_cmd(
    host: str = typer.Option("0.0.0.0", "--host", help="Host to bind to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to listen on"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload for development"),
):
    """Start the FastAPI webhook server for GitHub PR bot mode."""
    try:
        import uvicorn
    except ImportError:
        console.print("[red]Error: uvicorn is not installed. Run: uv sync[/red]")
        raise typer.Exit(code=1)

    console.print(
        f"[bold green]Starting AI Code Reviewer webhook server[/bold green] "
        f"on [cyan]http://{host}:{port}[/cyan]"
    )
    uvicorn.run(
        "code_reviewer.api.main:app",
        host=host,
        port=port,
        reload=reload,
    )


if __name__ == "__main__":
    app()
