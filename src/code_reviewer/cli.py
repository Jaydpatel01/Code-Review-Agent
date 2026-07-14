"""Typer CLI entrypoint for the AI Code Reviewer."""

import json
import re
import subprocess
import time
from collections import defaultdict
from typing import Optional

import typer
from rich import box
from rich.console import Console
from rich.panel import Panel

from code_reviewer.config import Settings, load_settings
from code_reviewer.core.llm_client import LLMClient, LLMClientError
from code_reviewer.core.models import Finding, ReviewResult
from code_reviewer.core.reviewer import DiffReviewer, FileReviewer
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


def _run_diff_review(diff_text: str, settings: Settings) -> list[ReviewResult]:
    """Parse the diff text, group hunks by file, and run DiffReviewer on each.

    Returns a list of ReviewResult objects (one per file). Errors per file are
    printed but do not stop review of remaining files.
    """
    hunks = parse_diff(diff_text)
    if not hunks:
        console.print("[yellow]No valid diff hunks parsed.[/yellow]")
        return []

    hunks_by_file = defaultdict(list)
    for hunk in hunks:
        hunks_by_file[hunk.file_path].append(hunk)

    llm_client = LLMClient(model=settings.model, max_tokens=settings.max_tokens)
    reviewer = DiffReviewer(llm_client=llm_client, settings=settings)

    results: list[ReviewResult] = []
    with console.status("[bold green]Analyzing diffs..."):
        for file_path, file_hunks in hunks_by_file.items():
            try:
                results.append(reviewer.review_hunks(file_path, file_hunks))
            except LLMClientError as e:
                console.print(f"[red]LLM Review Failed for {file_path}: {e}[/red]")
            except Exception as e:
                console.print(f"[red]Unexpected Error for {file_path}: {e}[/red]")
    return results


def _output_diff_results(
    results: list[ReviewResult],
    output_format: str,
    settings: Settings,
    elapsed: float,
) -> None:
    """Dispatch a list of diff ReviewResults to the correct output formatter."""
    total = sum(len(r.findings) for r in results)

    if output_format == "json":
        print(json.dumps([r.model_dump(mode="json") for r in results], indent=2))
        return

    if output_format == "github":
        for result in results:
            _print_github_findings([f for f in result.findings if f is not None])
        console.print(f"Reviewed {len(results)} files: {total} findings.")
        return

    # pretty
    for result in results:
        findings = [f for f in result.findings if f is not None]
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
