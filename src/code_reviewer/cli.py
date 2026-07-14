"""Typer CLI entrypoint for the AI Code Reviewer."""

import time
import typer
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich import box
import subprocess
from collections import defaultdict
from code_reviewer.config import load_settings
from code_reviewer.core.llm_client import LLMClient, LLMClientError
from code_reviewer.core.reviewer import FileReviewer, DiffReviewer
from code_reviewer.analyzers.diff_parser import parse_diff

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

# Setup Typer application and sub-commands
app = typer.Typer(name="code-reviewer", help="AI Code Reviewer CLI")
review_app = typer.Typer(help="Review code changes")
app.add_typer(review_app, name="review")

console = Console()


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
    # 1. Load settings from config file or env variables
    settings = load_settings()

    # 2. Apply command line overrides
    if severity:
        sev_upper = severity.upper()
        if sev_upper not in ["HIGH", "MEDIUM", "LOW", "INFO"]:
            console.print(f"[red]Error: Invalid severity '{severity}'. Must be HIGH, MEDIUM, LOW, or INFO.[/red]")
            raise typer.Exit(code=1)
        settings.severity_threshold = sev_upper

    if output:
        out_lower = output.lower()
        if out_lower not in ["pretty", "json", "github"]:
            console.print(f"[red]Error: Invalid output format '{output}'. Must be pretty, json, or github.[/red]")
            raise typer.Exit(code=1)
        settings.output.format = out_lower

    # 3. Instantiate client and reviewer
    llm_client = LLMClient(model=settings.model, max_tokens=settings.max_tokens)
    reviewer = FileReviewer(llm_client=llm_client, settings=settings)

    # 4. Perform code review
    start_time = time.time()
    try:
        with console.status("[bold green]Analyzing file..."):
            result = reviewer.review_file(file_path)
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

    elapsed = time.time() - start_time

    # 5. Output results
    format_type = settings.output.format

    if format_type == "json":
        # Print plain JSON to allow shell redirection without rich tags
        print(result.model_dump_json(indent=2))
    elif format_type == "github":
        for finding in result.findings:
            if finding is None:
                continue
            line_str = f"line={finding.line_number}" if finding.line_number else ""
            github_sev = "error" if finding.severity == "HIGH" else "warning"
            title = _sanitize_annotation(f"{finding.category} ({finding.severity})")
            safe_file = _sanitize_annotation(finding.file_path)
            safe_msg = _sanitize_annotation(finding.message)
            console.print(
                f"::{github_sev} file={safe_file},{line_str},title={title}::"
                f"{safe_msg} -> {finding.suggestion}"
            )
        console.print(f"Reviewed {file_path}: {len(result.findings)} findings.")
    else:  # pretty
        lines = []
        for i, finding in enumerate(result.findings):
            if i > 0:
                lines.append("")

            sev = finding.severity
            sev_color = (
                "red"
                if sev == "HIGH"
                else "yellow"
                if sev == "MEDIUM"
                else "blue"
                if sev == "LOW"
                else "dim"
            )
            line_text = f"Line {finding.line_number}" if finding.line_number else "File Scope"

            lines.append(f"[{sev_color}][{sev}][/{sev_color}]   {line_text} · {finding.category}")
            lines.append(finding.message)
            if settings.output.show_suggestions and finding.suggestion:
                lines.append(f"[green]→ {finding.suggestion}[/green]")

        panel_content = "\n".join(lines)
        panel = Panel(
            panel_content,
            title=file_path,
            title_align="left",
            box=box.SQUARE,
            width=60,
        )
        console.print(panel)
        console.print(f"{len(result.findings)} findings · reviewed in {elapsed:.1f}s · {result.model_used}")


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
    # 1. Run git diff
    cmd = ["git", "diff"]
    if staged:
        cmd.append("--cached")
    elif target:
        cmd.append(target)
    
    try:
        proc_result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        diff_text = proc_result.stdout
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Error running git diff: {e.stderr}[/red]")
        raise typer.Exit(code=1)

    if not diff_text.strip():
        console.print("[yellow]No diff output to review.[/yellow]")
        return
        
    # 2. Parse hunks
    hunks = parse_diff(diff_text)
    if not hunks:
        console.print("[yellow]No valid diff hunks parsed.[/yellow]")
        return
        
    # Group hunks by file
    hunks_by_file = defaultdict(list)
    for hunk in hunks:
        hunks_by_file[hunk.file_path].append(hunk)

    # 3. Load settings
    settings = load_settings()
    
    if severity:
        sev_upper = severity.upper()
        if sev_upper not in ["HIGH", "MEDIUM", "LOW", "INFO"]:
            console.print(f"[red]Error: Invalid severity '{severity}'. Must be HIGH, MEDIUM, LOW, or INFO.[/red]")
            raise typer.Exit(code=1)
        settings.severity_threshold = sev_upper

    if output:
        out_lower = output.lower()
        if out_lower not in ["pretty", "json", "github"]:
            console.print(f"[red]Error: Invalid output format '{output}'. Must be pretty, json, or github.[/red]")
            raise typer.Exit(code=1)
        settings.output.format = out_lower

    # 4. Instantiate client and reviewer
    llm_client = LLMClient(model=settings.model, max_tokens=settings.max_tokens)
    reviewer = DiffReviewer(llm_client=llm_client, settings=settings)

    all_results = []
    start_time = time.time()
    
    # 5. Perform code review
    with console.status("[bold green]Analyzing diffs..."):
        for file_path, file_hunks in hunks_by_file.items():
            try:
                review_result = reviewer.review_hunks(file_path, file_hunks)
                all_results.append(review_result)
            except LLMClientError as e:
                console.print(f"[red]LLM Review Failed for {file_path}: {e}[/red]")
            except Exception as e:
                console.print(f"[red]Unexpected Error for {file_path}: {e}[/red]")
                
    elapsed = time.time() - start_time
    format_type = settings.output.format
    
    total_findings = sum(len(res.findings) for res in all_results)
    
    # 6. Output results
    if format_type == "json":
        import json
        out_list = [res.model_dump() for res in all_results]
        print(json.dumps(out_list, indent=2))
    elif format_type == "github":
        for res in all_results:
            for finding in res.findings:
                if finding is None:
                    continue
                line_str = f"line={finding.line_number}" if finding.line_number else ""
                github_sev = "error" if finding.severity == "HIGH" else "warning"
                title = _sanitize_annotation(f"{finding.category} ({finding.severity})")
                safe_file = _sanitize_annotation(finding.file_path)
                safe_msg = _sanitize_annotation(finding.message)
                console.print(
                    f"::{github_sev} file={safe_file},{line_str},title={title}::"
                    f"{safe_msg} -> {finding.suggestion}"
                )
        console.print(f"Reviewed {len(all_results)} files: {total_findings} findings.")
    else:  # pretty
        for res in all_results:
            if not res.findings:
                continue
                
            lines = []
            for i, finding in enumerate(res.findings):
                if i > 0:
                    lines.append("")

                sev = finding.severity
                sev_color = (
                    "red" if sev == "HIGH"
                    else "yellow" if sev == "MEDIUM"
                    else "blue" if sev == "LOW"
                    else "dim"
                )
                line_text = f"Line {finding.line_number}" if finding.line_number else "File Scope"

                lines.append(f"[{sev_color}][{sev}][/{sev_color}]   {line_text} · {finding.category}")
                lines.append(finding.message)
                if settings.output.show_suggestions and finding.suggestion:
                    lines.append(f"[green]→ {finding.suggestion}[/green]")

            panel_content = "\n".join(lines)
            panel = Panel(
                panel_content,
                title=res.file_path,
                title_align="left",
                box=box.SQUARE,
                width=60,
            )
            console.print(panel)
            
        console.print(f"{total_findings} findings · reviewed in {elapsed:.1f}s · {settings.model}")


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
