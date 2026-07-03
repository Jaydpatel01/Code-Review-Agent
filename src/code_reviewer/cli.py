"""Typer CLI entrypoint for the AI Code Reviewer."""

import time
import typer
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich import box
from code_reviewer.config import load_settings
from code_reviewer.core.llm_client import LLMClient, LLMClientError
from code_reviewer.core.reviewer import FileReviewer

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
            line_str = f"line={finding.line_number}" if finding.line_number else ""
            github_sev = "error" if finding.severity == "HIGH" else "warning"
            title = f"{finding.category} ({finding.severity})"
            console.print(
                f"::{github_sev} file={finding.file_path},{line_str},title={title}::"
                f"{finding.message} -> {finding.suggestion}"
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


if __name__ == "__main__":
    app()
