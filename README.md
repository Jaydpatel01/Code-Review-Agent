# AI Code Reviewer (CodeRabbit-Comparable)

A production-grade AI-powered code review agent built with Python, Pydantic, and LiteLLM. This project is built incrementally, starting from a CLI-based single file reviewer up to a multi-agent, codebase-aware PR bot.

---

## Features
- **Git Diff Awareness**: Reviews only changed lines in a git diff (staged changes, branch comparisons, etc.), drastically reducing noise and token usage.
- **Local File Review**: Instantly reviews single source files for security, performance, complexity, logic, docstrings, and style.
- **LiteLLM Abstraction**: Integrates with swappable LLM providers (defaulting to Gemini 3.1 Flash Lite) via a unified interface with built-in retry logic.
- **Configurable Rules**: Uses a project-level `.codereviewer.yaml` config file for customizing severity thresholds and enabling/disabling specific rules.
- **Flexible Outputs**: Supports console pretty-printing (complete with Unicode borders and color coding via `rich`), JSON reports, and GitHub Action annotation formats.

---

## Setup & Installation

This project utilizes `uv` for python environment and dependency management.

1. **Clone the repository**:
   ```bash
   git clone <repo-url>
   cd ai-code-reviewer
   ```

2. **Sync the virtual environment and install packages**:
   ```bash
   uv sync
   ```

3. **Configure API Keys**:
   Create a `.env` file in the root directory:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

---

## Configuration (`.codereviewer.yaml`)

You can create a `.codereviewer.yaml` file in the root of your project:

```yaml
model: gemini/gemini-3.1-flash-lite
max_tokens: 2048
severity_threshold: MEDIUM    # HIGH | MEDIUM | LOW | INFO

rules:
  complexity:
    enabled: true
    max_cyclomatic_complexity: 10
    max_function_length: 50
  nesting:
    enabled: true
    max_nesting_depth: 4
  mutable_defaults:
    enabled: true
  magic_numbers:
    enabled: false
  security:
    enabled: true
  style:
    enabled: true
  docs:
    enabled: false            # Off by default

output:
  format: pretty              # pretty | json | github
  show_suggestions: true
```

---

## Static Analysis (AST & Tree-sitter)

Before sending code to the LLM, the reviewer runs deterministic static analysis. This ensures fast, zero-cost, hallucination-free feedback for structural issues. 

For **Python**, it uses the built-in `ast` module. For **JavaScript and Java**, it uses pre-compiled `tree-sitter` binaries.

### Supported Checks
1. **Cyclomatic Complexity**: > 10 (MEDIUM), > 15 (HIGH).
2. **Function Length**: > 50 lines (MEDIUM).
3. **Nesting Depth**: > 4 levels (MEDIUM), > 6 levels (HIGH).
4. **Mutable Default Arguments**: (Python only) High severity logic flaw.
5. **Missing Docstrings**: Checks public functions and classes (LOW).
6. **Magic Numbers**: Numeric literals without a named constant (INFO).

### Example Output (`source` attribute)
When you run the reviewer, findings will indicate where they came from. Static checks take precedence—if the LLM and AST both flag the same line for complexity, the LLM finding is discarded.

```json
{
  "file_path": "src/app.py",
  "line_number": 10,
  "severity": "HIGH",
  "category": "complexity",
  "message": "Extremely deep nesting (7 levels).",
  "suggestion": "Extract nested blocks into separate functions.",
  "source": "ast"
}
```
```json
{
  "file_path": "src/app.py",
  "line_number": 12,
  "severity": "HIGH",
  "category": "security",
  "message": "Hardcoded AWS Access Key.",
  "suggestion": "Use an environment variable.",
  "source": "llm"
}
```

---

## Multi-Agent Architecture

When reviewing git diffs, the agent spawns five parallel specialists via [LangGraph](https://github.com/langchain-ai/langgraph):

| Agent | Domain |
|-------|--------|
| **SecurityAgent** | Injection, hardcoded secrets, path traversal, XSS, eval/exec, weak crypto, missing auth |
| **PerformanceAgent** | N+1 queries, O(n²) algorithms, missing memoization, sync I/O in async context |
| **LogicAgent** | Null checks, off-by-one errors, is vs ==, silent exception swallowing, edge cases |
| **StyleAgent** | Misleading names, single-letter vars, SRP violations, dead/commented-out code |
| **DocsAgent** | Contradictory comments, untracked TODOs, lying function names |

All five agents run in **parallel** (LangGraph fan-out). Results are **deduplicated** — if two agents flag the same `(file, line, category)` key, the higher-severity finding wins. Static AST analysis still runs first and takes precedence over all LLM findings on the same line+category.

```
 START ──┬──► SecurityAgent     ──┐
         ├──► PerformanceAgent  ──┤
         ├──► LogicAgent        ──┼──► aggregate ──► END
         ├──► StyleAgent        ──┤
         └──► DocsAgent         ──┘
```

Each agent:
1. Receives only the **added lines** (`+`) and their surrounding context — never the removed lines.
2. Returns structured JSON: `{"findings": [{"line_number": int, "severity": ..., ...}]}`
3. Has findings on non-added lines silently rejected (hallucination guard).

---

## Usage

### CLI Commands

Run the reviewer using `uv run`:

#### Reviewing Git Diffs (Recommended)

Review only the lines that have changed (diffs) rather than entire files.

* **Review unstaged changes against HEAD**:
  ```bash
  uv run code-reviewer review diff
  ```

* **Review staged changes**:
  ```bash
  uv run code-reviewer review diff --staged
  ```

* **Review against a specific commit or branch range**:
  ```bash
  uv run code-reviewer review diff HEAD~1
  uv run code-reviewer review diff main..feature-branch
  ```

#### Reviewing Single Files

* **Standard pretty-printed review**:
  ```bash
  uv run code-reviewer review file src/code_reviewer/config.py
  ```

* **Filter by severity override**:
  ```bash
  uv run code-reviewer review file src/code_reviewer/config.py --severity HIGH
  ```

* **Output as JSON**:
  ```bash
  uv run code-reviewer review file src/code_reviewer/config.py --output json
  ```

* **Output as GitHub Action Annotation**:
  ```bash
  uv run code-reviewer review file src/code_reviewer/config.py --output github
  ```

#### Reviewing Repositories

Review an entire codebase or directory recursively, with smart static filtering.

* **Smart Mode (Default)**:
  Runs AST analysis on all files, then runs LLM review only on files classified as `HIGH` or `MEDIUM` risk:
  ```bash
  uv run code-reviewer review repo . --mode smart
  ```

* **Thorough Mode**:
  Runs AST analysis and LLM review on all matching files:
  ```bash
  uv run code-reviewer review repo src/ --mode thorough
  ```

* **Static-Only Mode (Free & Instant)**:
  Runs AST analysis only on all matching files without any LLM/API calls:
  ```bash
  uv run code-reviewer review repo . --mode static-only
  ```

* **Custom Include/Exclude and Safety Limits**:
  ```bash
  uv run code-reviewer review repo . --include "*.py,*.js" --exclude "tests,build" --max-files 100
  ```

---

## Development & Testing

Run the test suite with coverage reporting:
```bash
uv run pytest --cov=src
```

---

## GitHub PR Bot Setup

The reviewer can run as a GitHub webhook bot that automatically reviews pull requests and posts inline comments.

### 1. Authentication — Personal Access Token (PAT)

Create a fine-grained PAT at **GitHub → Settings → Developer settings → Personal access tokens** with:
- **Contents**: Read-only
- **Pull requests**: Read and Write

Set it as a repository secret named `GITHUB_TOKEN` (or use a separate `REVIEWER_TOKEN` to avoid conflicts with the built-in Actions token).

### 2. Configure the Webhook

In your target repository go to **Settings → Webhooks → Add webhook**:

| Field | Value |
|---|---|
| Payload URL | `https://<your-server>/webhook` |
| Content type | `application/json` |
| Secret | A strong random string (save it — you'll need it next) |
| Events | ✅ Pull requests |

### 3. Set Server Environment Variables

On your server (or in your deployment config):

```env
WEBHOOK_SECRET=<the secret from step 2>
GITHUB_TOKEN=<your PAT from step 1>
GEMINI_API_KEY=<your LLM key>
```

### 4. Start the Server

```bash
uv run code-reviewer serve --host 0.0.0.0 --port 8000
```

Or with auto-reload during development:
```bash
uv run code-reviewer serve --reload
```

### 5. GitHub Actions — Automatic Self-Review

The included [pr_review.yml](.github/workflows/pr_review.yml) workflow runs the reviewer against every PR to `main` using the `--output github` format (workflow annotations).

Add these secrets to your repository (**Settings → Secrets → Actions**):
- `GEMINI_API_KEY` or `ANTHROPIC_API_KEY`

The workflow uses:
```
${{ github.event.pull_request.base.sha }}..${{ github.event.pull_request.head.sha }}
```
to review exactly the changed commits, not the entire branch history.
