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
   uv sync --extra dev
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
    max_nesting_depth: 4
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

---

## Development & Testing

Run the test suite with coverage reporting:
```bash
uv run pytest --cov=src
```
