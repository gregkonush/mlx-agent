# Repository Guidelines

## Project Structure & Module Organization
- `mlx_dev_agent/`: CLI entrypoint (`cli.py`), runtime engine (`engine.py`), conversation, and config modules; extend features beside the owner component.
- `models/`: local MLX checkpoints referenced by config; add or move weights only after review.
- `pyproject.toml`: dependency, Ruff, and Pyright configuration; edit this when updating tooling.
- Place tests in `tests/` mirroring package layout with shared fixtures in `tests/conftest.py`.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate` creates an isolated interpreter.
- `pip install -e .[dev]` installs the editable package plus lint/test dependencies.
- `mlx-agent chat --help` confirms Typer wiring and lists subcommands.
- `ruff check mlx_dev_agent tests` enforces style and sorted imports; `--fix` applies safe changes.
- `pytest` (optionally `-k pattern`) runs async tests.
- `pyright` detects typing regressions; treat failures as release blockers.

## Coding Style & Naming Conventions
- Python 3.10+, four-space indentation, type hints on public APIs.
- Modules and files snake_case, classes PascalCase, Typer options kebab-case (`--max-tokens`).
- Respect Ruff's 100-character line length and rule set; avoid unused imports and implicit re-exports.
- Prefer dataclasses or TypedDicts to document shared state.

## Testing Guidelines
- Name files `test_<module>.py` and mark async coroutines with `pytest.mark.asyncio`.
- Assert both success and failure paths for conversation flow, streaming, and config parsing.
- Stub MLX model calls and rely on lightweight fixtures instead of loading checkpoints.
- Add a regression test for every bug fix and explain tricky behavior in docstrings.

## Commit & Pull Request Guidelines
- Use imperative subject lines ≤72 chars; Conventional Commits (`feat:`, `fix:`, `refactor:`) keep history legible.
- Commits must pass `ruff`, `pytest`, and `pyright`; reference issue IDs in the footer when relevant.
- PRs include a change summary, verification notes, and mention config or model impacts; add screenshots for CLI UX updates.
- Request review from owners of touched modules (`engine`, `cli`, `conversation`) and list follow-up items.

## Model & Agent Configuration
- `config.py` seeds defaults for `~/.config/mlx-agent/config.toml`; document schema changes with migration steps.
- Validate new model IDs by ensuring a directory in `models/` and noting expected VRAM usage in docs.
- Store secrets in environment variables or user config files; never commit tokens or private endpoints.
