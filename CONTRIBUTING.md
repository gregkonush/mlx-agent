# Contributing

Thanks for your interest in improving **MLX Dev Agent**! This project is designed to run locally on macOS/Apple Silicon, and we happily welcome bug reports, feature ideas, and pull requests.

## Quick start

```bash
# Clone and install in editable mode
pip install -e .[dev]

# Install tooling hooks (optional but recommended)
pre-commit install
```

Make sure you have the Hugging Face CLI (`hf`) available if you plan to exercise model-loading paths.

## Development workflow

1. **Linting** – run Ruff to keep imports and style tidy:
   ```bash
   ruff check mlx_dev_agent
   ```
2. **Static type checks** – Pyright validates type hints:
   ```bash
   pyright
   ```
3. **Tests** – the pytest suite runs quickly without pulling large models, thanks to lightweight stubs:
   ```bash
   python -m pytest
   ```
4. **Pre-commit** – the repo ships `.pre-commit-config.yaml` so formatting and type checks can run automatically before each commit:
   ```bash
   pre-commit run --all-files
   ```

## Pull request guidelines

- Create a feature branch off `main` for each change set.
- Keep pull requests focused and include context in the description (what changed and why).
- Update or add tests when you change behavior.
- Update documentation (`README.md`, `CHANGELOG.md`) when you add or remove user-visible features.

## Reporting bugs or feature requests

Use GitHub Issues for discussion. Include:
- OS version and Python version (`python --version`).
- CLI invocation or configuration snippet.
- Relevant logs or stack traces.

Thank you for helping build a fast, fully local development agent!
