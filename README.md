<p align="center">
  <img src="assets/social-card.png" alt="MLX Dev Agent" width="420" />
</p>

# MLX Dev Agent

A zero-latency terminal agent that keeps everything on-device by running large language models with Apple’s MLX framework. It is designed for developers who want a fast “copilot” without handing requests to remote services.

## Why MLX

- MLX is Apple’s array and machine learning framework tuned for Apple silicon, providing memory-efficient primitives and vectorized kernels that keep inference on the GPU/ANE without context switches ([Apple MLX](https://github.com/ml-explore/mlx)).
- The `mlx-lm` tooling ships optimized, quantized language models (e.g., 4-bit Phi-3, Mistral variants) plus a Python API for `stream_generate`, so you can decode tokens with minimal overhead straight from the CLI ([mlx-examples](https://github.com/ml-explore/mlx-examples/tree/main/llms)).

## Installation

```bash
pip install -e .
# Install MLX + MLX-LM wheels built for Apple silicon
pip install mlx-lm
# Optional: install the Hugging Face CLI for authenticated downloads
pip install "huggingface_hub[cli]"
```

> The project targets macOS 14+ on Apple Silicon. You need the MLX runtime from [Apple’s mlx project](https://github.com/ml-explore/mlx) for GPU/ANE acceleration.

## Quickstart

```bash
# Generate a default config in ~/.config/mlx-agent/config.toml
mlx-agent init-config

# Chat interactively with default Qwen3 4-bit weights
mlx-agent chat

# One-shot question without entering the REPL
mlx-agent chat "How do I stream results from MLX inside Python?"

# Toggle ANSI colors if your terminal supports them
mlx-agent chat --color "Review main.py"
```

### Downloading a model the first time

If the `models/` directory is empty, the CLI falls back to `mlx-community/Qwen3-8B-MLX-4bit` on Hugging Face. That repository requires accepting the license, so you must authenticate before the initial download.

1. Create an access token on <https://huggingface.co/settings/tokens>.
2. Log in from the terminal (the CLI installs an `hf` executable):

   ```bash
   hf login
   ```

3. Pull the model into a local directory (you can choose any destination):

   ```bash
   hf download mlx-community/Qwen3-8B-MLX-4bit \
     --local-dir models/Qwen3-8B-MLX-4bit
   ```

4. Point the CLI at the downloaded weights:

   ```bash
   mlx-agent chat --model models/Qwen3-8B-MLX-4bit "hello"
   ```

For other models, accept the license on Hugging Face first, then substitute the new repo ID in the commands above. The `.gitignore` already excludes `models/` so large weights never end up in commits.

Key REPL shortcuts:

- `:help` – show command list
- `:clear` – drop conversation history (keeps the system prompt)
- `:system <prompt>` – swap the system persona on the fly
- `:load <path>` – bring a file’s contents into context (auto-truncates to keep fast decoding)
- `!<command>` – execute a shell command and feed stdout back into the model as fresh context

For single prompts you can also pass files up front:

```bash
mlx-agent chat --context src/main.py --context docs/plan.md "Review the diff above."
```

## Configuration

The CLI merges CLI flags with TOML configuration. A generated config looks like:

```toml
model = "mlx-community/Qwen3-8B-MLX-4bit"
max_context_tokens = 8192
preload = true

[generation]
max_tokens = 1024
temperature = 0.2
top_p = 0.9
repetition_penalty = 1.05
```

Adjust the `model` field to point at any MLX-compatible repository (local path or Hugging Face ID). Quantized weights finish loading in seconds and keep VRAM usage low enough for interactive workflows ([mlx-examples docs](https://github.com/ml-explore/mlx-examples/tree/main/llms)).

## Performance-oriented defaults

- **Warmup compilation** – We issue a one-token warmup pass the first time you launch so MLX can JIT kernels and hit peak throughput immediately on subsequent prompts.
- **Streaming render** – Tokens arrive as soon as MLX emits them, making the CLI feel instantaneous compared with remote agents that wait on network round-trips.
- **Context trimming** – The conversation automatically prunes the oldest turns to stay under the configured token budget, avoiding runaway latency.

## Developer workflows

- Preload large files or recent command output via `:load` / `!<cmd>` and drill into errors without leaving the terminal.
- Persist transcripts to JSON with `--save-transcript` so you can diff prior sessions or feed them into unit tests.
- Use different system prompts per project: `mlx-agent chat --system-prompt "You are a code reviewer"`.

## Roadmap ideas

1. Leverage MLX prompt caching once upstream exposes a stable API.
2. Add a `benchmark` subcommand that reports tokens/sec across candidate models.
3. Ship optional retrieval plugins that watch git status and automatically provide diff context.

## Development

```bash
python -m mlx_dev_agent.cli --help
```

Run linting, type checks, and tests:

```bash
ruff check mlx_dev_agent
pyright
python -m pytest
```

Enable local pre-commit hooks once to run Ruff and Pyright before each commit:

```bash
pip install pre-commit
pre-commit install
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines, coding standards, and the expected test workflow. By contributing you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Security

If you discover a potential security issue, follow the coordinated disclosure process described in [SECURITY.md](SECURITY.md). Please avoid filing public GitHub issues for sensitive reports.
