"""Command line entry point for the MLX development agent."""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import typer
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.patch_stdout import patch_stdout
from rich.console import Console
from rich.panel import Panel

from .config import AgentConfig, GenerationSettings, load_config
from .conversation import Conversation
from .engine import MLXEngine

app = typer.Typer(help="Ultra-fast local development agent powered by Apple MLX")
logging.basicConfig(level=logging.WARNING)

COMMAND_PREFIX = ":"
SHELL_PREFIX = "!"
HISTORY_FILE = Path.home() / ".cache" / "mlx-agent" / "history.txt"

_ANSI_ESCAPE = re.compile(r"\x1B\[[0-9;]*[A-Za-z]")
_FAKE_ANSI_ESCAPE = re.compile(r"\?\[[0-9;]*[A-Za-z]")
_CONTROL_TRANSLATION = {
    code: None
    for code in range(0x20)
    if chr(code) not in {"\n", "\t"}
}

PROMPT_ARGUMENT = typer.Argument(None, help="Single prompt to run non-interactively")
CONFIG_OPTION = typer.Option(
    None,
    "--config",
    "-c",
    help="Path to a TOML config overriding defaults",
)
MODEL_OPTION = typer.Option(None, help="Model repository or local path")
SYSTEM_PROMPT_OPTION = typer.Option(None, help="Override the system prompt")
TEMPERATURE_OPTION = typer.Option(None, help="Sampling temperature")
TOP_P_OPTION = typer.Option(None, help="Top-p sampling cutoff")
MAX_TOKENS_OPTION = typer.Option(None, help="Maximum new tokens")
SAVE_TRANSCRIPT_OPTION = typer.Option(None, help="Write conversation to this JSON file")
DISABLE_WARMUP_OPTION = typer.Option(False, help="Skip model warmup to save a few seconds")
CONTEXT_OPTION = typer.Option(None, "--context", help="Preload file(s) as context")
COLOR_OPTION = typer.Option(
    None,
    "--color/--no-color",
    help="Enable or disable ANSI colors in CLI output",
)
INIT_CONFIG_ARGUMENT = typer.Argument(None, help="Where to write the default config file")


def _make_console(enable_color: bool) -> Console:
    if enable_color:
        return Console(highlight=False)
    return Console(highlight=False, color_system=None)


def _ensure_history_file() -> Path:
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not HISTORY_FILE.exists():
        HISTORY_FILE.touch()
    return HISTORY_FILE


def _apply_overrides(config: AgentConfig, overrides: dict[str, object]) -> AgentConfig:
    if not overrides:
        return config
    return config.merge(overrides)


def _gather_overrides(
    model: str | None,
    system_prompt: str | None,
    temperature: float | None,
    top_p: float | None,
    max_tokens: int | None,
    save_transcript: Path | None,
    disable_warmup: bool,
    color: bool | None,
) -> dict[str, object]:
    overrides: dict[str, object] = {}
    if model:
        overrides["model"] = model
    if system_prompt:
        overrides["system_prompt"] = system_prompt
    if save_transcript:
        overrides["save_transcript"] = str(save_transcript)
    if disable_warmup:
        overrides["preload"] = False
    if color is not None:
        overrides["color"] = color

    generation: dict[str, object] = {}
    if temperature is not None:
        generation["temperature"] = temperature
    if top_p is not None:
        generation["top_p"] = top_p
    if max_tokens is not None:
        generation["max_tokens"] = max_tokens
    if generation:
        overrides["generation"] = generation
    return overrides


def _read_context_file(path: Path, max_chars: int = 8000) -> str:
    text = path.read_text(encoding="utf-8")
    if len(text) > max_chars:
        truncated = text[:max_chars]
        truncated += "\n...\n[Context truncated to maintain fast generation]\n"
        return truncated
    return text


def _format_context_payload(path: Path, content: str) -> str:
    return f"Context from {path} (last modified {datetime.fromtimestamp(path.stat().st_mtime)}):\n\n{content}"


def _maybe_write_transcript(path: Path | None, conversation: Conversation) -> None:
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(conversation.trimmed_messages(), handle, indent=2)


def _sanitize_chunk(text: str) -> str:
    """Strip ANSI control sequences and non-printable characters from ``text``."""

    text = _ANSI_ESCAPE.sub("", text)
    text = _FAKE_ANSI_ESCAPE.sub("", text)
    return text.translate(_CONTROL_TRANSLATION)


@app.command()
def chat(
    prompt: str | None = PROMPT_ARGUMENT,
    config_path: Path | None = CONFIG_OPTION,
    model: str | None = MODEL_OPTION,
    system_prompt: str | None = SYSTEM_PROMPT_OPTION,
    temperature: float | None = TEMPERATURE_OPTION,
    top_p: float | None = TOP_P_OPTION,
    max_tokens: int | None = MAX_TOKENS_OPTION,
    save_transcript: Path | None = SAVE_TRANSCRIPT_OPTION,
    disable_warmup: bool = DISABLE_WARMUP_OPTION,
    context: list[Path] = CONTEXT_OPTION,
    color: bool | None = COLOR_OPTION,
) -> None:
    """Start an interactive chat or run a single prompt."""

    cfg = load_config(config_path)
    overrides = _gather_overrides(
        model=model,
        system_prompt=system_prompt,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        save_transcript=save_transcript,
        disable_warmup=disable_warmup,
        color=color,
    )
    cfg = _apply_overrides(cfg, overrides)

    console = _make_console(cfg.color)

    try:
        engine = MLXEngine(cfg)
    except RuntimeError as exc:
        border = "cyan" if cfg.color else ""
        console.print(Panel(str(exc), title="mlx-agent", border_style=border))
        raise typer.Exit(code=1) from exc

    if cfg.preload:
        warmup_msg = "Warming up model once to compile weights…"
        console.print(warmup_msg)
        engine.warmup()

    conversation = Conversation(
        tokenizer=engine.tokenizer,
        max_context_tokens=cfg.max_context_tokens,
        system_prompt=cfg.system_prompt,
    )

    for ctx in context or []:
        try:
            payload = _read_context_file(ctx)
        except OSError as exc:
            console.print(f"Failed to read {ctx}: {exc}")
            continue
        conversation.add_user(_format_context_payload(ctx, payload))

    if prompt is not None:
        _run_single_prompt(prompt, conversation, engine, cfg.generation, console)
        _maybe_write_transcript(cfg.save_transcript, conversation)
        return

    session = PromptSession(history=FileHistory(str(_ensure_history_file())))
    border = "cyan" if cfg.color else ""
    console.print(
        Panel(
            "Type `:help` for inline commands",
            title="mlx-agent",
            border_style=border,
        )
    )

    with patch_stdout():
        while True:
            try:
                user_input = session.prompt("» ")
            except (EOFError, KeyboardInterrupt):
                console.print("\nExiting.")
                break
            stripped = user_input.strip()
            if not stripped:
                continue
            if stripped.startswith(COMMAND_PREFIX):
                if _handle_meta_command(
                    stripped,
                    conversation,
                    console,
                    cfg,
                ):
                    continue
                else:
                    break
            if stripped.startswith(SHELL_PREFIX):
                _run_shell_command(stripped[1:], conversation, console)
                continue

            conversation.add_user(stripped)
            if cfg.color:
                console.print("agent› ", style="bold cyan", end="")
            else:
                console.print("agent› ", end="")
            full_reply = _stream_response(conversation, engine, cfg.generation, console)
            conversation.add_assistant(full_reply)
            if cfg.streaming_latency_hints and engine.last_stats:
                stats = engine.last_stats
                hint = (
                    f"\n{stats.tokens} tok in {stats.duration:.2f}s • {stats.tokens_per_second:.1f} tok/s"
                )
                console.print(hint)
            _maybe_write_transcript(cfg.save_transcript, conversation)


def _run_single_prompt(
    prompt: str,
    conversation: Conversation,
    engine: MLXEngine,
    generation: GenerationSettings,
    console: Console,
) -> None:
    conversation.add_user(prompt)
    reply = _stream_response(conversation, engine, generation, console)
    conversation.add_assistant(reply)
    stats = engine.last_stats
    if stats:
        hint = f"\n{stats.tokens} tok in {stats.duration:.2f}s • {stats.tokens_per_second:.1f} tok/s"
        console.print(hint)


def _stream_response(
    conversation: Conversation,
    engine: MLXEngine,
    generation: GenerationSettings,
    console: Console,
) -> str:
    reply_chunks: list[str] = []
    try:
        for chunk in engine.stream_chat(conversation.trimmed_messages(), generation):
            clean = _sanitize_chunk(chunk)
            if not clean:
                continue
            console.print(clean, end="", soft_wrap=True)
            reply_chunks.append(clean)
    finally:
        console.print("")  # ensure newline regardless of early exit
    return "".join(reply_chunks)


def _handle_meta_command(
    command: str,
    conversation: Conversation,
    console: Console,
    cfg: AgentConfig,
) -> bool:
    head, *tail = command[1:].split(" ", 1)
    head = head.lower()

    if head in {"quit", "q", "exit"}:
        console.print("Bye.")
        return False

    if head == "clear":
        conversation.reset()
        console.print("Context cleared.")
        return True

    if head == "help":
        console.print(
            """
Commands:
  :clear              Reset the conversation history
  :config             Show the active configuration
  :system <prompt>    Override the system prompt for the session
  :load <file>        Inject a file's contents as context
  :quit / :exit       Leave the chat
Shell passthrough:
  !<cmd>              Run a shell command and record its stdout as context
            """
        )
        return True

    if head == "config":
        console.print_json(data=asdict(cfg))
        return True

    if head == "system" and tail:
        new_prompt = tail[0]
        conversation.system_prompt = new_prompt
        conversation.reset()
        console.print("System prompt updated and context cleared.")
        return True

    if head == "load" and tail:
        path = Path(tail[0]).expanduser()
        try:
            payload = _read_context_file(path)
        except OSError as exc:
            console.print(f"Failed to read {path}: {exc}")
            return True
        conversation.add_user(_format_context_payload(path, payload))
        console.print(f"Loaded context from {path}")
        return True

    console.print(f"Unknown command: {command}")
    return True


def _run_shell_command(command: str, conversation: Conversation, console: Console) -> None:
    console.print(f"$ {command}")
    try:
        completed = subprocess.run(
            command,
            shell=True,
            check=False,
            capture_output=True,
            text=True,
            env=os.environ,
        )
    except OSError as exc:
        console.print(f"Command failed to start: {exc}")
        return

    output = completed.stdout.strip()
    if output:
        conversation.add_user(
            f"Command `{command}` output:\n\n{output}\n\n(Exit code: {completed.returncode})"
        )
    else:
        conversation.add_user(
            f"Command `{command}` produced no stdout. Exit code: {completed.returncode}"
        )


@app.command("init-config")
def init_config(
    path: Path | None = INIT_CONFIG_ARGUMENT,
) -> None:
    """Write a template configuration file to disk."""

    from .config import write_default_config

    console = _make_console(True)
    target = path or (Path.home() / ".config" / "mlx-agent" / "config.toml")
    try:
        write_default_config(target)
    except FileExistsError as exc:
        console.print(str(exc))
        raise typer.Exit(code=1) from exc
    console.print(f"Wrote template config to {target}")


if __name__ == "__main__":
    app()
