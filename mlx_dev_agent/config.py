"""Configuration loading for the MLX development agent CLI."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:  # Python >=3.11
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # Fallback for Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]

DEFAULT_CONFIG_PATHS: tuple[Path, ...] = (
    Path("mlx-agent.toml"),
    Path.home() / ".config" / "mlx-agent" / "config.toml",
)


def _find_local_model() -> Path | None:
    """Return the first bundled model directory that looks loadable."""

    package_root = Path(__file__).resolve().parent.parent
    models_dir = package_root / "models"
    if not models_dir.is_dir():
        return None

    priority = (
        "Qwen3-8B-MLX-4bit",
        "phi-3-mini-128k-instruct-4bit-mlx",
        "mistral-7b-instruct-v0.2",
    )
    for name in priority:
        candidate = models_dir / name
        if candidate.is_dir() and any(candidate.glob("model*.safetensors")):
            return candidate

    for candidate in sorted(models_dir.iterdir()):
        if candidate.is_dir() and any(candidate.glob("model*.safetensors")):
            return candidate
    return None


def _default_model() -> str:
    """Pick a sensible default model that works without network access."""

    bundled = _find_local_model()
    if bundled is not None:
        return str(bundled)
    # Fallback to a public MLX model; users may need to authenticate with HF.
    return "mlx-community/Qwen3-8B-MLX-4bit"


@dataclass(slots=True)
class GenerationSettings:
    """Sampling settings forwarded to ``mlx_lm.generate``."""

    max_tokens: int = 1024
    temperature: float = 0.2
    top_p: float = 0.9
    repetition_penalty: float = 1.05
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: tuple[str, ...] = field(default_factory=tuple)


@dataclass(slots=True)
class AgentConfig:
    """Top-level agent configuration."""

    model: str = field(default_factory=_default_model)
    system_prompt: str = (
        "You are a senior software engineer helping with coding tasks. "
        "Respond concisely, prefer actionable answers, and cite files when referencing code."
    )
    compile: bool = True
    max_context_tokens: int = 8192
    streaming_latency_hints: bool = True
    save_transcript: Path | None = None
    preload: bool = True
    color: bool = False
    generation: GenerationSettings = field(default_factory=GenerationSettings)

    def merge(self, overrides: dict[str, Any]) -> AgentConfig:
        """Return a new config with ``overrides`` applied recursively."""

        merged = AgentConfig(
            model=overrides.get("model", self.model),
            system_prompt=overrides.get("system_prompt", self.system_prompt),
            compile=overrides.get("compile", self.compile),
            max_context_tokens=overrides.get("max_context_tokens", self.max_context_tokens),
            streaming_latency_hints=overrides.get(
                "streaming_latency_hints", self.streaming_latency_hints
            ),
            save_transcript=_parse_path(overrides.get("save_transcript", self.save_transcript)),
            preload=overrides.get("preload", self.preload),
            color=overrides.get("color", self.color),
            generation=_merge_generation(self.generation, overrides.get("generation", {})),
        )
        return merged


def load_config(explicit_path: Path | None = None) -> AgentConfig:
    """Load configuration from disk, returning defaults when no file exists."""

    config = AgentConfig()
    if explicit_path:
        data = _read_toml(explicit_path)
        if data:
            config = config.merge(data)
        return config

    for candidate in DEFAULT_CONFIG_PATHS:
        data = _read_toml(candidate)
        if data:
            config = config.merge(data)
            break
    return config


def _merge_generation(
    base: GenerationSettings, overrides: dict[str, Any]
) -> GenerationSettings:
    return GenerationSettings(
        max_tokens=int(overrides.get("max_tokens", base.max_tokens)),
        temperature=float(overrides.get("temperature", base.temperature)),
        top_p=float(overrides.get("top_p", base.top_p)),
        repetition_penalty=float(overrides.get("repetition_penalty", base.repetition_penalty)),
        frequency_penalty=float(overrides.get("frequency_penalty", base.frequency_penalty)),
        presence_penalty=float(overrides.get("presence_penalty", base.presence_penalty)),
        stop=tuple(overrides.get("stop", base.stop)),
    )


def _parse_path(value: Any) -> Path | None:
    if value in (None, "", False):
        return None
    return Path(str(value)).expanduser()


def _read_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("rb") as fh:
        return tomllib.load(fh)


def serialize_config(config: AgentConfig) -> dict[str, Any]:
    """Serialize config to a TOML-compatible dictionary."""

    data: dict[str, Any] = {
        "model": config.model,
        "system_prompt": config.system_prompt,
        "compile": config.compile,
        "max_context_tokens": config.max_context_tokens,
        "streaming_latency_hints": config.streaming_latency_hints,
        "preload": config.preload,
        "color": config.color,
        "generation": {
            "max_tokens": config.generation.max_tokens,
            "temperature": config.generation.temperature,
            "top_p": config.generation.top_p,
            "repetition_penalty": config.generation.repetition_penalty,
            "frequency_penalty": config.generation.frequency_penalty,
            "presence_penalty": config.generation.presence_penalty,
            "stop": list(config.generation.stop),
        },
    }
    if config.save_transcript:
        data["save_transcript"] = str(config.save_transcript)
    return data


def write_default_config(path: Path) -> None:
    """Create a default config file at ``path`` if it does not exist."""

    if path.exists():
        raise FileExistsError(f"Configuration file already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)

    data = serialize_config(AgentConfig())
    with path.open("w", encoding="utf-8") as fh:
        fh.write("# MLX Dev Agent configuration\n")
        fh.write(_toml_dumps(data))


def _toml_dumps(data: dict[str, Any], indent: int = 0) -> str:
    """Hand-rolled minimal TOML writer to avoid extra dependencies."""

    lines: list[str] = []
    prefix = "".rjust(indent)
    generation = data.pop("generation", None)
    for key, value in data.items():
        if isinstance(value, bool):
            rendered = "true" if value else "false"
        elif isinstance(value, (int, float)):
            rendered = str(value)
        else:
            rendered = f"\"{value}\""
        lines.append(f"{prefix}{key} = {rendered}")

    if generation is not None:
        lines.append("")
        lines.append(f"{prefix}[generation]")
        for g_key, g_value in generation.items():
            if isinstance(g_value, list):
                list_values = ", ".join(f"\"{item}\"" for item in g_value)
                rendered = f"[{list_values}]"
            elif isinstance(g_value, (int, float)):
                rendered = str(g_value)
            else:
                rendered = f"\"{g_value}\""
            lines.append(f"{prefix}{g_key} = {rendered}")
    return "\n".join(lines) + "\n"
