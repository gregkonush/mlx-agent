"""MLX-backed generation engine used by the CLI."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from .config import AgentConfig, GenerationSettings

logger = logging.getLogger(__name__)

try:
    from mlx_lm import load, sample_utils, stream_generate  # type: ignore
except ImportError as exc:  # pragma: no cover - handled at runtime
    load = None  # type: ignore[assignment]
    stream_generate = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

try:  # ``huggingface_hub`` ships with ``mlx_lm`` so this should usually succeed.
    from huggingface_hub import errors as hf_errors  # type: ignore
except ImportError:  # pragma: no cover - defensive: allow running without hub extras
    hf_errors = None


@dataclass(slots=True)
class GenerationStats:
    """Timing information for a single completion."""

    tokens: int
    duration: float

    @property
    def tokens_per_second(self) -> float:
        if self.duration == 0:
            return float("inf")
        return self.tokens / self.duration


def _normalize_model_identifier(identifier: str) -> str:
    """Resolve convenience shortcuts to absolute filesystem paths."""

    path = Path(identifier).expanduser()
    if path.exists():
        return str(path)

    project_root = Path(__file__).resolve().parent.parent

    candidate = project_root / identifier
    if candidate.exists():
        return str(candidate)

    models_dir = project_root / "models"
    model_candidate = models_dir / identifier
    if model_candidate.exists():
        return str(model_candidate)

    return identifier


class MLXEngine:
    """Wrapper around ``mlx_lm`` utilities that adds small quality-of-life helpers."""

    def __init__(
        self,
        config: AgentConfig,
        tokenizer_config: dict[str, object] | None = None,
    ) -> None:
        if load is None or stream_generate is None:  # pragma: no cover - runtime guard
            raise RuntimeError(
                "mlx_lm is not installed. Install it with `pip install mlx-lm` to use this tool."
            ) from _IMPORT_ERROR

        assert load is not None  # help static type checkers
        assert stream_generate is not None

        assert load is not None  # help static type checkers
        assert stream_generate is not None

        raw_model = config.model
        resolved_model = _normalize_model_identifier(raw_model)
        logger.debug("Loading model %s (resolved to %s)", raw_model, resolved_model)
        load_kwargs = {
            "tokenizer_config": tokenizer_config or {},
            "lazy": False,
        }
        try:
            from inspect import signature

            if "trust_remote_code" in signature(load).parameters:
                load_kwargs["trust_remote_code"] = True
        except (ImportError, ValueError):  # pragma: no cover - defensive
            load_kwargs["trust_remote_code"] = True

        resolved_path = Path(resolved_model)
        if resolved_path.is_dir() and not any(resolved_path.glob("model*.safetensors")):
            message = (
                "Local model weights not found. Ensure the directory contains MLX "
                "`model*.safetensors` files or point `--model` to a valid repository."
            )
            raise RuntimeError(message)

        try:
            self.model, self.tokenizer = load(resolved_model, **load_kwargs)
        except FileNotFoundError as exc:
            message = (
                "Failed to read model weights. If you are using a local directory, "
                "confirm it contains MLX `model*.safetensors` files."
            )
            raise RuntimeError(message) from exc
        except Exception as exc:  # pragma: no cover - rely on runtime behaviour
            if hf_errors is not None:
                if isinstance(exc, hf_errors.GatedRepoError):
                    message = (
                        f"Access to '{raw_model}' is gated on Hugging Face. "
                        "Accept the license and run `hf login`, or "
                        "provide a local model path via `--model`."
                    )
                    raise RuntimeError(message) from exc
                if isinstance(exc, hf_errors.RepositoryNotFoundError):
                    message = (
                        f"Model '{raw_model}' was not found on Hugging Face. "
                        "Double-check the repo id or supply a local directory."
                    )
                    raise RuntimeError(message) from exc
                if isinstance(exc, hf_errors.HfHubHTTPError):
                    message = (
                        f"Hugging Face request for '{raw_model}' failed: {exc}. "
                        "If this is a private or gated repo, authenticate with Hugging Face or "
                        "use a local model directory."
                    )
                    raise RuntimeError(message) from exc
            raise RuntimeError(
                f"Failed to load model '{raw_model}' ({resolved_model}): {exc}"
            ) from exc

        if hasattr(self.model, "eval"):
            self.model.eval()
        self.config = config
        self._model_identifier = resolved_model
        self._warmed_up = False
        self._last_stats: GenerationStats | None = None
        self._stream_generate: Callable[..., Any] = stream_generate

    @property
    def last_stats(self) -> GenerationStats | None:
        return self._last_stats

    def warmup(self) -> None:
        """Run a single short decoding step to trigger MLX compilation."""

        if self._warmed_up:
            return
        dummy_messages = [
            {"role": "system", "content": "You are a blazing-fast assistant."},
            {"role": "user", "content": "Ready?"},
        ]
        tokenizer = cast(Any, self.tokenizer)
        prompt = tokenizer.apply_chat_template(
            dummy_messages, add_generation_prompt=True, tokenize=False
        )
        sampler_factory: Any = sample_utils.make_sampler
        sampler = cast(Callable[..., Any], sampler_factory(temp=0.0))
        try:
            for _ in self._stream_generate(
                self.model,
                self.tokenizer,
                prompt,
                max_tokens=1,
                sampler=sampler,
            ):  # type: ignore[reportCallIssue]
                break
        except Exception as exc:  # pragma: no cover - defensive: warmups shouldn't kill CLI
            logger.debug("Warmup failed: %%s", exc)
        finally:
            self._warmed_up = True

    def stream_chat(
        self,
        messages: Iterable[dict[str, str]],
        sampling: GenerationSettings,
    ) -> Iterator[str]:
        """Yield response chunks for the provided chat history."""

        tokenizer = cast(Any, self.tokenizer)
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        start = time.perf_counter()
        last_token_count = 0

        sampler_factory: Any = sample_utils.make_sampler
        sampler = cast(
            Callable[..., Any],
            sampler_factory(
                temp=sampling.temperature,
                top_p=sampling.top_p,
            ),
        )
        logits_processors = cast(
            Any,
            sample_utils.make_logits_processors(
                repetition_penalty=(
                    sampling.repetition_penalty if sampling.repetition_penalty != 1.0 else None
                )
            ),
        )

        kwargs: dict[str, Any] = {
            "max_tokens": sampling.max_tokens,
            "sampler": sampler,
        }
        if logits_processors:
            kwargs["logits_processors"] = logits_processors
        logger.debug("Starting generation with kwargs: %s", kwargs)
        try:
            for response in self._stream_generate(
                self.model,
                self.tokenizer,
                prompt,
                **kwargs,
            ):  # type: ignore[reportCallIssue]
                last_token_count = response.generation_tokens
                text = response.text  # type: ignore[attr-defined]
                if not isinstance(text, str) or not text:
                    continue
                yield text
        finally:
            duration = time.perf_counter() - start
            self._last_stats = GenerationStats(tokens=last_token_count, duration=duration)
