from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from mlx_dev_agent.config import AgentConfig
from mlx_dev_agent import engine as engine_module
from mlx_dev_agent.engine import MLXEngine, _normalize_model_identifier


class DummyTokenizer:
    def __init__(self) -> None:
        self.calls = []

    def apply_chat_template(self, messages, add_generation_prompt, tokenize):
        return "prompt"

    def encode(self, text, add_special_tokens=False):
        return [0] * len(text)


class DummyModel:
    def eval(self):
        return None


@pytest.fixture(autouse=True)
def stub_sample_utils(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(engine_module.sample_utils, "make_sampler", lambda **_: lambda _: 0)
    monkeypatch.setattr(engine_module.sample_utils, "make_logits_processors", lambda **_: None)


@pytest.fixture(autouse=True)
def stub_stream_generate(monkeypatch: pytest.MonkeyPatch):
    def fake_stream(*args, **kwargs):
        yield SimpleNamespace(text="hello", generation_tokens=1)

    monkeypatch.setattr(engine_module, "stream_generate", fake_stream)
    return fake_stream


def test_normalize_model_identifier_prefers_absolute(tmp_path: Path) -> None:
    local = tmp_path / "weights"
    local.mkdir()
    result = _normalize_model_identifier(str(local))
    assert result == str(local)


def test_engine_raises_when_no_safetensors(tmp_path: Path) -> None:
    model_dir = tmp_path / "empty"
    model_dir.mkdir()

    cfg = AgentConfig(model=str(model_dir), preload=False)

    with pytest.raises(RuntimeError):
        MLXEngine(cfg)


def test_engine_loads_with_stub(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model.safetensors").write_bytes(b"stub")

    def fake_load(path, **kwargs):
        assert str(model_dir) in path
        return DummyModel(), DummyTokenizer()

    monkeypatch.setattr(engine_module, "load", fake_load)

    cfg = AgentConfig(model=str(model_dir), preload=False)
    engine = MLXEngine(cfg)
    chunks = list(engine.stream_chat([{"role": "user", "content": "hello"}], cfg.generation))
    assert chunks == ["hello"]
    assert engine.last_stats is not None
