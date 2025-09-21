from pathlib import Path

import pytest

from mlx_dev_agent import config as config_module
from mlx_dev_agent.config import AgentConfig, GenerationSettings, _merge_generation, load_config


def test_default_model_prefers_local(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    models_dir = tmp_path / "models" / "Qwen3-8B-8bit"
    models_dir.mkdir(parents=True)
    (models_dir / "model.safetensors").write_bytes(b"stub")

    monkeypatch.setattr(config_module, "_find_local_model", lambda: models_dir)

    cfg = AgentConfig()
    assert cfg.model == str(models_dir)


def test_load_config_merges_overrides(tmp_path: Path) -> None:
    config_path = tmp_path / "custom.toml"
    config_path.write_text(
        """
model = "custom/model"
preload = false
color = true
[generation]
max_tokens = 256
        """
    )
    cfg = load_config(config_path)
    assert cfg.model == "custom/model"
    assert cfg.preload is False
    assert cfg.color is True
    assert cfg.generation.max_tokens == 256


def test_merge_generation_overrides_selected_fields() -> None:
    base = GenerationSettings()
    overrides = {"temperature": 0.5, "stop": ["done"], "max_tokens": 32}
    result = _merge_generation(base, overrides)
    assert result.temperature == 0.5
    assert result.max_tokens == 32
    assert result.stop == ("done",)
    # unchanged fields remain identical
    assert result.top_p == base.top_p
