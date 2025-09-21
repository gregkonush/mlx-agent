from pathlib import Path

from mlx_dev_agent import cli


def test_sanitize_chunk_strips_ansi() -> None:
    raw = "\x1b[1;32mhello\x1b[0m \u0007"
    assert cli._sanitize_chunk(raw) == "hello "


def test_read_context_truncates(tmp_path: Path) -> None:
    blob = "x" * 10
    path = tmp_path / "context.txt"
    path.write_text(blob)
    result = cli._read_context_file(path, max_chars=5)
    assert "Context truncated" in result
    assert result.startswith("xxxxx")
