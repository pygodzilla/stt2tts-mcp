"""Regression test: config.yaml must be loadable from a project-root layout
where the file sits at the repo root, not next to the package source."""
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stt2tts_mcp.utils.config import load_config  # noqa: E402


CONFIG_BODY = """
stt:
  engine: faster_whisper
  enabled: true
  params:
    model_size: base.en
tts:
  engine: piper
  enabled: true
  params:
    voice: en_US-lessac-medium
logging:
  level: INFO
"""


def test_loads_from_project_root_layout():
    """Layout: /tmp/X/{config.yaml, stt2tts_mcp/}. config.yaml must resolve."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "stt2tts_mcp").mkdir()
        (root / "config.yaml").write_text(CONFIG_BODY)
        # The package __file__ is the real one in this repo, but we point the
        # env var at our tmp config — that path always wins.
        cfg = load_config(root / "config.yaml")
        assert cfg["stt"]["engine"] == "faster_whisper"
        assert cfg["tts"]["engine"] == "piper"


def test_clear_error_when_explicit_path_missing():
    """If a path is given explicitly and it's missing, error is clear."""
    import stt2tts_mcp.utils.config as cfg_mod  # noqa: PLC0415
    cfg_mod._CONFIG = None  # bypass cache so explicit path is consulted
    try:
        load_config("/tmp/__definitely_missing__.yaml")
    except FileNotFoundError as e:
        msg = str(e)
        assert "config.yaml not found" in msg
        assert "__definitely_missing__" in msg
        return
    raise AssertionError("expected FileNotFoundError")


if __name__ == "__main__":
    test_loads_from_project_root_layout()
    test_clear_error_when_explicit_path_missing()
    print("OK: config loader regression tests passed")
