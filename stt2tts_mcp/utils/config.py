"""Config loader for STT2TTS MCP — hot-swappable engine config from YAML."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

import yaml


_CONFIG: dict[str, Any] | None = None
_CONFIG_LOCK = threading.RLock()


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load config.yaml with caching. Thread-safe."""
    global _CONFIG
    if config_path is None:
        config_path = os.environ.get(
            "STT2TTS_CONFIG",
            str(Path(__file__).parent.parent / "config.yaml"),
        )
    config_path = Path(config_path).expanduser()
    with _CONFIG_LOCK:
        if _CONFIG is not None:
            return _CONFIG
        if not config_path.exists():
            raise FileNotFoundError(f"config.yaml not found at {config_path}")
        with open(config_path) as f:
            _CONFIG = yaml.safe_load(f)
        return _CONFIG


def reload_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Force-reload config.yaml (e.g., after external edit)."""
    global _CONFIG
    with _CONFIG_LOCK:
        _CONFIG = None
    return load_config(config_path)


def get_stt_config(config: dict[str, Any]) -> dict[str, Any] | None:
    """Get the active STT engine config."""
    stt = config.get("stt", {})
    if not stt.get("enabled", True):
        return None
    return stt


def get_tts_config(config: dict[str, Any]) -> dict[str, Any] | None:
    """Get the active TTS engine config."""
    tts = config.get("tts", {})
    if not tts.get("enabled", True):
        return None
    return tts


def get_engine_params(config_section: dict[str, Any]) -> dict[str, Any]:
    """Extract engine params from a config section."""
    return dict(config_section.get("params", {}))
