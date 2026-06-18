"""Config loader for STT2TTS MCP — hot-swappable engine config from YAML."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

import yaml


_CONFIG: dict[str, Any] | None = None
_CONFIG_PATH: Path | None = None
_CONFIG_LOCK = threading.RLock()


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load config.yaml with caching. Thread-safe.

    Resolution order:
      1. Explicit `config_path` argument
      2. `$STT2TTS_CONFIG` environment variable
      3. `<package_parent>/config.yaml`  (project root — the common case)
      4. `<package_dir>/config.yaml`     (legacy: next to the package source)
      5. `./config.yaml`                 (CWD fallback)
    """
    global _CONFIG
    if config_path is None:
        env_path = os.environ.get("STT2TTS_CONFIG")
        candidates: list[Path] = []
        if env_path:
            candidates.append(Path(env_path))
        # Project root = parent of the `stt2tts_mcp` package directory.
        candidates.append(Path(__file__).parent.parent.parent / "config.yaml")
        # Legacy: next to the package source.
        candidates.append(Path(__file__).parent.parent / "config.yaml")
        # CWD fallback (useful for `python -m stt2tts_mcp.server` from project root).
        candidates.append(Path.cwd() / "config.yaml")

        config_path = None
        for cand in candidates:
            if cand.expanduser().exists():
                config_path = cand
                break
        if config_path is None:
            tried = "\n  ".join(str(c.expanduser()) for c in candidates)
            raise FileNotFoundError(
                f"config.yaml not found. Looked in:\n  {tried}\n"
                "Set $STT2TTS_CONFIG or pass an explicit path."
            )
    config_path = Path(config_path).expanduser()
    with _CONFIG_LOCK:
        # Invalidate cache when an explicit path differs from the cached one —
        # otherwise `load_config("/new/path.yaml")` would silently return the
        # previously-loaded config and ignore the new path entirely.
        global _CONFIG_PATH
        if _CONFIG is not None and _CONFIG_PATH != config_path:
            _CONFIG = None
        if _CONFIG is not None:
            return _CONFIG
        if not config_path.exists():
            raise FileNotFoundError(f"config.yaml not found at {config_path}")
        with open(config_path) as f:
            _CONFIG = yaml.safe_load(f)
        _CONFIG_PATH = config_path
        return _CONFIG


def reload_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Force-reload config.yaml (e.g., after external edit)."""
    global _CONFIG, _CONFIG_PATH
    with _CONFIG_LOCK:
        _CONFIG = None
        _CONFIG_PATH = None
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
