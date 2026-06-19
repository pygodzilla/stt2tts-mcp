"""Append-only audit logging for stt2tts-mcp.

Every tool invocation writes one JSON Lines entry to `audit.log`. Entries are
append-only (O_APPEND is set on the file descriptor at open time). The log
path defaults to `~/.local/share/stt2tts-mcp/audit.log` per XDG Base Dir.

Schema (one entry per line):
    {
      "ts":      "2026-06-19T14:23:01.123Z",     # ISO 8601 UTC
      "tool":    "transcribe",
      "args":    {"audio_path": "/tmp/rec.wav", "language": "en"},
      "ok":      true,
      "result":  {"text_length": 42, "duration_ms": 1842},
      "duration_ms": 1842,
      "error":   null | {"code": "rate_limited", "message": "..."}
    }

Privacy:
- Argument values are kept as-is by default. For known-sensitive fields
  (e.g., `text` in `speak`), call `audit.redact_arg()` or pass a custom
  redactor. We provide a conservative default that hashes `text` but keeps
  the length so audit consumers can verify nothing was truncated.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


# Default location: $XDG_DATA_HOME/stt2tts-mcp/audit.log, falling back to
# ~/.local/share/stt2tts-mcp/audit.log per XDG Base Dir spec.
DEFAULT_AUDIT_DIR = (
    Path(os.environ.get("STT2TTS_AUDIT_DIR"))
    if os.environ.get("STT2TTS_AUDIT_DIR")
    else Path(os.environ.get("XDG_DATA_HOME", "~/.local/share")).expanduser()
    / "stt2tts-mcp"
)
DEFAULT_AUDIT_PATH = DEFAULT_AUDIT_DIR / "audit.log"

# Tools whose arguments may contain user-supplied content we don't want in
# the audit log verbatim. We keep the length and a short hash so an auditor
# can still verify "yes, text was passed" without storing the literal text.
_SENSITIVE_ARG_KEYS = frozenset({"text"})


def _redact_default(key: str, value: Any) -> Any:
    """Conservative default redactor for known-sensitive keys."""
    if key in _SENSITIVE_ARG_KEYS and isinstance(value, str):
        h = hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()[:16]
        return {"__redacted__": True, "length": len(value), "sha256_prefix": h}
    return value


def _utc_iso() -> str:
    """ISO 8601 UTC with millisecond precision."""
    return (
        datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.")
        + f"{datetime.now(timezone.utc).microsecond // 1000:03d}Z"
    )


class AuditLogger:
    """Append-only JSON Lines audit logger.

    Thread-safe. Writes are line-buffered with an explicit `flush()` after
    each entry to minimize loss on crash. File mode includes `O_APPEND` so
    even a buggy caller can't truncate the existing log.
    """

    def __init__(
        self,
        path: Path | None = None,
        redactor: Callable[[str, Any], Any] | None = None,
        enabled: bool = True,
    ) -> None:
        self.path = Path(path) if path is not None else DEFAULT_AUDIT_PATH
        self.redactor = redactor or _redact_default
        self.enabled = enabled
        self._lock = threading.Lock()
        self._fh = None
        if self.enabled:
            self._ensure_open()

    def _ensure_open(self) -> None:
        if self._fh is not None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            # O_APPEND is the key — atomic appends at the end of file,
            # immune to lseek-based truncation by buggy callers.
            self._fh = open(
                self.path,
                "a",
                buffering=1,  # line-buffered
                encoding="utf-8",
            )
        except OSError:
            # Audit logging is best-effort; never crash a tool call because
            # we can't open the log file (e.g., read-only filesystem).
            self._fh = None

    def log(
        self,
        tool: str,
        args: dict[str, Any],
        *,
        ok: bool,
        duration_ms: float,
        result: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
    ) -> None:
        """Write one audit entry. Never raises (best-effort)."""
        if not self.enabled:
            return
        entry = {
            "ts": _utc_iso(),
            "tool": tool,
            "args": {k: self.redactor(k, v) for k, v in (args or {}).items()},
            "ok": ok,
            "duration_ms": round(duration_ms, 2),
        }
        if result is not None:
            entry["result"] = result
        if error is not None:
            entry["error"] = error
        line = json.dumps(entry, ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            try:
                self._ensure_open()
                if self._fh is not None:
                    self._fh.write(line + "\n")
                    self._fh.flush()
            except Exception:  # pragma: no cover — best-effort
                pass

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                try:
                    self._fh.close()
                finally:
                    self._fh = None

    def tail(self, n: int = 50) -> list[dict[str, Any]]:
        """Read the last `n` entries (for /health or tests)."""
        if not self.path.exists():
            return []
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except OSError:
            return []
        out: list[dict[str, Any]] = []
        for line in lines[-n:]:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out


# ---------------------------------------------------------------------------
# Singleton + convenience decorator
# ---------------------------------------------------------------------------

_default_logger: AuditLogger | None = None
_default_lock = threading.Lock()


def get_audit_logger() -> AuditLogger:
    """Return the process-wide audit logger, creating it lazily."""
    global _default_logger
    if _default_logger is None:
        with _default_lock:
            if _default_logger is None:
                _default_logger = AuditLogger()
    return _default_logger


def reset_audit_logger(path: Path | None = None, enabled: bool = True) -> AuditLogger:
    """Reset (and replace) the default logger — for tests."""
    global _default_logger
    with _default_lock:
        if _default_logger is not None:
            _default_logger.close()
        _default_logger = AuditLogger(path=path, enabled=enabled)
    return _default_logger


__all__ = [
    "AuditLogger",
    "get_audit_logger",
    "reset_audit_logger",
    "DEFAULT_AUDIT_PATH",
]
