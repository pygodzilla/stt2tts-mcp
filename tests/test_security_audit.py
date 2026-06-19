"""Unit tests for the append-only audit logger."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

from stt2tts_mcp.security.audit import (
    AuditLogger,
    get_audit_logger,
    reset_audit_logger,
)


def test_log_creates_file_and_appends(tmp_path: Path) -> None:
    log_path = tmp_path / "audit.log"
    al = AuditLogger(path=log_path, enabled=True)
    al.log("speak", {"text": "hello"}, ok=True, duration_ms=120.0)
    al.log(
        "transcribe",
        {"audio_path": "/tmp/x.wav"},
        ok=False,
        duration_ms=80.0,
        error={"code": "internal_error", "message": "boom"},
    )
    al.close()

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    e1, e2 = (json.loads(l) for l in lines)

    assert e1["tool"] == "speak"
    assert e1["ok"] is True
    assert e1["duration_ms"] == 120.0
    assert e1["args"]["text"]["__redacted__"] is True
    assert e1["args"]["text"]["length"] == 5  # "hello"

    assert e2["tool"] == "transcribe"
    assert e2["ok"] is False
    assert e2["error"]["code"] == "internal_error"
    assert e2["args"]["audio_path"] == "/tmp/x.wav"


def test_log_redacts_only_sensitive_keys(tmp_path: Path) -> None:
    log_path = tmp_path / "audit.log"
    al = AuditLogger(path=log_path, enabled=True)
    al.log("speak", {"text": "secret", "voice": "alice"}, ok=True, duration_ms=10)
    al.close()

    entry = json.loads(log_path.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert entry["args"]["text"]["__redacted__"] is True
    # Non-sensitive args must be passed through verbatim.
    assert entry["args"]["voice"] == "alice"


def test_log_creates_parent_dir(tmp_path: Path) -> None:
    log_path = tmp_path / "deep" / "nested" / "audit.log"
    al = AuditLogger(path=log_path, enabled=True)
    al.log("speak", {}, ok=True, duration_ms=1.0)
    al.close()
    assert log_path.exists()


def test_disabled_logger_writes_nothing(tmp_path: Path) -> None:
    log_path = tmp_path / "audit.log"
    al = AuditLogger(path=log_path, enabled=False)
    al.log("speak", {"text": "x"}, ok=True, duration_ms=1.0)
    al.close()
    assert not log_path.exists()


def test_log_failure_does_not_raise(tmp_path: Path) -> None:
    """Audit logging is best-effort — must never crash a tool call."""
    # Use a path we can't write to (under a non-directory).
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    bad_path = blocker / "audit.log"
    al = AuditLogger(path=bad_path, enabled=True)
    # Must not raise.
    al.log("speak", {}, ok=True, duration_ms=1.0)
    al.close()


def test_log_appends_not_overwrites(tmp_path: Path) -> None:
    """Two loggers on the same path must APPEND, not overwrite."""
    log_path = tmp_path / "audit.log"
    al1 = AuditLogger(path=log_path, enabled=True)
    al1.log("speak", {}, ok=True, duration_ms=1.0)
    al1.close()

    al2 = AuditLogger(path=log_path, enabled=True)
    al2.log("transcribe", {}, ok=True, duration_ms=1.0)
    al2.close()

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2


def test_tail_returns_last_n(tmp_path: Path) -> None:
    log_path = tmp_path / "audit.log"
    al = AuditLogger(path=log_path, enabled=True)
    for i in range(5):
        al.log(f"tool_{i}", {}, ok=True, duration_ms=float(i))
    al.close()

    tail = al.tail(3)
    assert len(tail) == 3
    assert tail[0]["tool"] == "tool_2"
    assert tail[-1]["tool"] == "tool_4"


def test_tail_handles_missing_file(tmp_path: Path) -> None:
    al = AuditLogger(path=tmp_path / "does_not_exist.log", enabled=True)
    assert al.tail() == []


def test_get_audit_logger_returns_singleton(tmp_path: Path) -> None:
    reset_audit_logger(path=tmp_path / "audit.log", enabled=True)
    a = get_audit_logger()
    b = get_audit_logger()
    assert a is b


def test_iso_timestamp_is_utc() -> None:
    """Timestamps must be UTC with Z suffix for SIEM-friendly logs."""
    log_path = Path("/tmp/_stt2tts_test_audit.log")
    if log_path.exists():
        log_path.unlink()
    al = AuditLogger(path=log_path, enabled=True)
    al.log("speak", {}, ok=True, duration_ms=1.0)
    al.close()
    entry = json.loads(log_path.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert entry["ts"].endswith("Z")
    # Should parse as ISO 8601.
    from datetime import datetime

    parsed = datetime.fromisoformat(entry["ts"].replace("Z", "+00:00"))
    assert parsed.utcoffset().total_seconds() == 0
