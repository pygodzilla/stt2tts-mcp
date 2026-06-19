"""Unit tests for input validation (path safety + text sanitization)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from stt2tts_mcp.security.errors import (
    EmptyTextError,
    InvalidPathError,
    TextTooLongError,
    UnsupportedFormatError,
)
from stt2tts_mcp.security.validation import (
    estimate_tts_output_bytes,
    safe_resolve_path,
    sanitize_text,
)


# ---------------------------------------------------------------------------
# sanitize_text
# ---------------------------------------------------------------------------


def test_sanitize_text_strips_control_chars() -> None:
    """NUL, BEL, ESC etc. must be stripped; \\n\\r\\t preserved.

    Note: ANSI escape sequences use ESC + '[' + params + '[' + 'm'. The
    sanitizer strips ESC (the trigger), keeping the brackets and 'm' (which
    are visible ASCII). This still defeats the terminal-control attack
    (which needs ESC to function) while not corrupting innocent text that
    happens to contain brackets.
    """
    raw = "hello\x00\x07\x1b[31m world\nfoo\tbar\r"
    # After stripping control chars (NUL=0x00, BEL=0x07, ESC=0x1B, CR=0x0D
    # is NOT stripped — only \r after \n is, because .strip() removes it).
    result = sanitize_text(raw, max_length=100)
    assert "\x00" not in result
    assert "\x07" not in result
    assert "\x1b" not in result
    # \\n and \\t are preserved.
    assert "\n" in result
    assert "\t" in result
    assert result == "hello[31m world\nfoo\tbar"


def test_sanitize_text_strips_outer_whitespace() -> None:
    assert sanitize_text("   spaced   ", max_length=100) == "spaced"


def test_sanitize_text_rejects_empty_after_strip() -> None:
    with pytest.raises(EmptyTextError):
        sanitize_text("\x00\x07\x1b", max_length=100)


def test_sanitize_text_rejects_empty_input() -> None:
    with pytest.raises(EmptyTextError):
        sanitize_text("", max_length=100)


def test_sanitize_text_rejects_too_long() -> None:
    with pytest.raises(TextTooLongError) as exc:
        sanitize_text("x" * 101, max_length=100)
    assert exc.value.details["length"] == 101
    assert exc.value.details["max_length"] == 100


def test_sanitize_text_accepts_exactly_at_limit() -> None:
    s = "x" * 100
    assert sanitize_text(s, max_length=100) == s


def test_sanitize_text_rejects_non_string() -> None:
    """Non-string input must raise a ToolError subclass."""
    from stt2tts_mcp.security.errors import ToolError

    with pytest.raises(ToolError) as exc:
        sanitize_text(12345, max_length=100)  # type: ignore[arg-type]
    assert exc.value.code in ("internal_error", "text_too_long", "empty_text")


# ---------------------------------------------------------------------------
# safe_resolve_path
# ---------------------------------------------------------------------------


def test_safe_resolve_path_expands_tilde(tmp_path: Path) -> None:
    f = tmp_path / "x.wav"
    f.write_bytes(b"RIFF")
    resolved = safe_resolve_path(f, must_exist=True, require_file=True)
    assert resolved == f.resolve()


def test_safe_resolve_path_accepts_relative(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "rel.wav"
    f.write_bytes(b"RIFF")
    resolved = safe_resolve_path("rel.wav", must_exist=True, require_file=True)
    assert resolved == f.resolve()


def test_safe_resolve_path_rejects_missing(tmp_path: Path) -> None:
    with pytest.raises(InvalidPathError) as exc:
        safe_resolve_path(tmp_path / "ghost.wav", must_exist=True, require_file=True)
    assert "does not exist" in exc.value.details["reason"]


def test_safe_resolve_path_rejects_directory(tmp_path: Path) -> None:
    with pytest.raises(InvalidPathError) as exc:
        safe_resolve_path(tmp_path, must_exist=False, require_file=True)
    assert "directory" in exc.value.details["reason"]


def test_safe_resolve_path_blocks_traversal_outside_allowed_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Path containing `..` that escapes the allowlist must be rejected."""
    inside = tmp_path / "inside"
    inside.mkdir()
    f = inside / "ok.wav"
    f.write_bytes(b"RIFF")
    # Construct a path with `..` that resolves outside `inside`.
    sneaky = inside / ".." / "outside" / "secret.wav"
    with pytest.raises(InvalidPathError) as exc:
        safe_resolve_path(
            sneaky, allowed_roots=[inside], must_exist=False, require_file=False
        )
    assert "outside allowed roots" in exc.value.details["reason"]
    assert exc.value.details["allowed_roots"] == [str(inside.resolve())]


def test_safe_resolve_path_allows_path_inside_allowed_root(tmp_path: Path) -> None:
    inside = tmp_path / "allowed"
    inside.mkdir()
    f = inside / "ok.wav"
    f.write_bytes(b"RIFF")
    resolved = safe_resolve_path(
        f, allowed_roots=[inside], must_exist=True, require_file=True
    )
    assert resolved == f.resolve()


def test_safe_resolve_path_rejects_nul_byte(tmp_path: Path) -> None:
    with pytest.raises(InvalidPathError) as exc:
        safe_resolve_path("ok\x00.wav", must_exist=False, require_file=False)
    assert "NUL" in exc.value.details["reason"]


def test_safe_resolve_path_rejects_empty_string() -> None:
    with pytest.raises(InvalidPathError):
        safe_resolve_path("", must_exist=False, require_file=False)


def test_safe_resolve_path_rejects_whitespace_only() -> None:
    with pytest.raises(InvalidPathError):
        safe_resolve_path("   ", must_exist=False, require_file=False)


def test_safe_resolve_path_rejects_non_string_non_path() -> None:
    with pytest.raises(InvalidPathError):
        safe_resolve_path(12345, must_exist=False, require_file=False)  # type: ignore[arg-type]


def test_safe_resolve_path_enforces_allowed_extensions(tmp_path: Path) -> None:
    f = tmp_path / "bad.xyz"
    f.write_bytes(b"x")
    with pytest.raises(UnsupportedFormatError) as exc:
        safe_resolve_path(
            f, must_exist=True, require_file=True, allowed_extensions=(".wav", ".mp3")
        )
    assert exc.value.details["extension"] == ".xyz"


# ---------------------------------------------------------------------------
# estimate_tts_output_bytes
# ---------------------------------------------------------------------------


def test_estimate_output_scales_with_text_length() -> None:
    short = estimate_tts_output_bytes("hi")
    long = estimate_tts_output_bytes("x" * 1000)
    assert long > short * 100  # should scale ~linearly


def test_estimate_output_has_minimum_for_tiny_text() -> None:
    """Even a 1-char text has 0.5s minimum + 44-byte WAV header."""
    est = estimate_tts_output_bytes("x")
    # 0.5s @ 22050Hz * 2 bytes (16-bit) * 1 channel + 44-byte WAV header.
    expected = 44 + int(0.5 * 22050 * 2 * 1)
    assert est == expected
    assert est > 0


def test_estimate_output_raises_on_zero_rate() -> None:
    with pytest.raises(ValueError):
        estimate_tts_output_bytes("hi", chars_per_second=0)
