"""Input validation for stt2tts-mcp tools.

Three layers of defense:
1. Type-shape validation (pydantic-style: required, types, ranges)
2. Path safety (resolves, blocks `..` traversal, checks allowlist)
3. Resource caps (text length, audio size, output bytes)

All validation functions raise `ToolError` subclasses so the MCP layer can
return structured `isError: true` results without leaking stack traces.
"""

from __future__ import annotations

import re
from pathlib import Path

from stt2tts_mcp.security.errors import (
    EmptyTextError,
    InvalidPathError,
    TextTooLongError,
    ToolError,
    UnsupportedFormatError,
)


# Strip ASCII control chars except newline (\n=0x0A), carriage return (\r=0x0D),
# and tab (\t=0x09). Anything else (NUL, BEL, ESC, etc.) is removed — defends
# against ANSI escapes, terminal-control sequences, and NUL byte injection.
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]")


def sanitize_text(text: str, max_length: int) -> str:
    """Strip control characters and enforce a maximum length.

    Raises `EmptyTextError` if the result is empty after sanitization, and
    `TextTooLongError` if input exceeds `max_length` before stripping.
    """
    if not isinstance(text, str):
        raise ToolError(
            message=f"Text must be a string, got {type(text).__name__}",
            details={"type": type(text).__name__},
        )
    if len(text) > max_length:
        raise TextTooLongError(length=len(text), max_length=max_length)
    cleaned = _CONTROL_CHARS_RE.sub("", text).strip()
    if not cleaned:
        raise EmptyTextError()
    return cleaned


def safe_resolve_path(
    raw_path: str | Path,
    *,
    allowed_roots: list[Path] | None = None,
    must_exist: bool = True,
    require_file: bool = True,
    allowed_extensions: tuple[str, ...] | None = None,
) -> Path:
    """Resolve a path with safety checks.

    Behavior:
    - Expands `~` and normalizes via `Path.resolve()` (which collapses `..`).
    - If `allowed_roots` is set, raises `InvalidPathError` if the resolved
      path is not under one of those roots. Empty list == deny all.
    - If `must_exist`, raises `InvalidPathError` if the path doesn't exist.
    - If `require_file`, raises `InvalidPathError` if the path is not a file
      (e.g., a directory or symlink-to-directory).
    - If `allowed_extensions` is set, raises `UnsupportedFormatError` for
      mismatches.

    Notes on symlinks:
    - `Path.resolve()` follows symlinks, so symlinks pointing OUTSIDE an
      allowed root are caught by the `allowed_roots` check.
    - If symlink resolution is undesired, pass `Path(raw_path).absolute()`
      instead — but the safer default is to resolve.
    """
    if not isinstance(raw_path, (str, Path)):
        raise InvalidPathError(
            path=str(raw_path),
            reason=f"path must be str or Path, got {type(raw_path).__name__}",
        )
    if isinstance(raw_path, str) and not raw_path.strip():
        raise InvalidPathError(path=str(raw_path), reason="path is empty")

    path = Path(raw_path).expanduser()

    # Reject NUL bytes (defense-in-depth — pathlib already rejects them, but
    # some downstream tools don't).
    if "\x00" in str(raw_path):
        raise InvalidPathError(path=str(raw_path), reason="path contains NUL byte")

    # Resolve (follows symlinks, normalizes `..`). Use strict=False so we can
    # give a clearer error than FileNotFoundError when must_exist=True.
    try:
        resolved = path.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise InvalidPathError(
            path=str(raw_path), reason=f"cannot resolve: {exc}"
        ) from exc

    if allowed_roots is not None:
        # Compare with is_relative_to (Python 3.9+). We allow the path itself
        # to BE the allowed root.
        ok = any(resolved == root or root in resolved.parents for root in allowed_roots)
        if not ok:
            roots_repr = ", ".join(str(r) for r in allowed_roots) or "<none>"
            raise InvalidPathError(
                path=str(resolved),
                reason=f"path is outside allowed roots: {roots_repr}",
                details={"allowed_roots": [str(r) for r in allowed_roots]},
            )

    if must_exist and not resolved.exists():
        raise InvalidPathError(path=str(resolved), reason="file does not exist")

    if require_file and resolved.exists() and not resolved.is_file():
        kind = "directory" if resolved.is_dir() else "special file"
        raise InvalidPathError(
            path=str(resolved),
            reason=f"path is a {kind}, expected a regular file",
        )

    if allowed_extensions is not None:
        ext = resolved.suffix.lower()
        if ext not in allowed_extensions:
            raise UnsupportedFormatError(path=str(resolved), ext=ext)

    return resolved


def estimate_tts_output_bytes(
    text: str,
    *,
    sample_rate: int = 22050,
    sample_width_bytes: int = 2,  # 16-bit PCM
    channels: int = 1,
    chars_per_second: float = 14.0,
    wav_header_bytes: int = 44,
) -> int:
    """Rough estimate of the WAV size a TTS engine would produce.

    Used to enforce `OutputTooLargeError` BEFORE invoking the engine (so a
    10-million-character text request doesn't first exhaust CPU on the TTS
    backend).

    Default chars/sec (14) is conservative for natural speech; Piper is
    typically 12-18 chars/sec depending on voice. For local-cap enforcement
    we err on the side of overestimation so we don't reject valid requests.
    """
    if chars_per_second <= 0:
        raise ValueError("chars_per_second must be positive")
    seconds = max(0.5, len(text) / chars_per_second)
    return wav_header_bytes + int(seconds * sample_rate * sample_width_bytes * channels)


__all__ = [
    "sanitize_text",
    "safe_resolve_path",
    "estimate_tts_output_bytes",
]
