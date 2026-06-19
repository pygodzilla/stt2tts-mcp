"""Unit tests for the structured error hierarchy.

These tests don't need a live MCP server subprocess — they exercise the
error class behavior directly, including serialization to `CallToolResult`.
"""

from __future__ import annotations

import pytest

from mcp.types import CallToolResult, TextContent

from stt2tts_mcp.security.errors import (
    ConfigError,
    EmptyTextError,
    EngineUnavailableError,
    InvalidAudioError,
    InvalidPathError,
    OutputTooLargeError,
    RateLimitError,
    TextTooLongError,
    ToolError,
    UnknownToolError,
    UnsupportedFormatError,
)


# ---------------------------------------------------------------------------
# Base ToolError behavior
# ---------------------------------------------------------------------------


def test_tool_error_has_stable_code() -> None:
    err = ToolError("oops", details={"foo": "bar"})
    assert err.code == "internal_error"
    assert err.message == "oops"
    assert err.details == {"foo": "bar"}
    assert str(err) == "oops"


def test_tool_error_default_message_when_none() -> None:
    err = ToolError()
    assert err.message == "An internal error occurred"
    assert err.details == {}


def test_tool_error_to_tool_result_includes_structured_content() -> None:
    """MCP 2.0 requires isError=True + structuredContent for tool errors."""
    err = ToolError("kaboom", details={"k": "v"})
    result = err.to_tool_result()
    assert isinstance(result, CallToolResult)
    assert result.isError is True
    # Human-readable text is still in content (backwards compat).
    assert any(c.type == "text" and "kaboom" in c.text for c in result.content)
    # Structured payload available to clients (MCP 2.0).
    sc = result.structuredContent
    assert sc is not None
    assert sc["ok"] is False
    assert sc["error"]["code"] == "internal_error"
    assert sc["error"]["message"] == "kaboom"
    assert sc["error"]["details"] == {"k": "v"}


# ---------------------------------------------------------------------------
# Specific subclasses — each must have a unique stable code
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cls, expected_code",
    [
        (InvalidPathError, "invalid_path"),
        (TextTooLongError, "text_too_long"),
        (EmptyTextError, "empty_text"),
        (InvalidAudioError, "invalid_audio"),
        (UnsupportedFormatError, "unsupported_format"),
        (RateLimitError, "rate_limited"),
        (OutputTooLargeError, "output_too_large"),
        (EngineUnavailableError, "engine_unavailable"),
        (UnknownToolError, "unknown_tool"),
        (ConfigError, "config_error"),
    ],
)
def test_each_error_class_has_stable_unique_code(
    cls: type[ToolError], expected_code: str
) -> None:
    """Stable codes are part of the public contract — clients route on them."""
    # Construct with the class-specific signature if available.
    if cls is InvalidPathError:
        inst = cls(path="/x", reason="bad")
    elif cls is TextTooLongError:
        inst = cls(length=200, max_length=100)
    elif cls is EmptyTextError:
        inst = cls()
    elif cls is InvalidAudioError:
        inst = cls(path="/x.wav", reason="corrupt")
    elif cls is UnsupportedFormatError:
        inst = cls(path="/x.txt", ext=".txt")
    elif cls is RateLimitError:
        inst = cls(tool="speak", retry_after=1.5)
    elif cls is OutputTooLargeError:
        inst = cls(estimated_bytes=200, max_bytes=100)
    elif cls is EngineUnavailableError:
        inst = cls(kind="stt", name="missing_engine")
    elif cls is UnknownToolError:
        inst = cls(name="nonexistent")
    else:  # ConfigError
        inst = cls(message="bad config")

    assert inst.code == expected_code, f"{cls.__name__}.code changed!"
    # Every error must serialize cleanly.
    result = inst.to_tool_result()
    assert result.isError is True
    assert result.structuredContent is not None
    assert result.structuredContent["error"]["code"] == expected_code


def test_invalid_path_error_includes_path_and_reason() -> None:
    err = InvalidPathError(path="/etc/passwd", reason="outside allowed roots")
    assert "/etc/passwd" in err.message
    assert "outside allowed roots" in err.message
    sc = err.to_tool_result().structuredContent
    assert sc["error"]["details"]["path"] == "/etc/passwd"
    assert sc["error"]["details"]["reason"] == "outside allowed roots"


def test_rate_limit_error_includes_retry_after() -> None:
    err = RateLimitError(tool="speak", retry_after=2.5)
    assert err.code == "rate_limited"
    assert err.details["tool"] == "speak"
    assert err.details["retry_after"] == 2.5
    assert "2.50s" in err.message or "2.5s" in err.message


def test_text_too_long_error_includes_lengths() -> None:
    err = TextTooLongError(length=200_000, max_length=100_000)
    assert err.details["length"] == 200_000
    assert err.details["max_length"] == 100_000


def test_unsupported_format_error_lists_supported() -> None:
    err = UnsupportedFormatError(path="/x.xyz", ext=".xyz")
    assert ".xyz" in err.message
    assert ".wav" in err.message  # must list at least one supported format


def test_engine_unavailable_without_name() -> None:
    err = EngineUnavailableError(kind="stt")
    assert "STT" in err.message
    assert err.details["engine_type"] == "stt"
    assert err.details["engine_name"] is None
