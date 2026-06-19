"""Structured errors for stt2tts-mcp.

Every error raised by tool handlers is a subclass of `ToolError`, which carries:
- a stable `code` (machine-readable, used by clients for routing)
- a human `message`
- optional `details` dict (path attempted, retry-after seconds, etc.)

`to_tool_result()` returns an MCP `CallToolResult` with `isError=True` and a
structured `structuredContent` block — the MCP 2.0 (2025-11-25) way to return
actionable errors that language models can use for self-correction.

Spec reference: https://modelcontextprotocol.io/specification/2025-11-25/server/tools
> "Tool Execution Errors: Reported in tool results with `isError: true`:
>    * API failures
>    * Input validation errors (e.g., date in wrong format, value out of range)
>    * Business logic errors
> Tool Execution Errors contain actionable feedback that language models can
> use to self-correct and retry with adjusted parameters."
"""

from __future__ import annotations

from typing import Any

from mcp.types import CallToolResult, TextContent


class ToolError(Exception):
    """Base class for all structured tool errors.

    Subclasses define the `code` (stable identifier) and may override the
    default human message. `details` is merged into `structuredContent` so
    clients can react programmatically.
    """

    code: str = "internal_error"
    default_message: str = "An internal error occurred"

    def __init__(
        self,
        message: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.message = message or self.default_message
        self.details = details or {}
        super().__init__(self.message)

    def to_tool_result(self) -> CallToolResult:
        """Serialize as an MCP tool execution error.

        MCP 2.0 (2025-11-25) requires `isError: true` for tool execution errors
        (vs protocol errors which use JSON-RPC error codes). We also return a
        `structuredContent` block so clients can route on `code` without
        parsing the human-readable text.
        """
        return CallToolResult(
            content=[TextContent(type="text", text=f"[{self.code}] {self.message}")],
            structuredContent={
                "ok": False,
                "error": {
                    "code": self.code,
                    "message": self.message,
                    "details": self.details,
                },
            },
            isError=True,
        )


# ---------------------------------------------------------------------------
# Input validation errors
# ---------------------------------------------------------------------------


class InvalidPathError(ToolError):
    code = "invalid_path"
    default_message = "Path is invalid, not allowed, or fails safety checks"

    def __init__(
        self, path: str, reason: str, details: dict[str, Any] | None = None
    ) -> None:
        merged = {"path": path, "reason": reason}
        if details:
            merged.update(details)
        super().__init__(message=f"Invalid path '{path}': {reason}", details=merged)


class TextTooLongError(ToolError):
    code = "text_too_long"
    default_message = "Input text exceeds the configured maximum length"

    def __init__(self, length: int, max_length: int) -> None:
        super().__init__(
            message=f"Text length {length} exceeds maximum of {max_length} characters",
            details={"length": length, "max_length": max_length},
        )


class EmptyTextError(ToolError):
    code = "empty_text"
    default_message = "Input text is empty"

    def __init__(self) -> None:
        super().__init__(message="Text must not be empty", details={"length": 0})


class InvalidAudioError(ToolError):
    code = "invalid_audio"
    default_message = "Audio file is invalid or unsupported"

    def __init__(self, path: str, reason: str) -> None:
        super().__init__(
            message=f"Audio file '{path}' is invalid: {reason}",
            details={"path": path, "reason": reason},
        )


class UnsupportedFormatError(ToolError):
    code = "unsupported_format"
    default_message = "File extension is not a supported audio format"

    SUPPORTED = (".wav", ".mp3", ".ogg", ".flac", ".m4a", ".wma", ".aac")

    def __init__(self, path: str, ext: str) -> None:
        super().__init__(
            message=f"Format '{ext}' is not supported. Use one of: {', '.join(self.SUPPORTED)}",
            details={"path": path, "extension": ext, "supported": list(self.SUPPORTED)},
        )


# ---------------------------------------------------------------------------
# Resource limit errors
# ---------------------------------------------------------------------------


class RateLimitError(ToolError):
    code = "rate_limited"
    default_message = "Rate limit exceeded"

    def __init__(self, tool: str, retry_after: float) -> None:
        super().__init__(
            message=f"Rate limit exceeded for tool '{tool}'. Retry after {retry_after:.2f}s",
            details={"tool": tool, "retry_after": retry_after},
        )


class OutputTooLargeError(ToolError):
    code = "output_too_large"
    default_message = "Generated output would exceed the configured size cap"

    def __init__(self, estimated_bytes: int, max_bytes: int) -> None:
        super().__init__(
            message=(
                f"Estimated output size {estimated_bytes:,} bytes "
                f"exceeds maximum of {max_bytes:,} bytes"
            ),
            details={"estimated_bytes": estimated_bytes, "max_bytes": max_bytes},
        )


# ---------------------------------------------------------------------------
# Engine / runtime errors
# ---------------------------------------------------------------------------


class EngineUnavailableError(ToolError):
    code = "engine_unavailable"
    default_message = "Required engine is not enabled or not available"

    def __init__(self, kind: str, name: str | None = None) -> None:
        msg = (
            f"No {kind.upper()} engine is enabled"
            if name is None
            else f"{kind.upper()} engine '{name}' is not available"
        )
        super().__init__(
            message=msg, details={"engine_type": kind, "engine_name": name}
        )


class UnknownToolError(ToolError):
    code = "unknown_tool"
    default_message = "Unknown tool name"

    def __init__(self, name: str) -> None:
        super().__init__(message=f"Unknown tool: {name}", details={"tool": name})


class ConfigError(ToolError):
    code = "config_error"
    default_message = "Configuration is invalid or missing"


__all__ = [
    "ToolError",
    "InvalidPathError",
    "TextTooLongError",
    "EmptyTextError",
    "InvalidAudioError",
    "UnsupportedFormatError",
    "RateLimitError",
    "OutputTooLargeError",
    "EngineUnavailableError",
    "UnknownToolError",
    "ConfigError",
]
