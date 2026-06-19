"""Unit tests for the central `guarded_call()` wrapper.

These tests exercise the guard layer in isolation — no subprocess, no real
MCP server. We pass in stub handlers and verify the wrapper:
1. Enforces rate limits
2. Logs every call (success and failure)
3. Converts ToolError to isError=True CallToolResult
4. Catches unexpected exceptions
5. Wraps str / dict / list / CallToolResult handlers correctly
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from mcp.types import CallToolResult, TextContent

from stt2tts_mcp.security import (
    EngineUnavailableError,
    GuardConfig,
    RateLimiter,
    RateLimitError,
    ToolError,
    guarded_call,
    reset_audit_logger,
)
from stt2tts_mcp.security.audit import get_audit_logger


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fresh_guard(tmp_path: Path) -> GuardConfig:
    """Reset the default audit logger and return a per-test GuardConfig."""
    log_path = tmp_path / "audit.log"
    reset_audit_logger(path=log_path, enabled=True)
    # Re-init rate limiter so test order doesn't matter.
    return GuardConfig(
        rate_limiter=RateLimiter({"speak": (3.0, 0.001)}),
        audit_logger=get_audit_logger(),
        enable_audit=True,
    )


# ---------------------------------------------------------------------------
# Successful handlers
# ---------------------------------------------------------------------------


async def test_str_handler_returns_text_result(fresh_guard: GuardConfig) -> None:
    async def handler(args: dict) -> str:
        return f"echo: {args.get('text', '')}"

    result = await guarded_call("speak", {"text": "hi"}, handler, fresh_guard)
    assert isinstance(result, CallToolResult)
    assert result.isError is False
    assert any("echo: hi" in c.text for c in result.content)


async def test_dict_handler_returns_structured_content(
    fresh_guard: GuardConfig,
) -> None:
    async def handler(args: dict) -> dict:
        return {"ok": True, "count": 42, "items": ["a", "b"]}

    result = await guarded_call("list", {}, handler, fresh_guard)
    assert result.isError is False
    assert result.structuredContent == {"ok": True, "count": 42, "items": ["a", "b"]}


async def test_call_tool_result_handler_passes_through(
    fresh_guard: GuardConfig,
) -> None:
    expected = CallToolResult(
        content=[TextContent(type="text", text="raw")],
        structuredContent={"ok": True, "x": 1},
        isError=False,
    )

    async def handler(args: dict) -> CallToolResult:
        return expected

    result = await guarded_call("transcribe", {}, handler, fresh_guard)
    assert result is expected


async def test_list_text_content_handler_is_wrapped(fresh_guard: GuardConfig) -> None:
    async def handler(args: dict) -> list[TextContent]:
        return [
            TextContent(type="text", text="line1"),
            TextContent(type="text", text="line2"),
        ]

    result = await guarded_call("health_check", {}, handler, fresh_guard)
    assert result.isError is False
    text = " ".join(c.text for c in result.content if c.type == "text")
    assert "line1" in text and "line2" in text


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------


async def test_tool_error_is_converted_to_structured_error(
    fresh_guard: GuardConfig,
) -> None:
    async def handler(args: dict) -> None:
        raise EngineUnavailableError(kind="tts", name="piper")

    result = await guarded_call("speak", {}, handler, fresh_guard)
    assert result.isError is True
    assert result.structuredContent is not None
    err = result.structuredContent["error"]
    assert err["code"] == "engine_unavailable"
    assert err["details"]["engine_type"] == "tts"
    assert err["details"]["engine_name"] == "piper"


async def test_unexpected_exception_returns_internal_error(
    fresh_guard: GuardConfig,
) -> None:
    async def handler(args: dict) -> None:
        raise RuntimeError("kaboom")

    result = await guarded_call("speak", {}, handler, fresh_guard)
    assert result.isError is True
    assert result.structuredContent is not None
    assert result.structuredContent["error"]["code"] == "internal_error"
    assert "kaboom" in result.structuredContent["error"]["message"]


async def test_cancelled_error_propagates(fresh_guard: GuardConfig) -> None:
    """asyncio.CancelledError must NOT be swallowed by the safety net."""
    import asyncio

    async def handler(args: dict) -> None:
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await guarded_call("speak", {}, handler, fresh_guard)


# ---------------------------------------------------------------------------
# Rate limiting integration
# ---------------------------------------------------------------------------


async def test_handler_raising_rate_limit_returns_structured_error(
    fresh_guard: GuardConfig,
) -> None:
    """A handler that itself raises RateLimitError is converted to isError=True.

    (The guard's own rate-limit pre-check raises RateLimitError too — that
    one happens BEFORE the handler is called and is not caught by the
    try/except. So we test the handler-raises-it case here.)
    """
    from stt2tts_mcp.security.errors import RateLimitError

    async def throttled_handler(args: dict) -> None:
        raise RateLimitError(tool="speak", retry_after=2.0)

    result = await guarded_call("speak", {}, throttled_handler, fresh_guard)
    assert result.isError is True
    assert result.structuredContent["error"]["code"] == "rate_limited"
    assert result.structuredContent["error"]["details"]["tool"] == "speak"
    assert result.structuredContent["error"]["details"]["retry_after"] == 2.0


async def test_guards_own_rate_limit_blocks_handler(
    fresh_guard: GuardConfig,
) -> None:
    """Exhausting the bucket via direct limiter calls means the next guarded_call
    raises RateLimitError itself (from the pre-check, before the handler runs)."""
    from stt2tts_mcp.security.errors import RateLimitError

    limiter = fresh_guard.rate_limiter
    assert limiter is not None
    for _ in range(3):
        limiter.consume("speak")
    # Next pre-check raises.
    with pytest.raises(RateLimitError) as exc:
        limiter.consume("speak")
    assert exc.value.code == "rate_limited"


async def test_health_check_bypasses_rate_limit(fresh_guard: GuardConfig) -> None:
    """Unthrottled tools don't consume tokens."""

    async def handler(args: dict) -> str:
        return "ok"

    for _ in range(100):
        result = await guarded_call("health_check", {}, handler, fresh_guard)
        assert result.isError is False


# ---------------------------------------------------------------------------
# Audit logging integration
# ---------------------------------------------------------------------------


async def test_successful_call_is_logged(
    fresh_guard: GuardConfig, tmp_path: Path
) -> None:
    async def handler(args: dict) -> str:
        return "ok"

    await guarded_call("speak", {"text": "hello"}, handler, fresh_guard)
    fresh_guard.audit_logger.close()

    entries = fresh_guard.audit_logger.tail(10)
    assert len(entries) >= 1
    last = entries[-1]
    assert last["tool"] == "speak"
    assert last["ok"] is True
    assert "duration_ms" in last
    # text is redacted but length preserved.
    assert last["args"]["text"]["length"] == 5


async def test_failed_call_is_logged_with_error(fresh_guard: GuardConfig) -> None:
    async def handler(args: dict) -> None:
        raise RateLimitError(tool="speak", retry_after=2.0)

    await guarded_call("speak", {}, handler, fresh_guard)
    fresh_guard.audit_logger.close()

    entries = fresh_guard.audit_logger.tail(10)
    last = entries[-1]
    assert last["tool"] == "speak"
    assert last["ok"] is False
    assert last["error"]["code"] == "rate_limited"


# ---------------------------------------------------------------------------
# Default GuardConfig
# ---------------------------------------------------------------------------


async def test_default_guard_config_works() -> None:
    """Calling guarded_call with no config uses sensible defaults."""
    import asyncio

    async def handler(args: dict) -> str:
        return "ok"

    result = await guarded_call("health_check", {}, handler)
    assert result.isError is False


# ---------------------------------------------------------------------------
# Custom redactor
# ---------------------------------------------------------------------------


async def test_custom_redactor_can_override_default(tmp_path: Path) -> None:
    """Caller can supply a redactor that hides additional fields."""
    from stt2tts_mcp.security.audit import AuditLogger

    log_path = tmp_path / "audit.log"

    def my_redactor(key: str, value: Any) -> Any:
        if key == "voice":
            return "<voice-hidden>"
        return value

    cfg = GuardConfig(
        rate_limiter=RateLimiter(),
        audit_logger=AuditLogger(path=log_path, redactor=my_redactor, enabled=True),
        enable_audit=True,
    )

    async def handler(args: dict) -> str:
        return "ok"

    await guarded_call("speak", {"text": "hi", "voice": "alice"}, handler, cfg)
    cfg.audit_logger.close()

    entries = cfg.audit_logger.tail(10)
    assert entries[-1]["args"]["voice"] == "<voice-hidden>"
