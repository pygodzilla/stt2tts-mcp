"""Central guard wrapper for tool handlers.

`guarded_call()` is the single place that enforces all Sprint 1 safeguards
for a tool call:

1. Rate limiting (per-tool token bucket)
2. Argument redaction before audit
3. Time the call (for audit + the duration_ms result field)
4. Catch `ToolError` and convert to `CallToolResult(isError=True)`
5. Catch unexpected exceptions and convert to a structured internal_error
6. Log every invocation to the audit logger

This keeps individual tool handlers focused on their happy-path logic;
they just raise `ToolError` on any guard violation and the wrapper turns
it into the correct MCP-shaped response.

MCP 2.0 spec ref: https://modelcontextprotocol.io/specification/2025-11-25/server/tools#error-handling
> "Tool Execution Errors: Reported in tool results with `isError: true`"
"""

from __future__ import annotations

import asyncio
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from mcp.types import CallToolResult, TextContent

from stt2tts_mcp.security.audit import AuditLogger, get_audit_logger
from stt2tts_mcp.security.errors import ToolError
from stt2tts_mcp.security.rate_limit import RateLimiter


@dataclass
class GuardConfig:
    """Per-process configuration for the guard layer.

    Pass `None` for any field to use the default. Defaults are intentionally
    permissive so a fresh install "just works"; enterprise deployments can
    tighten via config.yaml (see Sprint 2).
    """

    rate_limiter: RateLimiter | None = None
    audit_logger: AuditLogger | None = None
    enable_audit: bool = True
    # Tools not in this set skip rate limiting (cheap read tools).
    unthrottled_tools: set[str] = field(
        default_factory=lambda: {
            "health_check",
            "list_stt_models",
            "list_tts_voices",
            "reload_config",
        }
    )

    def get_limiter(self) -> RateLimiter:
        return self.rate_limiter or RateLimiter()

    def get_audit(self) -> AuditLogger:
        if not self.enable_audit:
            # Return a disabled logger so call sites don't need to branch.
            return AuditLogger(enabled=False)
        return self.audit_logger or get_audit_logger()


# ---------------------------------------------------------------------------
# Result helpers
# ---------------------------------------------------------------------------


def _text_result(text: str, structured: dict[str, Any] | None = None) -> CallToolResult:
    """Build a successful CallToolResult with optional structured content.

    MCP 2.0 (2025-11-25) — "Structured content" is returned in the
    structuredContent field of a result. For backwards compatibility, a tool
    that returns structured content SHOULD also return the serialized JSON
    in a TextContent block.
    """
    if structured is None:
        return CallToolResult(
            content=[TextContent(type="text", text=text)],
            isError=False,
        )
    import json

    return CallToolResult(
        content=[
            TextContent(type="text", text=text),
            TextContent(type="text", text=json.dumps(structured, indent=2)),
        ],
        structuredContent=structured,
        isError=False,
    )


# ---------------------------------------------------------------------------
# Core guard
# ---------------------------------------------------------------------------


async def guarded_call(
    name: str,
    arguments: dict[str, Any],
    handler: Callable[
        [dict[str, Any]], Awaitable[CallToolResult | list[TextContent] | str | dict]
    ],
    config: GuardConfig | None = None,
) -> CallToolResult:
    """Run `handler` with rate limiting, audit, error mapping, and timing.

    The handler may return:
    - `CallToolResult` — passed through unchanged
    - `list[TextContent]` — wrapped into a success CallToolResult
    - `str` — wrapped into a single TextContent success
    - `dict` — wrapped as structuredContent (MCP 2.0)
    """
    cfg = config or GuardConfig()
    audit = cfg.get_audit()
    limiter = cfg.get_limiter()

    args_for_audit = dict(arguments or {})

    # 1. Rate limit (skipped for cheap read tools).
    if name not in cfg.unthrottled_tools:
        limiter.consume(name)

    start = time.monotonic()
    ok = False
    error_payload: dict[str, Any] | None = None
    result_payload: dict[str, Any] | None = None
    response: CallToolResult | None = None

    try:
        raw = await handler(arguments or {})

        if isinstance(raw, CallToolResult):
            response = raw
        elif isinstance(raw, list):
            response = _text_result(
                text="\n".join(
                    c.text for c in raw if getattr(c, "type", None) == "text"
                ),
            )
        elif isinstance(raw, str):
            response = _text_result(text=raw)
        elif isinstance(raw, dict):
            response = _text_result(
                text=raw.get("message", "") or str(raw),
                structured=raw,
            )
        else:
            response = _text_result(text=str(raw))

        # If the handler returned its own isError=True, propagate that
        # instead of treating it as success.
        ok = not bool(getattr(response, "isError", False))
        if ok:
            # Capture a compact summary for the audit log. Avoid dumping the
            # whole structured content to keep logs small.
            sc = getattr(response, "structuredContent", None)
            if isinstance(sc, dict):
                result_payload = {
                    "keys": sorted(sc.keys()),
                    "is_error": False,
                }
        else:
            sc = getattr(response, "structuredContent", None)
            if isinstance(sc, dict) and isinstance(sc.get("error"), dict):
                error_payload = sc["error"]
            else:
                error_payload = {
                    "code": "handler_error",
                    "message": " ".join(
                        c.text
                        for c in (response.content or [])
                        if getattr(c, "type", None) == "text"
                    )[:500],
                }

    except ToolError as exc:
        response = exc.to_tool_result()
        error_payload = {
            "code": exc.code,
            "message": exc.message,
            "details": exc.details,
        }
    except asyncio.CancelledError:
        # Don't swallow cancellation — let it propagate.
        raise
    except Exception as exc:  # noqa: BLE001 — last-resort safety net
        # Unexpected: still return a structured error, never crash the server.
        tb = traceback.format_exc(limit=3)
        response = _text_result(
            text=f"[internal_error] {exc}",
            structured={
                "ok": False,
                "error": {
                    "code": "internal_error",
                    "message": str(exc),
                    "details": {"traceback": tb[-1000:]},
                },
            },
        )
        # Mark as error by swapping isError
        response = CallToolResult(
            content=response.content,
            structuredContent=response.structuredContent,
            isError=True,
        )
        error_payload = {"code": "internal_error", "message": str(exc)}
    finally:
        duration_ms = (time.monotonic() - start) * 1000.0
        audit.log(
            tool=name,
            args=args_for_audit,
            ok=ok,
            duration_ms=duration_ms,
            result=result_payload,
            error=error_payload,
        )

    # At this point `response` is guaranteed set.
    assert response is not None
    return response


__all__ = ["GuardConfig", "guarded_call"]
