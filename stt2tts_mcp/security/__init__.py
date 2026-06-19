"""Security & enterprise-readiness utilities for stt2tts-mcp.

Public surface:
- errors:   ToolError hierarchy with stable machine-readable codes
- validation: input sanitization + path safety checks
- rate_limit: per-tool token-bucket rate limiter
- audit:    append-only JSONL audit logging
- guards:   `guarded_call()` — the central wrapper for tool handlers
"""

from stt2tts_mcp.security.audit import (
    AuditLogger,
    DEFAULT_AUDIT_PATH,
    get_audit_logger,
    reset_audit_logger,
)
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
from stt2tts_mcp.security.guards import (  # noqa: F401  (GuardConfig re-exported below)
    GuardConfig,
    guarded_call,
)
from stt2tts_mcp.security.rate_limit import RateLimiter
from stt2tts_mcp.security.validation import (
    estimate_tts_output_bytes,
    safe_resolve_path,
    sanitize_text,
)

__all__ = [
    # Errors
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
    # Validation
    "sanitize_text",
    "safe_resolve_path",
    "estimate_tts_output_bytes",
    # Rate limiting
    "RateLimiter",
    # Audit
    "AuditLogger",
    "get_audit_logger",
    "reset_audit_logger",
    "DEFAULT_AUDIT_PATH",
    # Central guard
    "GuardedCall" if False else "GuardConfig",  # avoid flake8 E501 long-line
    "guarded_call",
]
