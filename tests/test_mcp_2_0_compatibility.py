"""Tests for MCP 2.0 (2025-11-25) features on the stt2tts-mcp server.

These tests verify the schema and annotation declarations at module-load
time (no subprocess required). They guard against accidentally removing
MCP 2.0 features (title, annotations, outputSchema, structuredContent,
Server version) during refactors.
"""

from __future__ import annotations

import pytest

from stt2tts_mcp import __version__
from stt2tts_mcp.server import (
    HEALTH_CHECK_TOOL,
    LIST_STT_MODELS_TOOL,
    LIST_TTS_VOICES_TOOL,
    RELOAD_CONFIG_TOOL,
    SPEAK_TOOL,
    TRANSCRIBE_TOOL,
    server,
)


# ---------------------------------------------------------------------------
# Server-level MCP 2.0 features
# ---------------------------------------------------------------------------


def test_server_advertises_version() -> None:
    """MCP 2.0 (2025-11-25): servers SHOULD declare their version."""
    assert server.version == __version__
    assert server.version, "Server must declare a version"


def test_server_advertises_instructions() -> None:
    """MCP 2.0: Server(instructions=...) is shown to LLMs as guidance."""
    assert server.instructions, "Server must declare instructions for LLMs"
    assert "transcribe" in server.instructions.lower()
    assert "speak" in server.instructions.lower()


# ---------------------------------------------------------------------------
# Per-tool MCP 2.0 features
# ---------------------------------------------------------------------------


def test_all_tools_have_title() -> None:
    """MCP 2.0: `title` is the human-readable display name (optional but recommended)."""
    for tool in (
        TRANSCRIBE_TOOL,
        SPEAK_TOOL,
        LIST_STT_MODELS_TOOL,
        LIST_TTS_VOICES_TOOL,
        RELOAD_CONFIG_TOOL,
        HEALTH_CHECK_TOOL,
    ):
        assert tool.title, f"{tool.name} missing title (MCP 2.0)"


def test_all_tools_have_annotations() -> None:
    """MCP 2.0: ToolAnnotations are required for trust & safety."""
    for tool in (
        TRANSCRIBE_TOOL,
        SPEAK_TOOL,
        LIST_STT_MODELS_TOOL,
        LIST_TTS_VOICES_TOOL,
        RELOAD_CONFIG_TOOL,
        HEALTH_CHECK_TOOL,
    ):
        ann = tool.annotations
        assert ann is not None, f"{tool.name} missing annotations"
        assert ann.readOnlyHint is not None, f"{tool.name} missing readOnlyHint"
        assert ann.destructiveHint is not None, f"{tool.name} missing destructiveHint"
        assert ann.idempotentHint is not None, f"{tool.name} missing idempotentHint"


def test_readonly_tools_have_readonly_hint() -> None:
    """Read-only tools (transcribe, list_*, health_check) must declare readOnlyHint=True."""
    for tool in (
        TRANSCRIBE_TOOL,
        LIST_STT_MODELS_TOOL,
        LIST_TTS_VOICES_TOOL,
        HEALTH_CHECK_TOOL,
    ):
        assert tool.annotations.readOnlyHint is True, (
            f"{tool.name} should be readOnlyHint=True"
        )


def test_speak_marks_destructive_appropriately() -> None:
    """speak writes a file → readOnlyHint=False, destructiveHint=False (overwrites only)."""
    assert SPEAK_TOOL.annotations.readOnlyHint is False
    assert SPEAK_TOOL.annotations.destructiveHint is False
    # It's idempotent (same input → same WAV).
    assert SPEAK_TOOL.annotations.idempotentHint is True


def test_reload_config_marks_side_effect() -> None:
    """reload_config mutates global state → readOnlyHint=False."""
    assert RELOAD_CONFIG_TOOL.annotations.readOnlyHint is False


# ---------------------------------------------------------------------------
# outputSchema (MCP 2.0 structured outputs)
# ---------------------------------------------------------------------------


def test_transcribe_declares_output_schema() -> None:
    """MCP 2.0: outputSchema lets clients validate structuredContent."""
    assert TRANSCRIBE_TOOL.outputSchema is not None
    props = TRANSCRIBE_TOOL.outputSchema["properties"]
    assert "ok" in props
    assert "engine" in props
    assert "text" in props
    assert "text_length" in props


def test_speak_declares_output_schema() -> None:
    assert SPEAK_TOOL.outputSchema is not None
    props = SPEAK_TOOL.outputSchema["properties"]
    assert "ok" in props
    assert "engine" in props
    assert "audio_path" in props
    assert "text_length" in props
    assert "dry_run" in props


def test_health_check_declares_output_schema() -> None:
    assert HEALTH_CHECK_TOOL.outputSchema is not None
    props = HEALTH_CHECK_TOOL.outputSchema["properties"]
    assert "ok" in props
    assert "stt" in props
    assert "tts" in props


def test_input_schemas_use_draft_2020_12() -> None:
    """MCP 2.0 (2025-11-25): JSON Schema 2020-12 is the default dialect."""
    for tool in (
        TRANSCRIBE_TOOL,
        SPEAK_TOOL,
        LIST_STT_MODELS_TOOL,
        LIST_TTS_VOICES_TOOL,
        RELOAD_CONFIG_TOOL,
        HEALTH_CHECK_TOOL,
    ):
        # Either explicit $schema field OR 2020-12 default (no $schema).
        # Either way the schemas must be valid JSON Schema objects.
        schema = tool.inputSchema
        assert isinstance(schema, dict)
        assert schema.get("type") == "object"
        if "$schema" in schema:
            assert "2020-12" in schema["$schema"]


def test_input_schemas_disallow_extra_properties() -> None:
    """MCP 2.0 best practice: additionalProperties:false on tool schemas."""
    for tool in (
        TRANSCRIBE_TOOL,
        SPEAK_TOOL,
        LIST_STT_MODELS_TOOL,
        LIST_TTS_VOICES_TOOL,
        RELOAD_CONFIG_TOOL,
        HEALTH_CHECK_TOOL,
    ):
        assert tool.inputSchema.get("additionalProperties") is False, (
            f"{tool.name} should set additionalProperties: false"
        )


def test_empty_args_tools_have_empty_object_schema() -> None:
    """Tools with no args (health_check, reload_config, list_*) use empty object schema."""
    for tool in (
        LIST_STT_MODELS_TOOL,
        LIST_TTS_VOICES_TOOL,
        RELOAD_CONFIG_TOOL,
        HEALTH_CHECK_TOOL,
    ):
        schema = tool.inputSchema
        assert schema["type"] == "object"
        assert schema.get("properties") == {} or "properties" not in schema
        assert schema.get("required") is None or schema.get("required") == []


# ---------------------------------------------------------------------------
# Tool names conform to MCP 2.0 naming guidance
# ---------------------------------------------------------------------------


def test_tool_names_meet_mcp_2_0_guidelines() -> None:
    """SEP-986: names 1-128 chars, ASCII letters/digits/_-. only, no spaces."""
    import re

    for tool in (
        TRANSCRIBE_TOOL,
        SPEAK_TOOL,
        LIST_STT_MODELS_TOOL,
        LIST_TTS_VOICES_TOOL,
        RELOAD_CONFIG_TOOL,
        HEALTH_CHECK_TOOL,
    ):
        name = tool.name
        assert 1 <= len(name) <= 128
        assert re.match(r"^[A-Za-z0-9_\-\.]+$", name), (
            f"{name!r} contains illegal characters (per MCP 2.0 SEP-986)"
        )


# ---------------------------------------------------------------------------
# speak: dry-run option
# ---------------------------------------------------------------------------


def test_speak_supports_dry_run() -> None:
    """Sprint 1 A4: speak accepts `dry_run` for safe previews."""
    assert "dry_run" in SPEAK_TOOL.inputSchema["properties"]
    assert SPEAK_TOOL.inputSchema["properties"]["dry_run"]["type"] == "boolean"
    assert SPEAK_TOOL.inputSchema["properties"]["dry_run"]["default"] is False
