"""STT2TTS MCP Server — exposes STT and TTS tools via MCP protocol.

This server is MCP 2.0 (protocol version 2025-11-25) ready. It uses:
- `Tool.title` for human-readable display names
- `Tool.annotations` for trust & safety hints (readOnlyHint, etc.)
- `Tool.execution` to declare task-support policy
- `CallToolResult.structuredContent` for machine-readable outputs
- `Server(instructions=...)` for server-level guidance shown to LLMs

Spec reference:
- https://modelcontextprotocol.io/specification/2025-11-25
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolResult,
    TextContent,
    Tool,
    ToolAnnotations,
    ToolExecution,
)

from stt2tts_mcp import __version__
from stt2tts_mcp.engines import STT_ENGINES, TTS_ENGINES, EngineConfig
from stt2tts_mcp.security import (
    EngineUnavailableError,
    GuardConfig,
    OutputTooLargeError,
    RateLimiter,
    estimate_tts_output_bytes,
    get_audit_logger,
    guarded_call,
    safe_resolve_path,
    sanitize_text,
)
from stt2tts_mcp.security.errors import ToolError
from stt2tts_mcp.utils import (
    convert_to_wav,
    get_stt_config,
    get_tts_config,
    load_config,
    reload_config as _reload_config_yaml,
)


# ---------------------------------------------------------------------------
# Server instance — MCP 2.0 ready
# ---------------------------------------------------------------------------

SERVER_NAME = "stt2tts-mcp"
SERVER_INSTRUCTIONS = (
    "STT2TTS exposes speech-to-text and text-to-speech tools. "
    "Prefer `transcribe` for audio-to-text and `speak` for text-to-audio. "
    "`health_check`, `list_stt_models`, `list_tts_voices`, and `reload_config` "
    "are read/admin tools. Engines and voices are selected via config.yaml; "
    "call `reload_config` after editing it. Audio formats accepted: wav, mp3, "
    "ogg, flac, m4a, wma, aac. Output is always 16kHz mono PCM WAV for "
    "transcription; speak() writes a WAV at the engine's native sample rate."
)

server: Server = Server(
    SERVER_NAME,
    version=__version__,
    instructions=SERVER_INSTRUCTIONS,
)


# ---------------------------------------------------------------------------
# Engine lifecycle (unchanged from prior version — locked behind guards)
# ---------------------------------------------------------------------------

_stt_engine = None
_tts_engine = None


def _build_stt_engine() -> None:
    global _stt_engine
    config = load_config()
    stt_cfg = get_stt_config(config)
    if stt_cfg is None:
        _stt_engine = None
        return
    engine_name = stt_cfg.get("engine", "faster_whisper")
    if engine_name not in STT_ENGINES:
        raise EngineUnavailableError(kind="stt", name=engine_name)
    params = stt_cfg.get("params", {})
    engine_config = EngineConfig(name=engine_name, enabled=True, params=params)
    _stt_engine = STT_ENGINES[engine_name](engine_config)


def _build_tts_engine() -> None:
    global _tts_engine
    config = load_config()
    tts_cfg = get_tts_config(config)
    if tts_cfg is None:
        _tts_engine = None
        return
    engine_name = tts_cfg.get("engine", "piper")
    if engine_name not in TTS_ENGINES:
        raise EngineUnavailableError(kind="tts", name=engine_name)
    params = tts_cfg.get("params", {})
    engine_config = EngineConfig(name=engine_name, enabled=True, params=params)
    _tts_engine = TTS_ENGINES[engine_name](engine_config)


def _ensure_stt() -> None:
    global _stt_engine
    if _stt_engine is None:
        _build_stt_engine()


def _ensure_tts() -> None:
    global _tts_engine
    if _tts_engine is None:
        _build_tts_engine()


# ---------------------------------------------------------------------------
# Tool definitions — MCP 2.0 annotated
# ---------------------------------------------------------------------------

AUDIO_EXTS = (".wav", ".mp3", ".ogg", ".flac", ".m4a", ".wma", ".aac")


TRANSCRIBE_TOOL = Tool(
    name="transcribe",
    title="Audio Transcription",
    description=(
        "Transcribe an audio file to text using the active STT engine. "
        "Supports faster-whisper, sherpa-onnx, OpenAI API, Ollama, and LMStudio "
        "backends. Engine is selected via config.yaml. ffmpeg is used for format "
        "conversion to 16kHz mono PCM before transcription."
    ),
    inputSchema={
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "audio_path": {
                "type": "string",
                "description": (
                    "Path to the audio file (wav, mp3, ogg, flac, m4a, wma, aac). "
                    "Path must point to an existing file inside an allowed root "
                    "directory (default: any path the user can read)."
                ),
            },
            "language": {
                "type": "string",
                "description": "Language code (e.g., 'en', 'es', 'fr'). Defaults to config value or 'en'.",
            },
            "task": {
                "type": "string",
                "enum": ["transcribe", "translate"],
                "default": "transcribe",
                "description": "'transcribe' (default) or 'translate' (to English).",
            },
        },
        "required": ["audio_path"],
        "additionalProperties": False,
    },
    outputSchema={
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "ok": {"type": "boolean"},
            "engine": {"type": "string"},
            "language": {"type": ["string", "null"]},
            "duration": {"type": ["number", "null"]},
            "text": {"type": "string"},
            "text_length": {"type": "integer"},
            "duration_ms": {"type": "number"},
        },
        "required": ["ok", "engine", "text", "text_length"],
    },
    annotations=ToolAnnotations(
        title="Audio Transcription",
        readOnlyHint=True,  # doesn't mutate user files
        destructiveHint=False,
        idempotentHint=True,  # same input → same output
        openWorldHint=False,  # doesn't reach external services by default
    ),
    execution=ToolExecution(taskSupport="forbidden"),  # synchronous only
)

SPEAK_TOOL = Tool(
    name="speak",
    title="Text-to-Speech Synthesis",
    description=(
        "Synthesize text to speech and save as a WAV file. Supports Piper, Kokoro, "
        "Coqui, OpenAI API, Ollama, and LMStudio backends. Engine and voice are "
        "selected via config.yaml. Pass `dry_run: true` to preview without writing "
        "any files."
    ),
    inputSchema={
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": (
                    "Text to synthesize. Control characters are stripped; "
                    "length is capped (see config.yaml → `limits.max_text_chars`)."
                ),
            },
            "output_path": {
                "type": "string",
                "description": (
                    "Path to save the output WAV file. Defaults to a temp file. "
                    "Parent directory must already exist."
                ),
            },
            "voice": {
                "type": "string",
                "description": "Voice name override (e.g., 'en_US-lessac-medium'). Defaults to config value.",
            },
            "dry_run": {
                "type": "boolean",
                "default": False,
                "description": (
                    "If true, do not synthesize — just return the parameters "
                    "that would have been used. Useful for confirmation flows."
                ),
            },
        },
        "required": ["text"],
        "additionalProperties": False,
    },
    outputSchema={
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "ok": {"type": "boolean"},
            "engine": {"type": "string"},
            "duration": {"type": ["number", "null"]},
            "audio_path": {"type": ["string", "null"]},
            "text_length": {"type": "integer"},
            "duration_ms": {"type": "number"},
            "dry_run": {"type": "boolean"},
        },
        "required": ["ok", "engine", "text_length", "dry_run"],
    },
    annotations=ToolAnnotations(
        title="Text-to-Speech Synthesis",
        readOnlyHint=False,  # writes a file
        destructiveHint=False,  # overwrites the named output_path only
        idempotentHint=True,  # same input → same WAV bytes
        openWorldHint=False,
    ),
    execution=ToolExecution(taskSupport="forbidden"),
)

LIST_STT_MODELS_TOOL = Tool(
    name="list_stt_models",
    title="List STT Models",
    description="List available STT models for the active engine.",
    inputSchema={
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
    execution=ToolExecution(taskSupport="forbidden"),
)

LIST_TTS_VOICES_TOOL = Tool(
    name="list_tts_voices",
    title="List TTS Voices",
    description="List available TTS voices for the active engine.",
    inputSchema={
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
    execution=ToolExecution(taskSupport="forbidden"),
)

RELOAD_CONFIG_TOOL = Tool(
    name="reload_config",
    title="Reload Configuration",
    description=(
        "Hot-reload config.yaml and rebuild engines. Use this after editing "
        "config.yaml to switch engines without restarting the server."
    ),
    inputSchema={
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
    annotations=ToolAnnotations(
        readOnlyHint=False,  # mutates global engine state
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
    execution=ToolExecution(taskSupport="forbidden"),
)

HEALTH_CHECK_TOOL = Tool(
    name="health_check",
    title="Health Check",
    description="Check health of active STT and TTS engines.",
    inputSchema={
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
    outputSchema={
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "ok": {"type": "boolean"},
            "stt": {
                "type": "object",
                "properties": {
                    "name": {"type": ["string", "null"]},
                    "initialized": {"type": "boolean"},
                    "healthy": {"type": "boolean"},
                },
                "required": ["initialized", "healthy"],
            },
            "tts": {
                "type": "object",
                "properties": {
                    "name": {"type": ["string", "null"]},
                    "initialized": {"type": "boolean"},
                    "healthy": {"type": "boolean"},
                },
                "required": ["initialized", "healthy"],
            },
        },
        "required": ["ok", "stt", "tts"],
    },
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
    execution=ToolExecution(taskSupport="forbidden"),
)


# ---------------------------------------------------------------------------
# Request handlers
# ---------------------------------------------------------------------------

_GUARD_CONFIG = GuardConfig(
    rate_limiter=RateLimiter(),
    audit_logger=get_audit_logger(),
    enable_audit=True,
)


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List all available MCP tools."""
    return [
        TRANSCRIBE_TOOL,
        SPEAK_TOOL,
        LIST_STT_MODELS_TOOL,
        LIST_TTS_VOICES_TOOL,
        RELOAD_CONFIG_TOOL,
        HEALTH_CHECK_TOOL,
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> CallToolResult:
    """Dispatch entry point — routes to per-tool handlers through the guard layer.

    The dispatch is wrapped in `guarded_call()` which provides:
    - Per-tool rate limiting (token bucket)
    - Append-only audit logging
    - Structured error responses (`isError: true` + `structuredContent.error`)
    - Catch-all for unexpected exceptions

    MCP 2.0 (2025-11-25): we return the full `CallToolResult` so the SDK
    preserves the `isError` flag, `structuredContent`, and `_meta` block.
    Returning just `result.content` would discard the error flag and the
    structured payload — defeating the entire guard layer.
    """
    handler = _DISPATCH.get(name)
    if handler is None:
        return CallToolResult(
            content=[
                TextContent(type="text", text=f"[unknown_tool] Unknown tool: {name}")
            ],
            structuredContent={
                "ok": False,
                "error": {
                    "code": "unknown_tool",
                    "message": f"Unknown tool: {name}",
                    "details": {"tool": name},
                },
            },
            isError=True,
        )
    return await guarded_call(
        name=name,
        arguments=arguments or {},
        handler=handler,
        config=_GUARD_CONFIG,
    )


# ---------------------------------------------------------------------------
# Per-tool handlers (return CallToolResult / dict / list[TextContent] / str)
# ---------------------------------------------------------------------------


def _max_text_chars() -> int:
    """Read the configured max text length (env override wins)."""
    raw = os.environ.get("STT2TTS_MAX_TEXT_CHARS")
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    # Fall back to config.yaml `limits.max_text_chars` if set, else default.
    try:
        cfg = load_config()
        return int(cfg.get("limits", {}).get("max_text_chars", 100_000))
    except Exception:  # noqa: BLE001 — config may not be present in tests
        return 100_000


def _max_audio_bytes() -> int:
    """Read the configured max output WAV size for `speak`."""
    raw = os.environ.get("STT2TTS_MAX_OUTPUT_BYTES")
    if raw:
        try:
            return max(1024, int(raw))
        except ValueError:
            pass
    try:
        cfg = load_config()
        return int(cfg.get("limits", {}).get("max_output_bytes", 100 * 1024 * 1024))
    except Exception:  # noqa: BLE001
        return 100 * 1024 * 1024


async def _handle_transcribe(args: dict) -> CallToolResult:
    audio_path_arg = args.get("audio_path")
    if not audio_path_arg:
        raise ToolError(
            message="Missing required argument: audio_path",
            details={"argument": "audio_path"},
        )

    # Path safety — resolve, check existence, check extension.
    resolved = safe_resolve_path(
        audio_path_arg,
        must_exist=True,
        require_file=True,
        allowed_extensions=AUDIO_EXTS,
    )

    language = args.get("language")
    task = args.get("task", "transcribe")
    if task not in ("transcribe", "translate"):
        raise ToolError(
            message=f"Invalid task: {task!r}",
            details={"allowed": ["transcribe", "translate"]},
        )

    # Convert to wav if needed.
    try:
        wav_path = convert_to_wav(resolved)
    except Exception as exc:  # noqa: BLE001
        from stt2tts_mcp.security.errors import InvalidAudioError

        raise InvalidAudioError(path=str(resolved), reason=str(exc)) from exc

    try:
        _ensure_stt()
        if _stt_engine is None:
            raise EngineUnavailableError(kind="stt")

        kwargs: dict[str, Any] = {}
        if language:
            kwargs["language"] = language
        if task:
            kwargs["task"] = task

        result = _stt_engine.transcribe(wav_path, **kwargs)
        text = (result.text or "").strip()

        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=(
                        f"Transcription ({result.engine})\n"
                        f"Language: {result.language or 'auto'}\n"
                        f"Duration: {result.duration:.2f}s\n"
                        f"Text: {text}"
                    ),
                )
            ],
            structuredContent={
                "ok": True,
                "engine": result.engine,
                "language": result.language,
                "duration": result.duration,
                "text": text,
                "text_length": len(text),
            },
            isError=False,
        )
    finally:
        # Cleanup the converted wav if it's not the source file.
        if wav_path != str(resolved):
            try:
                Path(wav_path).unlink(missing_ok=True)
            except OSError:
                pass


async def _handle_speak(args: dict) -> CallToolResult:
    text = args.get("text")
    if text is None:
        raise ToolError(
            message="Missing required argument: text",
            details={"argument": "text"},
        )

    # Sanitize + length cap.
    text = sanitize_text(text, max_length=_max_text_chars())

    output_path = args.get("output_path")
    voice = args.get("voice")
    dry_run = bool(args.get("dry_run", False))

    # Pre-allocate output path (or tempfile) — required even in dry-run so
    # we can report the would-be target.
    if output_path:
        # User-supplied path: validate it points to a writable location
        # (parent dir must exist). We do NOT create it; that's the engine's
        # job.
        out = Path(output_path).expanduser()
        parent = out.parent
        if not parent.exists():
            raise ToolError(
                message=f"output_path parent does not exist: {parent}",
                details={"parent": str(parent)},
            )
        output_path = str(out)
    else:
        fd, output_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

    # Output size cap (estimate before invoking engine).
    estimated = estimate_tts_output_bytes(text)
    if estimated > _max_audio_bytes():
        raise OutputTooLargeError(
            estimated_bytes=estimated,
            max_bytes=_max_audio_bytes(),
        )

    # Dry-run: return preview, do NOT invoke the engine.
    if dry_run:
        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=(
                        f"Dry run — would synthesize {len(text)} characters to:\n"
                        f"  output: {output_path}\n"
                        f"  voice:  {voice or '<config default>'}\n"
                        f"  engine: <active TTS engine>\n"
                        f"  estimated WAV size: ~{estimated:,} bytes\n"
                        f"No files were written. Re-run with dry_run=false to synthesize."
                    ),
                )
            ],
            structuredContent={
                "ok": True,
                "engine": None,
                "duration": None,
                "audio_path": None,
                "text_length": len(text),
                "estimated_bytes": estimated,
                "dry_run": True,
            },
            isError=False,
        )

    _ensure_tts()
    if _tts_engine is None:
        raise EngineUnavailableError(kind="tts")

    kwargs: dict[str, Any] = {}
    if voice:
        kwargs["voice"] = voice

    result = _tts_engine.speak(text, output_path, **kwargs)

    return CallToolResult(
        content=[
            TextContent(
                type="text",
                text=(
                    f"Synthesis ({result.engine})\n"
                    f"Duration: {result.duration:.2f}s\n"
                    f"Output: {result.audio_path}"
                ),
            )
        ],
        structuredContent={
            "ok": True,
            "engine": result.engine,
            "duration": result.duration,
            "audio_path": result.audio_path,
            "text_length": len(text),
            "dry_run": False,
        },
        isError=False,
    )


async def _handle_list_stt_models(args: dict) -> CallToolResult:
    _ensure_stt()
    if _stt_engine is None:
        return CallToolResult(
            content=[TextContent(type="text", text="No STT engine enabled")],
            structuredContent={"ok": True, "engine": None, "models": []},
            isError=False,
        )
    models = _stt_engine.list_models()
    lines = [f"STT Engine: {_stt_engine.name}"]
    for m in models:
        lines.append(f"  - {m['name']} ({m.get('params', '?')}) — {m.get('note', '')}")
    return CallToolResult(
        content=[TextContent(type="text", text="\n".join(lines))],
        structuredContent={
            "ok": True,
            "engine": _stt_engine.name,
            "models": models,
        },
        isError=False,
    )


async def _handle_list_tts_voices(args: dict) -> CallToolResult:
    _ensure_tts()
    if _tts_engine is None:
        return CallToolResult(
            content=[TextContent(type="text", text="No TTS engine enabled")],
            structuredContent={"ok": True, "engine": None, "voices": []},
            isError=False,
        )
    voices = _tts_engine.list_voices()
    lines = [f"TTS Engine: {_tts_engine.name}"]
    for v in voices:
        lines.append(
            f"  - {v['name']} [{v.get('language', '')}/{v.get('gender', '')}] — {v.get('note', '')}"
        )
    return CallToolResult(
        content=[TextContent(type="text", text="\n".join(lines))],
        structuredContent={
            "ok": True,
            "engine": _tts_engine.name,
            "voices": voices,
        },
        isError=False,
    )


async def _handle_reload_config(args: dict) -> CallToolResult:
    global _stt_engine, _tts_engine
    _reload_config_yaml()
    _stt_engine = None
    _tts_engine = None
    _build_stt_engine()
    _build_tts_engine()
    stt_name = _stt_engine.name if _stt_engine else None
    tts_name = _tts_engine.name if _tts_engine else None
    return CallToolResult(
        content=[
            TextContent(
                type="text",
                text=f"Config reloaded. STT engine: {stt_name or 'none'}, TTS engine: {tts_name or 'none'}",
            )
        ],
        structuredContent={
            "ok": True,
            "stt_engine": stt_name,
            "tts_engine": tts_name,
        },
        isError=False,
    )


async def _handle_health_check(args: dict) -> CallToolResult:
    stt_healthy = False
    tts_healthy = False
    if _stt_engine is not None:
        try:
            stt_healthy = bool(_stt_engine.health_check())
        except Exception:  # noqa: BLE001
            stt_healthy = False
    if _tts_engine is not None:
        try:
            tts_healthy = bool(_tts_engine.health_check())
        except Exception:  # noqa: BLE001
            tts_healthy = False

    stt_block = {
        "name": _stt_engine.name if _stt_engine else None,
        "initialized": _stt_engine is not None,
        "healthy": stt_healthy,
    }
    tts_block = {
        "name": _tts_engine.name if _tts_engine else None,
        "initialized": _tts_engine is not None,
        "healthy": tts_healthy,
    }
    overall = (stt_block["initialized"] and stt_healthy) or (
        tts_block["initialized"] and tts_healthy
    )
    lines = [
        f"STT ({stt_block['name'] or 'none'}): {'ok' if stt_healthy else ('not initialized' if not stt_block['initialized'] else 'FAIL')}",
        f"TTS ({tts_block['name'] or 'none'}): {'ok' if tts_healthy else ('not initialized' if not tts_block['initialized'] else 'FAIL')}",
    ]
    return CallToolResult(
        content=[TextContent(type="text", text="\n".join(lines))],
        structuredContent={
            "ok": bool(overall),
            "stt": stt_block,
            "tts": tts_block,
        },
        isError=False,
    )


# Dispatch table — built once, used by the call_tool handler.
_DISPATCH: dict[str, Any] = {
    "transcribe": _handle_transcribe,
    "speak": _handle_speak,
    "list_stt_models": _handle_list_stt_models,
    "list_tts_voices": _handle_list_tts_voices,
    "reload_config": _handle_reload_config,
    "health_check": _handle_health_check,
}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main() -> None:
    """Run the MCP server over stdio."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
