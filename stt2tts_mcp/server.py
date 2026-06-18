"""STT2TTS MCP Server — exposes STT and TTS tools via MCP protocol."""

from __future__ import annotations

import tempfile
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from stt2tts_mcp.engines import STT_ENGINES, TTS_ENGINES, EngineConfig
from stt2tts_mcp.utils import (
    convert_to_wav,
    get_audio_duration,
    get_engine_params,
    get_stt_config,
    get_tts_config,
    load_config,
    validate_audio_file,
)


# ---------------------------------------------------------------------------
# Server instance
# ---------------------------------------------------------------------------

SERVER_NAME = "stt2tts-mcp"
server = Server(SERVER_NAME)


# ---------------------------------------------------------------------------
# Engine lifecycle
# ---------------------------------------------------------------------------

_stt_engine = None
_tts_engine = None


def _build_stt_engine() -> None:
    """Build the active STT engine from config.yaml."""
    global _stt_engine
    config = load_config()
    stt_cfg = get_stt_config(config)
    if stt_cfg is None:
        _stt_engine = None
        return
    engine_name = stt_cfg.get("engine", "faster_whisper")
    if engine_name not in STT_ENGINES:
        raise ValueError(f"Unknown STT engine: {engine_name}")
    params = get_engine_params(stt_cfg)
    engine_config = EngineConfig(name=engine_name, enabled=True, params=params)
    _stt_engine = STT_ENGINES[engine_name](engine_config)


def _build_tts_engine() -> None:
    """Build the active TTS engine from config.yaml."""
    global _tts_engine
    config = load_config()
    tts_cfg = get_tts_config(config)
    if tts_cfg is None:
        _tts_engine = None
        return
    engine_name = tts_cfg.get("engine", "piper")
    if engine_name not in TTS_ENGINES:
        raise ValueError(f"Unknown TTS engine: {engine_name}")
    params = get_engine_params(tts_cfg)
    engine_config = EngineConfig(name=engine_name, enabled=True, params=params)
    _tts_engine = TTS_ENGINES[engine_name](engine_config)


def _ensure_stt() -> None:
    """Lazily build STT engine on first use."""
    global _stt_engine
    if _stt_engine is None:
        _build_stt_engine()


def _ensure_tts() -> None:
    """Lazily build TTS engine on first use."""
    global _tts_engine
    if _tts_engine is None:
        _build_tts_engine()


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

TRANSCRIBE_TOOL = Tool(
    id="transcribe",
    name="transcribe",
    description="Transcribe an audio file to text using the active STT engine. "
    "Supports faster-whisper, sherpa-onnx, OpenAI API, Ollama, and LMStudio backends. "
    "Engine is selected via config.yaml.",
    inputSchema={
        "type": "object",
        "properties": {
            "audio_path": {
                "type": "string",
                "description": "Path to the audio file (wav, mp3, ogg, flac, m4a, etc.). "
                "ffmpeg is used for format conversion if needed.",
            },
            "language": {
                "type": "string",
                "description": "Language code (e.g., 'en', 'es', 'fr'). "
                "Defaults to config value or 'en'.",
            },
            "task": {
                "type": "string",
                "enum": ["transcribe", "translate"],
                "description": "'transcribe' (default) or 'translate' (to English).",
            },
        },
        "required": ["audio_path"],
    },
)

SPEAK_TOOL = Tool(
    id="speak",
    name="speak",
    description="Synthesize text to speech and save as a WAV file. "
    "Supports Piper TTS, Kokoro, Coqui, OpenAI API, Ollama, and LMStudio backends. "
    "Engine and voice are selected via config.yaml.",
    inputSchema={
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Text to synthesize.",
            },
            "output_path": {
                "type": "string",
                "description": "Path to save the output WAV file. "
                "Defaults to a temp file if not provided.",
            },
            "voice": {
                "type": "string",
                "description": "Voice name override (e.g., 'en_US-lessac-medium'). "
                "Defaults to config value.",
            },
        },
        "required": ["text"],
    },
)

LIST_STT_MODELS_TOOL = Tool(
    id="list_stt_models",
    name="list_stt_models",
    description="List available STT models for the active engine.",
    inputSchema={"type": "object", "properties": {}},
)

LIST_TTS_VOICES_TOOL = Tool(
    id="list_tts_voices",
    name="list_tts_voices",
    description="List available TTS voices for the active engine.",
    inputSchema={"type": "object", "properties": {}},
)

RELOAD_CONFIG_TOOL = Tool(
    id="reload_config",
    name="reload_config",
    description="Hot-reload config.yaml and rebuild engines. "
    "Use this after editing config.yaml to switch engines without restarting.",
    inputSchema={"type": "object", "properties": {}},
)

HEALTH_CHECK_TOOL = Tool(
    id="health_check",
    name="health_check",
    description="Check health of active STT and TTS engines.",
    inputSchema={"type": "object", "properties": {}},
)


# ---------------------------------------------------------------------------
# Request handlers
# ---------------------------------------------------------------------------

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
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""
    if name == "transcribe":
        return await _transcribe(arguments)
    elif name == "speak":
        return await _speak(arguments)
    elif name == "list_stt_models":
        return await _list_stt_models(arguments)
    elif name == "list_tts_voices":
        return await _list_tts_voices(arguments)
    elif name == "reload_config":
        return await _reload_config(arguments)
    elif name == "health_check":
        return await _health_check(arguments)
    else:
        raise ValueError(f"Unknown tool: {name}")


async def _transcribe(args: dict) -> list[TextContent]:
    """Transcribe audio file to text."""
    from stt2tts_mcp.utils.audio import convert_to_wav

    audio_path = args["audio_path"]
    language = args.get("language")
    task = args.get("task", "transcribe")

    # Validate
    valid, reason = validate_audio_file(audio_path)
    if not valid:
        raise ValueError(f"Invalid audio file: {reason}")

    # Convert to wav if needed
    try:
        wav_path = convert_to_wav(audio_path)
    except Exception as exc:
        raise RuntimeError(f"Audio conversion failed: {exc}")

    try:
        _ensure_stt()
        if _stt_engine is None:
            raise RuntimeError("No STT engine enabled in config.yaml")

        kwargs = {}
        if language:
            kwargs["language"] = language
        if task:
            kwargs["task"] = task

        result = _stt_engine.transcribe(wav_path, **kwargs)
        return [
            TextContent(
                type="text",
                text=(
                    f"Transcription ({result.engine})\n"
                    f"Language: {result.language or 'auto'}\n"
                    f"Duration: {result.duration:.2f}s\n"
                    f"Text: {result.text}"
                ),
            )
        ]
    finally:
        if wav_path != audio_path and Path(wav_path).exists():
            Path(wav_path).unlink(missing_ok=True)


async def _speak(args: dict) -> list[TextContent]:
    """Synthesize text to speech."""
    text = args["text"]
    output_path = args.get("output_path")
    voice = args.get("voice")

    if not output_path:
        fd, output_path = tempfile.mkstemp(suffix=".wav")
        import os
        os.close(fd)

    _ensure_tts()
    if _tts_engine is None:
        raise RuntimeError("No TTS engine enabled in config.yaml")

    kwargs = {}
    if voice:
        kwargs["voice"] = voice

    result = _tts_engine.speak(text, output_path, **kwargs)
    return [
        TextContent(
            type="text",
            text=(
                f"Synthesis ({result.engine})\n"
                f"Duration: {result.duration:.2f}s\n"
                f"Output: {result.audio_path}"
            ),
        )
    ]


async def _list_stt_models(args: dict) -> list[TextContent]:
    """List available STT models."""
    _ensure_stt()
    if _stt_engine is None:
        return [TextContent(type="text", text="No STT engine enabled")]
    models = _stt_engine.list_models()
    lines = [f"STT Engine: {_stt_engine.name}"]
    for m in models:
        lines.append(
            f"  - {m['name']} ({m.get('params','?')}) — {m.get('note','')}"
        )
    return [TextContent(type="text", text="\n".join(lines))]


async def _list_tts_voices(args: dict) -> list[TextContent]:
    """List available TTS voices."""
    _ensure_tts()
    if _tts_engine is None:
        return [TextContent(type="text", text="No TTS engine enabled")]
    voices = _tts_engine.list_voices()
    lines = [f"TTS Engine: {_tts_engine.name}"]
    for v in voices:
        lines.append(
            f"  - {v['name']} [{v.get('language','')}/{v.get('gender','')}] — {v.get('note','')}"
        )
    return [TextContent(type="text", text="\n".join(lines))]


async def _reload_config(args: dict) -> list[TextContent]:
    """Hot-reload config and rebuild engines."""
    global _stt_engine, _tts_engine
    from stt2tts_mcp.utils.config import reload_config
    reload_config()
    _stt_engine = None
    _tts_engine = None
    _build_stt_engine()
    _build_tts_engine()
    stt_name = _stt_engine.name if _stt_engine else "none"
    tts_name = _tts_engine.name if _tts_engine else "none"
    return [
        TextContent(
            type="text",
            text=f"Config reloaded. STT engine: {stt_name}, TTS engine: {tts_name}",
        )
    ]


async def _health_check(args: dict) -> list[TextContent]:
    """Check engine health."""
    lines = []
    if _stt_engine is not None:
        ok = _stt_engine.health_check()
        lines.append(f"STT ({_stt_engine.name}): {'ok' if ok else 'FAIL'}")
    else:
        lines.append("STT: not initialized")
    if _tts_engine is not None:
        ok = _tts_engine.health_check()
        lines.append(f"TTS ({_tts_engine.name}): {'ok' if ok else 'FAIL'}")
    else:
        lines.append("TTS: not initialized")
    return [TextContent(type="text", text="\n".join(lines))]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    """Run the MCP server over stdio."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
