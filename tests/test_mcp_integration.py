"""End-to-end integration tests for the STT2TTS MCP server.

Boots the server as a real subprocess and exchanges raw JSON-RPC over
newline-delimited stdio. This is the same transport any MCP client uses,
so these tests catch protocol/framing bugs (not just handler bugs).

Why a hand-rolled JSON-RPC client instead of `mcp.client.stdio`?
The official SDK uses `anyio.create_task_group()` internally, which
conflicts with pytest-asyncio's per-test event loop and triggers
"Attempted to exit cancel scope in a different task than it was entered"
errors during teardown. A subprocess + raw JSON-RPC sidesteps anyio
entirely and is identical to what the SDK does on the wire.

Test matrix (8 MCP interactions):
  1. initialize              — MCP handshake + server identity
  2. tools/list              — advertises 6 tools
  3. health_check            — engine health probe
  4. list_tts_voices         — TTS engine voices
  5. list_stt_models         — STT engine models
  6. reload_config           — hot-reload config + rebuild engines
  7. speak                   — synthesize audio to disk
  8. transcribe              — speech-to-text (round-trip via #7)

Plus error-path coverage:
  - unknown tool name
  - missing required argument
  - validation error for non-existent audio file
"""
from __future__ import annotations

import asyncio
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Environment guards
# ---------------------------------------------------------------------------


def _has_piper_binary() -> bool:
    """Piper TTS requires a CLI binary in PATH or common install locations."""
    if shutil.which("piper") is not None:
        return True
    for cand in (
        Path("~/Library/Piper/piper").expanduser(),
        Path("/usr/local/bin/piper"),
        Path("/usr/bin/piper"),
        Path("~/.local/bin/piper"),
    ):
        if cand.exists():
            return True
    return False


def _has_whisper_model() -> bool:
    """faster-whisper needs a cached model. tiny.en is the smallest (75MB)."""
    cache = Path("~/.cache/whisper").expanduser()
    if not cache.exists():
        return False
    # Either the huggingface hub layout or a converted .pt file.
    return any(cache.glob("**/model.bin")) or (cache / "tiny.pt").exists()


def _has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


# ---------------------------------------------------------------------------
# Minimal JSON-RPC client over stdio
# ---------------------------------------------------------------------------


class MCPTestClient:
    """Async JSON-RPC client that talks to an MCP server subprocess.

    Each request is a newline-delimited JSON object on stdin; each response
    is a newline-delimited JSON object on stdout. Notifications have no id
    and no response (we don't send any, so this doesn't matter).

    The `initialize` handshake uses the protocol version and capabilities
    the real MCP client would send. After handshake, requests are routed by
    integer `id`.
    """

    PROTOCOL_VERSION = "2024-11-05"
    CLIENT_INFO = {"name": "stt2tts-mcp-test-client", "version": "0.1.0"}

    def __init__(self, proc: asyncio.subprocess.Process) -> None:
        self._proc = proc
        self._next_id = 1
        self._server_info: dict[str, Any] = {}
        self._server_capabilities: dict[str, Any] = {}
        self._protocol_version: str = ""

    @classmethod
    async def spawn(cls) -> "MCPTestClient":
        """Spawn the server subprocess and return a connected client."""
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-u",  # unbuffered — critical for stdio JSON-RPC
            "-m",
            "stt2tts_mcp.server",
            cwd=str(PROJECT_ROOT),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        client = cls(proc)
        await client._initialize()
        return client

    async def _send(self, message: dict[str, Any]) -> None:
        line = json.dumps(message) + "\n"
        assert self._proc.stdin is not None
        self._proc.stdin.write(line.encode("utf-8"))
        await self._proc.stdin.drain()

    async def _read_message(self, timeout: float = 30.0) -> dict[str, Any]:
        """Read one newline-delimited JSON message from stdout."""
        assert self._proc.stdout is not None
        loop = asyncio.get_event_loop()
        # Each line is a complete JSON-RPC message. Read until we get a line.
        # The server is single-threaded so messages don't interleave on a line,
        # but we still read line-by-line for safety.
        while True:
            line_bytes = await asyncio.wait_for(
                self._proc.stdout.readline(), timeout=timeout
            )
            if not line_bytes:
                # Server closed stdout — collect stderr for diagnostics.
                stderr_data = b""
                if self._proc.stderr is not None:
                    stderr_data = await self._proc.stderr.read()
                raise RuntimeError(
                    f"Server closed stdout unexpectedly. stderr: "
                    f"{stderr_data.decode('utf-8', errors='replace')}"
                )
            line = line_bytes.decode("utf-8", errors="replace").strip()
            if not line:
                continue  # blank line, keep reading
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                # Non-JSON stdout from the server (e.g., Python warnings on
                # stderr that leaked to stdout). Skip and try the next line.
                continue

    async def _request(
        self, method: str, params: dict[str, Any] | None = None, timeout: float = 60.0
    ) -> dict[str, Any]:
        """Send a request and wait for the matching response (by id)."""
        req_id = self._next_id
        self._next_id += 1
        msg: dict[str, Any] = {"jsonrpc": "2.0", "id": req_id, "method": method}
        if params is not None:
            msg["params"] = params
        await self._send(msg)

        # Read messages until we find one with our id.
        deadline = asyncio.get_event_loop().time() + timeout
        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                raise TimeoutError(
                    f"No response to {method} (id={req_id}) within {timeout}s"
                )
            resp = await self._read_message(timeout=remaining)
            if resp.get("id") == req_id:
                return resp

    async def _initialize(self) -> None:
        resp = await self._request(
            "initialize",
            {
                "protocolVersion": self.PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": self.CLIENT_INFO,
            },
        )
        if "error" in resp:
            raise RuntimeError(f"initialize failed: {resp['error']}")
        result = resp.get("result", {})
        self._server_info = result.get("serverInfo", {})
        self._server_capabilities = result.get("capabilities", {})
        self._protocol_version = result.get("protocolVersion", "")
        # MCP requires the client to send an `initialized` notification.
        await self._send(
            {"jsonrpc": "2.0", "method": "notifications/initialized"}
        )

    # -- Public API --------------------------------------------------------

    @property
    def server_info(self) -> dict[str, Any]:
        return self._server_info

    @property
    def server_capabilities(self) -> dict[str, Any]:
        return self._server_capabilities

    @property
    def protocol_version(self) -> str:
        return self._protocol_version

    async def list_tools(self) -> list[dict[str, Any]]:
        resp = await self._request("tools/list")
        if "error" in resp:
            raise RuntimeError(f"tools/list failed: {resp['error']}")
        return resp.get("result", {}).get("tools", [])

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        resp = await self._request(
            "tools/call", {"name": name, "arguments": arguments}
        )
        if "error" in resp:
            # Some errors come back inside the result with isError=True;
            # others come back as a JSON-RPC error. Both are testable.
            return {
                "isError": True,
                "content": [{"type": "text", "text": str(resp["error"])}],
            }
        result = resp.get("result", {})
        return {
            "isError": bool(result.get("isError", False)),
            "content": result.get("content", []),
        }

    async def close(self) -> None:
        if self._proc.returncode is None:
            try:
                self._proc.terminate()
                await asyncio.wait_for(self._proc.wait(), timeout=5.0)
            except (asyncio.TimeoutError, ProcessLookupError):
                try:
                    self._proc.kill()
                except ProcessLookupError:
                    pass


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
async def client():
    """Spawn a fresh MCP server subprocess for each test."""
    c = await MCPTestClient.spawn()
    try:
        yield c
    finally:
        await c.close()


# ---------------------------------------------------------------------------
# 1. initialize
# ---------------------------------------------------------------------------


async def test_initialize_returns_server_identity(client: MCPTestClient) -> None:
    """The MCP handshake must succeed and identify the server."""
    assert client.server_info.get("name") == "stt2tts-mcp"
    assert client.protocol_version, "protocol version is empty"
    # Server should declare tool capability — otherwise tools/list won't work.
    assert "tools" in client.server_capabilities, (
        f"Server capabilities missing 'tools': {client.server_capabilities}"
    )


# ---------------------------------------------------------------------------
# 2. tools/list
# ---------------------------------------------------------------------------


async def test_list_tools_advertises_six_tools(client: MCPTestClient) -> None:
    """list_tools must return all six tools defined in server.py."""
    tools = await client.list_tools()
    names = {t["name"] for t in tools}
    assert names == {
        "transcribe",
        "speak",
        "list_stt_models",
        "list_tts_voices",
        "reload_config",
        "health_check",
    }, f"Unexpected tool list: {names}"

    # Every tool must have a name and a non-empty description.
    for tool in tools:
        assert tool["name"]
        assert tool.get("description")


# ---------------------------------------------------------------------------
# 3. health_check
# ---------------------------------------------------------------------------


async def test_health_check_reports_engine_status(client: MCPTestClient) -> None:
    """health_check must return text describing STT + TTS engine state."""
    result = await client.call_tool("health_check", {})
    assert not result["isError"], f"health_check errored: {result['content']}"
    text = result["content"][0]["text"]
    assert "STT" in text
    assert "TTS" in text
    # Either initialized (with engine name) or explicitly "not initialized".
    assert ("not initialized" in text) or ("ok" in text or "FAIL" in text)


# ---------------------------------------------------------------------------
# 4. list_tts_voices
# ---------------------------------------------------------------------------


async def test_list_tts_voices_returns_voices(client: MCPTestClient) -> None:
    """list_tts_voices must return at least one voice for the active engine."""
    result = await client.call_tool("list_tts_voices", {})
    assert not result["isError"], f"list_tts_voices errored: {result['content']}"
    text = result["content"][0]["text"]
    if "No TTS engine enabled" in text:
        pytest.skip("TTS engine disabled in config.yaml")
    assert "TTS Engine:" in text
    assert "-" in text  # bullet list of voices


# ---------------------------------------------------------------------------
# 5. list_stt_models
# ---------------------------------------------------------------------------


async def test_list_stt_models_returns_models(client: MCPTestClient) -> None:
    """list_stt_models must return at least one model for the active engine."""
    result = await client.call_tool("list_stt_models", {})
    assert not result["isError"], f"list_stt_models errored: {result['content']}"
    text = result["content"][0]["text"]
    if "No STT engine enabled" in text:
        pytest.skip("STT engine disabled in config.yaml")
    assert "STT Engine:" in text
    assert "-" in text


# ---------------------------------------------------------------------------
# 6. reload_config
# ---------------------------------------------------------------------------


async def test_reload_config_hot_swaps_engines(client: MCPTestClient) -> None:
    """reload_config must report the active engine names after a fresh load."""
    result = await client.call_tool("reload_config", {})
    assert not result["isError"], f"reload_config errored: {result['content']}"
    text = result["content"][0]["text"]
    assert "Config reloaded" in text
    assert "STT engine:" in text
    assert "TTS engine:" in text
    # The STT name should match a known engine (default config has it enabled).
    stt_part = text.split("STT engine:")[1].split(",")[0].strip()
    assert stt_part in {
        "faster_whisper",
        "sherpa_onnx",
        "openai_api",
        "ollama",
        "lmstudio",
        "none",
    }


# ---------------------------------------------------------------------------
# 7. speak
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_piper_binary(), reason="piper binary not installed")
async def test_speak_writes_wav_file(client: MCPTestClient, tmp_path: Path) -> None:
    """speak must synthesize text and write a non-empty .wav file."""
    output = tmp_path / "out.wav"
    result = await client.call_tool(
        "speak", {"text": "Hello", "output_path": str(output)}
    )
    assert not result["isError"], f"speak errored: {result['content']}"
    text = result["content"][0]["text"]
    assert "Synthesis" in text
    assert "Duration" in text
    assert str(output) in text

    assert output.exists(), f"speak did not create {output}"
    assert output.stat().st_size > 0, f"speak wrote empty file: {output}"
    # Quick sanity: file starts with RIFF/WAVE magic.
    with open(output, "rb") as fh:
        magic = fh.read(4)
    assert magic == b"RIFF", f"output is not a RIFF/WAVE file (got {magic!r})"


# ---------------------------------------------------------------------------
# 8. transcribe (round-trip with speak)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_piper_binary(), reason="piper binary not installed")
@pytest.mark.skipif(not _has_whisper_model(), reason="whisper model not cached")
@pytest.mark.skipif(not _has_ffmpeg(), reason="ffmpeg not installed")
async def test_transcribe_round_trip_with_speak(
    client: MCPTestClient, tmp_path: Path
) -> None:
    """Generate audio via speak, then transcribe it back. Text should appear."""
    # 1. Synthesize a short, easily-recognizable phrase.
    audio_path = tmp_path / "roundtrip.wav"
    speak_result = await client.call_tool(
        "speak", {"text": "hello world", "output_path": str(audio_path)}
    )
    assert not speak_result["isError"], (
        f"speak failed: {speak_result['content']}"
    )
    assert audio_path.exists()

    # 2. Transcribe the generated audio.
    tx_result = await client.call_tool(
        "transcribe", {"audio_path": str(audio_path)}
    )
    assert not tx_result["isError"], (
        f"transcribe errored: {tx_result['content']}"
    )
    text = tx_result["content"][0]["text"]
    assert "Transcription" in text
    assert "Duration" in text
    # We don't assert specific words — tiny.en on synthetic Piper audio is
    # not deterministic enough for that. We just confirm a real transcription
    # was produced (Duration line exists, body is non-empty after "Text:").
    body = text.split("Text:", 1)[1].strip() if "Text:" in text else ""
    assert body, f"Empty transcription: {text!r}"


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


async def test_call_unknown_tool_returns_error(client: MCPTestClient) -> None:
    """An unknown tool name must produce an error result (not a crash)."""
    result = await client.call_tool("definitely_not_a_real_tool", {})
    assert result["isError"], f"expected error, got: {result['content']}"


async def test_transcribe_missing_audio_file_returns_error(
    client: MCPTestClient, tmp_path: Path
) -> None:
    """Transcribing a non-existent file must return a clean error."""
    fake = tmp_path / "does_not_exist.wav"
    result = await client.call_tool("transcribe", {"audio_path": str(fake)})
    assert result["isError"], (
        f"expected error for missing file, got: {result['content']}"
    )
    text = " ".join(c.get("text", "") for c in result["content"])
    assert "not found" in text.lower() or "missing" in text.lower()


async def test_speak_missing_text_argument_returns_error(
    client: MCPTestClient,
) -> None:
    """speak requires a 'text' argument — omitting it must error cleanly."""
    result = await client.call_tool("speak", {})
    assert result["isError"], (
        f"expected error for missing 'text', got: {result['content']}"
    )


# ---------------------------------------------------------------------------
# Stand-alone runner (for sanity-checking without pytest)
# ---------------------------------------------------------------------------


def main() -> None:
    """Allow running this file directly to inspect the test environment.

    Usage: python tests/test_mcp_integration.py
    Pytest-asyncio will skip in CLI mode — this just verifies the file is
    importable and reports which optional dependencies are available.
    """
    print(f"PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"piper available: {_has_piper_binary()}")
    print(f"whisper model available: {_has_whisper_model()}")
    print(f"ffmpeg available: {_has_ffmpeg()}")
    print("OK: test_mcp_integration.py is importable")


if __name__ == "__main__":
    main()