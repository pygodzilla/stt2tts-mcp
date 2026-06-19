<!-- TOP_ANCHOR -->

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://img.shields.io/badge/STT2TTS-0.1.0-1f6feb?style=for-the-badge&labelColor=0d1117&color=1f6feb">
    <img alt="STT2TTS" src="https://img.shields.io/badge/STT2TTS-0.1.0-1f6feb?style=for-the-badge">
  </picture>
</p>

<div align="center">

# STT2TTS MCP Server

**Give your AI agent a voice — and the ability to listen.**

Local-first speech-to-text and text-to-speech over the Model Context Protocol. Hot-swappable engines via `config.yaml`, no API keys required, fully offline-capable.

</div>

<div align="center">

![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square&labelColor=0d1117)
![Python](https://img.shields.io/badge/Python-3.11%2B-3776ab?style=flat-square&labelColor=0d1117&logo=python&logoColor=white)
![MCP](https://img.shields.io/badge/MCP-2.0%20%282025--11--25%29-a855f7?style=flat-square&labelColor=0d1117)
![Status](https://img.shields.io/badge/Status-Alpha-orange?style=flat-square&labelColor=0d1117)
![Tests](https://img.shields.io/badge/Tests-108%20passing-22c55e?style=flat-square&labelColor=0d1117)
![Bandit](https://img.shields.io/badge/Bandit-clean-22c55e?style=flat-square&labelColor=0d1117)
![pip-audit](https://img.shields.io/badge/pip--audit-0%20CVEs-22c55e?style=flat-square&labelColor=0d1117)
![Local](https://img.shields.io/badge/Local--only-22c55e?style=flat-square&labelColor=0d1117)

</div>

---

<p align="center">
  <img src="https://img.shields.io/badge/6_tools-transcribe%20%C2%B7%20speak%20%C2%B7%20health_check%20%C2%B7%20list_stt_models%20%C2%B7%20list_tts_voices%20%C2%B7%20reload_config-1f6feb?style=flat-square" />
  &nbsp;
  <img src="https://img.shields.io/badge/5_STT_engines-faster__whisper%20%C2%B7%20sherpa__onnx%20%C2%B7%20openai__api%20%C2%B7%20ollama%20%C2%B7%20lmstudio-a855f7?style=flat-square" />
  &nbsp;
  <img src="https://img.shields.io/badge/6_TTS_engines-piper%20%C2%B7%20kokoro%20%C2%B7%20coqui%20%C2%B7%20openai__api%20%C2%B7%20ollama%20%C2%B7%20lmstudio-a855f7?style=flat-square" />
</p>

---

## Why

Replaces `whisper-mcp`. Works fully offline, ships with five STT and six TTS engines, switches per-task via `config.yaml`. No code changes, no API keys, no cloud round-trips.

## Install

```bash
pip install stt2tts-mcp

# Add only the engines you actually use:
pip install stt2tts-mcp[stt-faster-whisper]   # local STT
pip install stt2tts-mcp[tts-piper]            # local TTS (~50MB voices)
```

Register with your MCP client (consult your client's docs for the exact config file location — most use `mcp_config.json` or a per-client equivalent):

```json
{
  "mcpServers": {
    "stt2tts": {
      "type": "local",
      "command": ["stt2tts-mcp"],
      "enabled": true
    }
  }
}
```

## Engines

| STT | Size | License | Best for |
|-----|------|---------|----------|
| faster-whisper | 39M – 2.9 GB | MIT | English, INT8 CPU, fastest |
| sherpa-onnx | 39M – large | Apache 2.0 | Multilingual |
| OpenAI API | cloud | Proprietary | Highest accuracy, needs key |
| Ollama | varies | MIT | Local LLM integration |
| LMStudio | varies | MIT | Local model server |

| TTS | Voice size | License | Best for |
|-----|-----------|---------|----------|
| Piper | 20 – 50 MB | Apache 2.0 | Smallest, 10–20× realtime |
| Kokoro-82M | ~330 MB | Apache 2.0 | Quality/size ratio |
| Coqui XTTS | ~1.5 GB | MPL 2.0 | Voice cloning, needs GPU |
| OpenAI API | cloud | Proprietary | All voices, needs key |
| Ollama | varies | MIT | LLM-based voices |
| LMStudio | varies | MIT | Local model server |

## Configure

`config.yaml`:

```yaml
stt:
  engine: faster_whisper   # sherpa_onnx | openai_api | ollama | lmstudio
  enabled: true
  params:
    model_size: base.en     # tiny.en | base.en | small.en | medium.en
    device: cpu             # cpu | cuda

tts:
  engine: piper             # kokoro | coqui | openai_api | ollama | lmstudio
  enabled: true
  params:
    voice: en_US-lessac-medium
    model_dir: ~/.cache/piper

limits:                     # Sprint 1 — enterprise defaults
  max_text_chars: 100000    # speak input cap
  max_output_bytes: 104857600  # 100MB WAV cap
```

Reload without restart by calling the `reload_config` MCP tool.

## MCP Tools

| Tool | What it does | MCP 2.0 annotations |
|------|--------------|---------------------|
| `transcribe(audio_path, language?, task?)` | Audio file → text | `readOnlyHint: true` |
| `speak(text, output_path?, voice?, dry_run?)` | Text → WAV file | `readOnlyHint: false` |
| `list_stt_models` | Available STT models | `readOnlyHint: true` |
| `list_tts_voices` | Available TTS voices | `readOnlyHint: true` |
| `reload_config` | Re-read `config.yaml`, rebuild engines | `readOnlyHint: false` |
| `health_check` | Engine status | `readOnlyHint: true` |

All formats ffmpeg supports (wav, mp3, ogg, flac, m4a) are accepted; STT input is auto-converted to 16 kHz mono.

## Enterprise-Readiness (Sprint 1)

Built against the **MCP 2.0 (protocol 2025-11-25)** specification.

| Feature | What it does |
|---------|--------------|
| **Structured errors** | Every error carries a stable `code` (`invalid_path`, `rate_limited`, `text_too_long`, `engine_unavailable`, …) so clients can route without parsing human text |
| **Input validation** | Path traversal blocked, NUL bytes rejected, file-extension allowlist enforced, control characters stripped from `speak.text` |
| **Output caps** | `max_text_chars` (100k default) and `max_output_bytes` (100MB default) prevent accidental DoS |
| **Rate limiting** | Token-bucket per tool (30/0.5/s for `transcribe`, 15/0.25/s for `speak`); cheap reads are unthrottled |
| **Dry-run mode** | `speak(text=..., dry_run=true)` previews without writing any file |
| **Audit log** | Append-only JSONL at `~/.local/share/stt2tts-mcp/audit.log`; every tool call recorded with timestamp, args (text redacted), duration, outcome |
| **MCP 2.0 features** | `Tool.title`, `Tool.annotations`, `Tool.outputSchema`, `CallToolResult.structuredContent`, `Server(version=...)`, `Server(instructions=...)` |
| **Security scans** | `bandit` clean (0 high/medium), `pip-audit` clean (0 CVEs in our direct deps), `ruff` lint + format |

## Develop

Source-only releases ship on `main` for clean installs. The `dev` branch carries the test suite for contributors.

```bash
git clone https://github.com/pygodzilla/stt2tts-mcp
cd stt2tts-mcp
git checkout dev                 # for tests + dev iteration
pip install -e ".[all]"
python -m stt2tts_mcp.server
```

Run the security gates locally:

```bash
ruff check stt2tts_mcp/ tests/         # lint
ruff format stt2tts_mcp/ tests/        # format
bandit -r stt2tts_mcp/ -ll             # security linter
pip-audit                              # dependency CVEs
python -m pytest tests/                # 108 tests
```

## License

MIT © Adarsh