# STT2TTS MCP Server

Local-first speech-to-text and text-to-speech MCP server. Hot-swappable engines via `config.yaml` — no code changes, no API keys required.

```
┌──────────────┐     stdio      ┌──────────────────┐
│ MCP client   │ ◀────────────▶ │ stt2tts-mcp      │
│              │                │  ├─ STT engine   │ ──▶ faster-whisper
│              │                │  └─ TTS engine   │ ──▶ piper / kokoro / coqui
└──────────────┘                └──────────────────┘
                                       │
                                       ▼
                              config.yaml (hot-reload)
```

## Why

Replaces `whisper-mcp`. Works offline, ships with five STT and six TTS engines, switches per-task via config.

## Install

```bash
pip install stt2tts-mcp

# Add the engines you actually use:
pip install stt2tts-mcp[stt-faster-whisper]   # local STT
pip install stt2tts-mcp[tts-piper]            # local TTS (~50MB voices)

# Register with your MCP client (consult your client's docs for the exact
# config file location — most use mcp_config.json or a per-client equivalent):
{
  "mcp": {
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
| Piper | 20 – 50 MB | Apache 2.0 | Smallest, 10-20× realtime |
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
```

Reload without restart by calling the `reload_config` MCP tool.

## MCP Tools

| Tool | What it does |
|------|--------------|
| `transcribe(audio_path, language?)` | Audio file → text |
| `speak(text, output_path, voice?)` | Text → WAV file |
| `list_stt_models` | Available STT models |
| `list_tts_voices` | Available TTS voices |
| `reload_config` | Re-read `config.yaml`, rebuild engines |
| `health_check` | Engine status |

All formats ffmpeg supports (wav, mp3, ogg, flac, m4a) are accepted; STT input is auto-converted to 16 kHz mono.

## Develop

Source-only releases ship on `main` for clean installs. The `dev` branch
carries the test suite (`tests/test_config_loader.py`,
`tests/test_mcp_integration.py`, `tests/test_piper_no_json.py`) for
contributors.

```bash
git clone https://github.com/pygodzilla/stt2tts-mcp
cd stt2tts-mcp
git checkout dev                 # for tests + dev iteration
pip install -e ".[all]"
python -m stt2tts_mcp.server
```

## License

MIT
