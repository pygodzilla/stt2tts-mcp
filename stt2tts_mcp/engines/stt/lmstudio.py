"""LM Studio STT engine — local LLM inference server with Whisper support."""

import base64
import time
import wave
from typing import Any

import requests

from .base import EngineConfig, STTEngine, TranscriptionResult


class LMStudioSTTEngine(STTEngine):
    """LM Studio STT engine — runs Whisper models via LM Studio's local server.

    Default: http://localhost:1234
    LM Studio exposes an OpenAI-compatible API for Whisper.
    """

    name = "lmstudio"
    supports_streaming = False

    _config: EngineConfig | None = None

    def __init__(self, config: EngineConfig) -> None:
        self._config = config
        self._params = config.params

    def transcribe(self, audio_path: str, **kwargs) -> TranscriptionResult:
        """Transcribe audio via LM Studio's OpenAI-compatible Whisper endpoint."""
        base_url = self._params.get("base_url", "http://localhost:1234")
        model = self._params.get("model", "whisper")
        timeout = self._params.get("timeout", 30)
        language = kwargs.get("language") or self._params.get("language", "en")

        start = time.monotonic()

        # Read audio duration
        with wave.open(audio_path, "rb") as wav:
            frames = wav.getnframes()
            rate = wav.getframerate()
            audio_duration = frames / float(rate)

        # Encode audio as base64
        with open(audio_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode()

        response = requests.post(
            f"{base_url}/v1/audio/transcriptions",
            data={
                "model": model,
                "language": language,
            },
            files={
                "file": (audio_path, open(audio_path, "rb"), "audio/wav"),
            },
            timeout=timeout,
        )
        response.raise_for_status()
        result = response.json()

        text = result.get("text", "")
        duration = time.monotonic() - start

        return TranscriptionResult(
            text=text.strip(),
            language=language,
            duration=duration,
            engine=self.name,
        )

    def swap_config(self, config: EngineConfig) -> None:
        """Hot-swap: no model reload needed for LM Studio."""
        self._config = config
        self._params = config.params

    def list_models(self) -> list[dict]:
        """List available LM Studio Whisper models."""
        return [
            {
                "name": "whisper",
                "size_mb": "varies",
                "languages": ["many"],
                "note": "Default Whisper via LM Studio",
            },
        ]

    def health_check(self) -> bool:
        """Check if LM Studio server is reachable."""
        base_url = self._params.get("base_url", "http://localhost:1234")
        try:
            response = requests.get(f"{base_url}/v1/models", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
