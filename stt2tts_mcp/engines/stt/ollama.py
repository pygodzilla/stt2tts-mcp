"""Ollama STT engine — local LLM inference with Whisper support."""

import os
import time
from typing import Any

from .base import EngineConfig, STTEngine, TranscriptionResult


class OllamaSTTEngine(STTEngine):
    """Ollama STT engine — runs Whisper models locally via Ollama.

    Set OLLAMA_BASE_URL env var or configure base_url in config.yaml.
    Default: http://localhost:11434
    """

    name = "ollama"
    supports_streaming = False

    _config: EngineConfig | None = None

    def __init__(self, config: EngineConfig) -> None:
        self._config = config
        self._params = config.params

    def transcribe(self, audio_path: str, **kwargs) -> TranscriptionResult:
        """Transcribe audio via Ollama's Whisper endpoint."""
        import base64
        import wave

        base_url = self._params.get("base_url", "http://localhost:11434")
        model = self._params.get("model", "whisper")
        timeout = self._params.get("timeout", 30)
        language = kwargs.get("language") or self._params.get("language", "en")

        start = time.monotonic()

        # Read audio and encode as base64
        with wave.open(audio_path, "rb") as wav:
            frames = wav.getnframes()
            rate = wav.getframerate()
            audio_duration = frames / float(rate)

        with open(audio_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode()

        import requests

        response = requests.post(
            f"{base_url}/api/whisper",
            json={
                "model": model,
                "audio": audio_b64,
                "language": language,
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
        """Hot-swap: no model reload needed for Ollama."""
        self._config = config
        self._params = config.params

    def list_models(self) -> list[dict]:
        """List available Ollama Whisper models."""
        return [
            {
                "name": "whisper",
                "size_mb": "varies",
                "languages": ["many"],
                "note": "Default Whisper via Ollama",
            },
            {
                "name": "whisper-large-v3",
                "size_mb": "varies",
                "languages": ["many"],
                "note": "Large v3 model via Ollama",
            },
        ]

    def health_check(self) -> bool:
        """Check if Ollama server is reachable."""
        import requests

        base_url = self._params.get("base_url", "http://localhost:11434")
        try:
            response = requests.get(f"{base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
