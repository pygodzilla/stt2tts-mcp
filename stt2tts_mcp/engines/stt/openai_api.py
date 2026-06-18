"""OpenAI-compatible API STT engine — works with any OpenAI API-compatible backend."""

import os
import time
from typing import Any

from .base import EngineConfig, STTEngine, TranscriptionResult


class OpenAISTTEngine(STTEngine):
    """OpenAI Whisper API-compatible STT engine.

    Works with:
    - OpenAI Whisper API (api_key required)
    - LM Studio (local)
    - Ollama (local)
    - Any OpenAI API-compatible server
    """

    name = "openai_api"
    supports_streaming = False

    _client: Any = None
    _config: EngineConfig | None = None

    def __init__(self, config: EngineConfig) -> None:
        self._config = config
        self._params = config.params
        self._client = None

    def _get_client(self) -> Any:
        """Lazy-load the OpenAI client."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install stt2tts-mcp[stt-openai]")

        if self._client is not None:
            return self._client

        api_key = self._params.get("api_key") or os.getenv("OPENAI_API_KEY", "")
        base_url = self._params.get("base_url", "https://api.openai.com/v1")
        timeout = self._params.get("timeout", 60)

        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        return self._client

    def transcribe(self, audio_path: str, **kwargs) -> TranscriptionResult:
        """Transcribe audio via OpenAI-compatible API."""
        import wave

        client = self._get_client()
        model = self._params.get("model", "whisper-1")
        language = kwargs.get("language") or self._params.get("language")
        start = time.monotonic()

        with open(audio_path, "rb") as f:
            # Read audio duration for estimate
            with wave.open(audio_path) as wav:
                frames = wav.getnframes()
                rate = wav.getframerate()
                audio_duration = frames / float(rate)

            response = client.audio.transcriptions.create(
                model=model,
                file=f,
                language=language,
                response_format="verbose_json",
            )

        text = str(response.text) if hasattr(response, "text") else str(response)
        duration = time.monotonic() - start

        return TranscriptionResult(
            text=text.strip(),
            language=response.language if hasattr(response, "language") else (language or "en"),
            duration=duration,
            engine=self.name,
        )

    def swap_config(self, config: EngineConfig) -> None:
        """Hot-swap: recreate client if base_url or api_key changed."""
        changed = (
            self._config is None
            or self._config.params.get("base_url") != config.params.get("base_url")
            or self._config.params.get("api_key") != config.params.get("api_key")
        )
        if changed:
            self._client = None
        self._config = config
        self._params = config.params

    def list_models(self) -> list[dict]:
        """List available models (OpenAI Whisper and compatible)."""
        return [
            {
                "name": "whisper-1",
                "type": "openai",
                "languages": ["many"],
                "note": "OpenAI's main Whisper model",
            },
            {
                "name": "whisper-large-v3",
                "type": "openai",
                "languages": ["many"],
                "note": "OpenAI's latest, highest accuracy",
            },
            {
                "name": "distil-whisper-large-v3",
                "type": "openai",
                "languages": ["many"],
                "note": "DistilWhisper: 6x faster, 49% smaller, competitive accuracy",
            },
        ]

    def health_check(self) -> bool:
        """Check if the API endpoint is reachable."""
        try:
            client = self._get_client()
            # Lightweight check: try to list models
            client.models.list()
            return True
        except Exception:
            return False
