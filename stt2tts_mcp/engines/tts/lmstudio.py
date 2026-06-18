"""LM Studio TTS engine — text-to-speech via LM Studio's OpenAI-compatible API."""

import time
from typing import Any

import requests

from .base import EngineConfig, SynthesisResult, TTSEngine


class LMStudioTTSEngine(TTSEngine):
    """LM Studio TTS engine — uses LM Studio's OpenAI-compatible TTS endpoint.

    Default: http://localhost:1234
    LM Studio exposes an OpenAI-compatible /v1/audio/speech endpoint.
    """

    name = "lmstudio"

    _config: EngineConfig | None = None

    def __init__(self, config: EngineConfig) -> None:
        self._config = config
        self._params = config.params

    def speak(self, text: str, output_path: str, **kwargs) -> SynthesisResult:
        """Synthesize text via LM Studio TTS endpoint."""
        base_url = self._params.get("base_url", "http://localhost:1234")
        model = self._params.get("model", "kokoro")
        timeout = self._params.get("timeout", 30)
        voice = kwargs.get("voice") or self._params.get("voice", "alloy")

        start = time.monotonic()

        if not output_path.endswith(".mp3") and not output_path.endswith(".wav"):
            output_path = output_path + ".mp3"

        response = requests.post(
            f"{base_url}/v1/audio/speech",
            json={
                "model": model,
                "voice": voice,
                "input": text,
                "response_format": "mp3" if output_path.endswith(".mp3") else "wav",
            },
            timeout=timeout,
        )
        response.raise_for_status()

        with open(output_path, "wb") as f:
            f.write(response.content)

        duration = time.monotonic() - start

        return SynthesisResult(
            audio_path=output_path,
            duration=duration,
            engine=self.name,
            text_length=len(text),
        )

    def list_voices(self) -> list[dict]:
        """List LM Studio TTS voices (model-dependent)."""
        return [
            {"name": "alloy", "language": "en", "gender": "neutral", "note": "Default neutral"},
            {"name": "echo", "language": "en", "gender": "male", "note": "Warm male"},
            {"name": "fable", "language": "en", "gender": "male", "note": "Storytelling male"},
            {"name": "onyx", "language": "en", "gender": "male", "note": "Deep male"},
            {"name": "nova", "language": "en", "gender": "female", "note": "Bright female"},
            {"name": "shimmer", "language": "en", "gender": "female", "note": "Soft female"},
        ]

    def swap_config(self, config: EngineConfig) -> None:
        """Hot-swap: no restart needed."""
        self._config = config
        self._params = config.params

    def health_check(self) -> bool:
        """Check if LM Studio server is reachable."""
        base_url = self._params.get("base_url", "http://localhost:1234")
        try:
            response = requests.get(f"{base_url}/v1/models", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
