"""OpenAI-compatible API TTS engine — works with any OpenAI API-compatible backend."""

import os
import time
from typing import Any

from .base import EngineConfig, SynthesisResult, TTSEngine


class OpenAITTSEngine(TTSEngine):
    """OpenAI TTS API-compatible engine.

    Works with:
    - OpenAI TTS API (api_key required)
    - LM Studio (local)
    - Ollama (local)
    - Any OpenAI API-compatible server
    """

    name = "openai_api"

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

    def speak(self, text: str, output_path: str, **kwargs) -> SynthesisResult:
        """Synthesize text via OpenAI-compatible TTS API."""
        client = self._get_client()
        model = self._params.get("model", "tts-1")
        voice = kwargs.get("voice") or self._params.get("voice", "alloy")
        response_format = kwargs.get("response_format", "wav")
        speed = kwargs.get("speed", 1.0)

        start = time.monotonic()

        if not output_path.endswith(".mp3") and not output_path.endswith(".wav"):
            output_path = output_path + ".mp3"

        response = client.audio.speech.create(
            model=model,
            voice=voice,
            input=text,
            response_format=response_format,
            speed=speed,
        )

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
        """List OpenAI TTS voices."""
        return [
            {"name": "alloy", "language": "en", "gender": "neutral", "note": "Neutral, versatile"},
            {"name": "echo", "language": "en", "gender": "male", "note": "Warm, older male"},
            {"name": "fable", "language": "en", "gender": "male", "note": "Expressive, storytelling"},
            {"name": "onyx", "language": "en", "gender": "male", "note": "Deep, authoritative male"},
            {"name": "nova", "language": "en", "gender": "female", "note": "Bright, cheerful female"},
            {"name": "shimmer", "language": "en", "gender": "female", "note": "Soft, lyrical female"},
        ]

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

    def health_check(self) -> bool:
        """Check if the API endpoint is reachable."""
        try:
            client = self._get_client()
            client.models.list()
            return True
        except Exception:
            return False
