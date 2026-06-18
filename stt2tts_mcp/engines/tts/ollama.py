"""Ollama TTS engine — text-to-speech via Ollama's built-in TTS."""

import time
from typing import Any

import requests

from .base import EngineConfig, SynthesisResult, TTSEngine


class OllamaTTSEngine(TTSEngine):
    """Ollama TTS engine — uses Ollama's built-in TTS generation.

    Default: http://localhost:11434
    Note: Ollama's native TTS is model-dependent. Set model in config.yaml.
    """

    name = "ollama"

    _config: EngineConfig | None = None

    def __init__(self, config: EngineConfig) -> None:
        self._config = config
        self._params = config.params

    def speak(self, text: str, output_path: str, **kwargs) -> SynthesisResult:
        """Synthesize text via Ollama TTS."""
        base_url = self._params.get("base_url", "http://localhost:11434")
        model = self._params.get("model", "llama3")
        timeout = self._params.get("timeout", 30)
        voice = kwargs.get("voice") or self._params.get("voice", "alloy")

        start = time.monotonic()

        if not output_path.endswith(".wav"):
            output_path = output_path.rsplit(".", 1)[0] + ".wav"

        # Ollama TTS: POST /api/generate with speech output
        response = requests.post(
            f"{base_url}/api/generate",
            json={
                "model": model,
                "prompt": f"[SPEAK]{text}[/SPEAK]",
                "stream": False,
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
        """List available Ollama TTS voices (model-dependent)."""
        return [
            {"name": "alloy", "language": "en", "gender": "neutral", "note": "Default voice"},
            {"name": "nova", "language": "en", "gender": "female", "note": "Cheerful female"},
            {"name": "shimmer", "language": "en", "gender": "female", "note": "Soft female"},
        ]

    def swap_config(self, config: EngineConfig) -> None:
        """Hot-swap: no restart needed."""
        self._config = config
        self._params = config.params

    def health_check(self) -> bool:
        """Check if Ollama server is reachable."""
        base_url = self._params.get("base_url", "http://localhost:11434")
        try:
            response = requests.get(f"{base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
