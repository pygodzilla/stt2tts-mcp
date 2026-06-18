"""Coqui XTTS v2 TTS engine — high quality, multilingual, larger models."""

import time
from pathlib import Path
from typing import Any

from .base import EngineConfig, SynthesisResult, TTSEngine


class CoquiTTSEngine(TTSEngine):
    """Coqui XTTS v2 — high-quality multilingual TTS.

    1.2GB model, 2-4x realtime, 17 languages, excellent quality.
    MPL 2.0 license. Good for when quality > size/speed.
    """

    name = "coqui"

    _model: Any = None
    _config: EngineConfig | None = None

    def __init__(self, config: EngineConfig) -> None:
        self._config = config
        self._params = config.params
        self._model = None

    def _load_model(self) -> Any:
        """Lazy-load the Coqui XTTS model."""
        try:
            from TTS.api import TTS
        except ImportError:
            raise ImportError(
                "Coqui TTS not installed. Run: pip install stt2tts-mcp[tts-coqui]"
            )

        if self._model is not None:
            return self._model

        model_name = self._params.get("model", "tts_models/multilingual/multi-dataset/xtts_v2")
        device = self._params.get("device", "cpu")
        cache_dir = str(Path(self._params.get("cache_dir", "~/.cache/tts")).expanduser())

        self._model = TTS(model_name=model_name, gpu=(device == "cuda"), progress_bar=False)
        return self._model

    def speak(self, text: str, output_path: str, **kwargs) -> SynthesisResult:
        """Synthesize text to speech using Coqui XTTS v2."""
        tts = self._load_model()
        language = kwargs.get("language") or self._params.get("language", "en")
        speaker = kwargs.get("speaker") or self._params.get("speaker", "Female")

        start = time.monotonic()

        if not output_path.endswith(".wav"):
            output_path = output_path.rsplit(".", 1)[0] + ".wav"

        tts.tts_to_file(
            text=text,
            speaker=speaker,
            language=language,
            file_path=output_path,
        )

        duration = time.monotonic() - start

        return SynthesisResult(
            audio_path=output_path,
            duration=duration,
            engine=self.name,
            text_length=len(text),
        )

    def swap_config(self, config: EngineConfig) -> None:
        """Hot-swap: reset model if model name changed."""
        if self._config and self._config.params.get("model") != config.params.get("model"):
            self._model = None
        self._config = config
        self._params = config.params

    def list_voices(self) -> list[dict]:
        """List Coqui XTTS v2 voices (reference)."""
        return [
            {"name": "Female (default)", "language": "en", "gender": "female", "note": "Default English female"},
            {"name": "Male (default)", "language": "en", "gender": "male", "note": "Default English male"},
            {"name": "Spanish Female", "language": "es", "gender": "female", "note": "Spanish female"},
            {"name": "French Female", "language": "fr", "gender": "female", "note": "French female"},
            {"name": "German Female", "language": "de", "gender": "female", "note": "German female"},
            {"name": "Italian Female", "language": "it", "gender": "female", "note": "Italian female"},
            {"name": "Portuguese Female", "language": "pt", "gender": "female", "note": "Portuguese female"},
            {"name": "Polish Female", "language": "pl", "gender": "female", "note": "Polish female"},
            {"name": "Dutch Female", "language": "nl", "gender": "female", "note": "Dutch female"},
            {"name": "Russian Female", "language": "ru", "gender": "female", "note": "Russian female"},
            {"name": "Japanese Female", "language": "ja", "gender": "female", "note": "Japanese female"},
            {"name": "Korean Female", "language": "ko", "gender": "female", "note": "Korean female"},
            {"name": "Chinese Female", "language": "zh", "gender": "female", "note": "Mandarin Chinese female"},
            {"name": "Hindi Female", "language": "hi", "gender": "female", "note": "Hindi female"},
            {"name": "Arabic Female", "language": "ar", "gender": "female", "note": "Arabic female"},
            {"name": "Czech Female", "language": "cs", "gender": "female", "note": "Czech female"},
            {"name": "Hungarian Female", "language": "hu", "gender": "female", "note": "Hungarian female"},
        ]

    def health_check(self) -> bool:
        """Check if TTS is installed."""
        try:
            from TTS.api import TTS  # noqa: F401
            return True
        except ImportError:
            return False
