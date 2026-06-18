"""faster-whisper STT engine — CTranslate2-based, fastest local Whisper."""

from pathlib import Path
from typing import Any

from .base import EngineConfig, STTEngine, TranscriptionResult


class FasterWhisperEngine(STTEngine):
    """faster-whisper: CTranslate2-based Whisper implementation.

    Fastest local STT engine. Uses INT8 quantized models for CPU inference.
    Supports English-only variants (tiny.en, base.en, etc.) for better accuracy.
    """

    name = "faster_whisper"
    supports_streaming = False

    _model: Any = None
    _config: EngineConfig | None = None

    def __init__(self, config: EngineConfig) -> None:
        self._config = config
        self._params = config.params
        self._model = None

    def _load_model(self) -> Any:
        """Lazy-load the Whisper model."""
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise ImportError(
                "faster-whisper not installed. Run: pip install stt2tts-mcp[stt-faster-whisper]"
            )

        if self._model is not None:
            return self._model

        model_size = self._params.get("model_size", "tiny.en")
        device = self._params.get("device", "cpu")
        compute_type = self._params.get("compute_type", "int8")
        download_dir = str(Path(self._params.get("download_dir", "~/.cache/whisper")).expanduser())

        self._model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            download_root=download_dir,
        )
        return self._model

    def transcribe(self, audio_path: str, **kwargs) -> TranscriptionResult:
        """Transcribe audio file using faster-whisper."""
        import time

        model = self._load_model()
        language = kwargs.get("language") or self._params.get("language") or "en"
        task = kwargs.get("task", "transcribe")
        beam_size = kwargs.get("beam_size", 5)
        vad_filter = kwargs.get("vad_filter", True)

        start = time.monotonic()
        segments, info = model.transcribe(
            audio_path,
            language=language,
            task=task,
            beam_size=beam_size,
            vad_filter=vad_filter,
        )

        text = "".join(seg.text for seg in segments)
        duration = time.monotonic() - start

        return TranscriptionResult(
            text=text.strip(),
            language=info.language if hasattr(info, "language") else language,
            duration=duration,
            engine=self.name,
        )

    def swap_config(self, config: EngineConfig) -> None:
        """Hot-swap: unload model if model_size changed."""
        if self._config and self._config.params.get("model_size") != config.params.get("model_size"):
            self._model = None
        self._config = config
        self._params = config.params

    def list_models(self) -> list[dict]:
        """List available faster-whisper model sizes."""
        return [
            {
                "name": "tiny.en",
                "size_mb": 75,
                "languages": ["en"],
                "params": "39M",
                "note": "Fastest, English-only, good for clean audio",
            },
            {
                "name": "base.en",
                "size_mb": 140,
                "languages": ["en"],
                "params": "74M",
                "note": "Best accuracy/speed for English",
            },
            {
                "name": "small.en",
                "size_mb": 465,
                "languages": ["en"],
                "params": "244M",
                "note": "Higher accuracy, slower",
            },
            {
                "name": "medium.en",
                "size_mb": 1500,
                "languages": ["en"],
                "params": "769M",
                "note": "Very high accuracy, requires more RAM",
            },
            {
                "name": "tiny",
                "size_mb": 75,
                "languages": ["many"],
                "params": "39M",
                "note": "Multilingual, slightly lower accuracy than tiny.en",
            },
            {
                "name": "base",
                "size_mb": 140,
                "languages": ["many"],
                "params": "74M",
                "note": "Multilingual base model",
            },
        ]

    def health_check(self) -> bool:
        """Check if faster-whisper is installed."""
        try:
            from faster_whisper import WhisperModel  # noqa: F401
            return True
        except ImportError:
            return False
