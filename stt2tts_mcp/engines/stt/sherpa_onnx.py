"""sherpa-onnx STT engine — ultra-lightweight ONNX-based Whisper."""

from pathlib import Path
from typing import Any

from .base import EngineConfig, STTEngine, TranscriptionResult


class SherpaOnnxEngine(STTEngine):
    """sherpa-onnx: k2-fsa's ONNX Whisper implementation.

    Smallest model sizes (~40MB tiny). Excellent for CPU-only, low-latency.
    Supports streaming and non-streaming. Multiple pre-built model variants.
    """

    name = "sherpa_onnx"
    supports_streaming = True

    _model: Any = None
    _config: EngineConfig | None = None

    def __init__(self, config: EngineConfig) -> None:
        self._config = config
        self._params = config.params
        self._model = None

    def _load_model(self) -> Any:
        """Lazy-load the sherpa-onnx model."""
        try:
            import sherpa_onnx
        except ImportError:
            raise ImportError(
                "sherpa-onnx not installed. Run: pip install stt2tts-mcp[stt-sherpa]"
            )

        if self._model is not None:
            return self._model

        model = self._params.get("model", "zeinelevink/sherpa-onnx-whisper-tiny")
        language = self._params.get("language", "en")
        device = self._params.get("device", "cpu")
        download_dir = str(Path(self._params.get("download_dir", "~/.cache/sherpa-onnx")).expanduser())

        # Use offline model if downloaded, otherwise stream from HuggingFace
        model_path = self._params.get("model_path")
        if model_path and Path(model_path).exists():
            recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
                model_path=model_path,
                device=device,
            )
        else:
            # Auto-download from HuggingFace
            recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
                model=model,
                language=language,
                device=device,
                download_dir=download_dir,
            )

        self._model = recognizer
        return self._model

    def transcribe(self, audio_path: str, **kwargs) -> TranscriptionResult:
        """Transcribe audio file using sherpa-onnx."""
        import time

        recognizer = self._load_model()
        start = time.monotonic()

        waves = []
        try:
            import scipy.io.wavfile as wavfile
            sr, data = wavfile.read(audio_path)
            waves.append(sherpa_onnx.OfflineRecognitionResult(
                stream=sherpa_onnx.OfflineStream(samples=data, sample_rate=sr)
            ))
        except Exception:
            # Fallback: try soundfile
            import soundfile as sf
            data, sr = sf.read(audio_path)
            waves.append(sherpa_onnx.OfflineRecognitionResult(
                stream=sherpa_onnx.OfflineStream(samples=data, sample_rate=sr)
            ))

        text = recognizer(waves)
        duration = time.monotonic() - start

        return TranscriptionResult(
            text=text.strip() if text else "",
            language=self._params.get("language", "en"),
            duration=duration,
            engine=self.name,
        )

    def swap_config(self, config: EngineConfig) -> None:
        """Hot-swap: reset model if model path changed."""
        if self._config and self._config.params.get("model") != config.params.get("model"):
            self._model = None
        self._config = config
        self._params = config.params

    def list_models(self) -> list[dict]:
        """List available sherpa-onnx Whisper models."""
        return [
            {
                "name": "sherpa-onnx-whisper-tiny",
                "size_mb": 40,
                "languages": ["en"],
                "params": "39M",
                "note": "Smallest, ultra-fast, English-only",
            },
            {
                "name": "sherpa-onnx-whisper-tiny-multilingual",
                "size_mb": 40,
                "languages": ["many"],
                "params": "39M",
                "note": "Tiny multilingual variant",
            },
            {
                "name": "sherpa-onnx-whisper-base",
                "size_mb": 140,
                "languages": ["en", "many"],
                "params": "74M",
                "note": "Base model, good accuracy/size",
            },
            {
                "name": "sherpa-onnx-whisper-small",
                "size_mb": 465,
                "languages": ["many"],
                "params": "244M",
                "note": "Small multilingual, higher accuracy",
            },
        ]

    def health_check(self) -> bool:
        """Check if sherpa-onnx is installed."""
        try:
            import sherpa_onnx  # noqa: F401
            return True
        except ImportError:
            return False
