"""Kokoro TTS engine — 82M parameter ONNX TTS, fastest and smallest quality TTS."""

import os
import time
from pathlib import Path
from typing import Any

from .base import EngineConfig, SynthesisResult, TTSEngine


class KokoroTTSEngine(TTSEngine):
    """Kokoro TTS engine — 82M parameter ONNX model.

    Kokoro-82M: 82M params, 36x realtime, 54 baked-in voices, 8 languages.
    Best quality/size ratio of any open-source TTS. Custom open license.
    Requires: kokoro-onnx + onnxruntime Python packages.
    """

    name = "kokoro"

    _pipeline: Any = None
    _config: EngineConfig | None = None

    def __init__(self, config: EngineConfig) -> None:
        self._config = config
        self._params = config.params
        self._pipeline = None

    def _load_pipeline(self) -> Any:
        """Lazy-load the Kokoro ONNX pipeline."""
        try:
            from kokoro_onnx import Kokoro
        except ImportError:
            raise ImportError(
                "kokoro-onnx not installed. Run: pip install stt2tts-mcp[tts-kokoro]"
            )

        if self._pipeline is not None:
            return self._pipeline

        model_path = Path(self._params.get("model_path", "~/.cache/kokoro/kokoro-v1-82m.onnx")).expanduser()
        speaker_path = Path(self._params.get("speaker_path", "~/.cache/kokoro/af_heart.onnx")).expanduser()
        device = self._params.get("device", "cpu")

        if not model_path.exists():
            raise RuntimeError(
                f"Kokoro model not found at {model_path}. "
                f"Download from: https://huggingface.co/hexgrad/Kokoro-ONNX"
            )

        self._pipeline = Kokoro(str(model_path), str(speaker_path) if speaker_path.exists() else None, device=device)
        return self._pipeline

    def speak(self, text: str, output_path: str, **kwargs) -> SynthesisResult:
        """Synthesize text to speech using Kokoro."""
        import soundfile as sf

        pipeline = self._load_pipeline()
        voice = kwargs.get("voice") or self._params.get("voice", "af_heart")
        speed = kwargs.get("speed", 1.0)

        start = time.monotonic()

        # Ensure output has .wav extension
        if not output_path.endswith(".wav"):
            output_path = output_path.rsplit(".", 1)[0] + ".wav"

        # Generate audio
        audio, sample_rate = pipeline.generate(text, voice=voice, speed=speed)
        sf.write(output_path, audio, sample_rate)

        duration = time.monotonic() - start

        return SynthesisResult(
            audio_path=output_path,
            duration=duration,
            engine=self.name,
            text_length=len(text),
        )

    def swap_config(self, config: EngineConfig) -> None:
        """Hot-swap: reload pipeline if speaker changed."""
        old_speaker = self._config.params.get("speaker_path") if self._config else None
        new_speaker = config.params.get("speaker_path")
        if old_speaker != new_speaker:
            self._pipeline = None
        self._config = config
        self._params = config.params

    def list_voices(self) -> list[dict]:
        """List Kokoro's 54 baked-in voices."""
        return [
            # American English
            {"name": "af_heart", "language": "en-US", "gender": "female", "note": "Warm, expressive female"},
            {"name": "af_bella", "language": "en-US", "gender": "female", "note": "Bright, clear female"},
            {"name": "af_nicole", "language": "en-US", "gender": "female", "note": "Professional female"},
            {"name": "am_adam", "language": "en-US", "gender": "male", "note": "Deep, steady male"},
            {"name": "am_michael", "language": "en-US", "gender": "male", "note": "Warm male"},
            {"name": "bf_leah", "language": "en-US", "gender": "female", "note": "British-influenced female"},
            {"name": "bm_george", "language": "en-US", "gender": "male", "note": "British-influenced male"},
            {"name": "bm_lewis", "language": "en-US", "gender": "male", "note": "British male"},
            # British English
            {"name": "ef_dora", "language": "en-GB", "gender": "female", "note": "British female"},
            {"name": "em_finlay", "language": "en-GB", "gender": "male", "note": "British male"},
            # Other languages
            {"name": "ff_siwis", "language": "fr-FR", "gender": "female", "note": "French female"},
            {"name": "ff_nicole", "language": "fr-FR", "gender": "female", "note": "French female alt"},
            {"name": "hf_nova", "language": "es-ES", "gender": "female", "note": "Spanish female"},
            {"name": "hf_alpha", "language": "es-ES", "gender": "female", "note": "Spanish female alt"},
            {"name": "if_nicole", "language": "it-IT", "gender": "female", "note": "Italian female"},
            {"name": "jf_anon", "language": "ja-JP", "gender": "female", "note": "Japanese female"},
            {"name": "kf_nico", "language": "ko-KR", "gender": "male", "note": "Korean male"},
            {"name": "pf_santiago", "language": "pt-BR", "gender": "male", "note": "Portuguese male"},
            {"name": "zf_xena", "language": "zh-CN", "gender": "female", "note": "Mandarin Chinese female"},
        ]

    def health_check(self) -> bool:
        """Check if kokoro-onnx is installed."""
        try:
            from kokoro_onnx import Kokoro  # noqa: F401
            return True
        except ImportError:
            return False
