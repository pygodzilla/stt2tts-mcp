"""Piper TTS engine — small, fast, high-quality local TTS."""

import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from .base import EngineConfig, SynthesisResult, TTSEngine


class PiperTTSEngine(TTSEngine):
    """Piper TTS engine — ONNX-based neural TTS.

    Very small models (20-50MB), 10-20x realtime, Apache 2.0 license.
    Excellent quality for the size. Run `piper --voices` to list all voices.
    """

    name = "piper"

    _config: EngineConfig | None = None
    _process: Any = None

    def __init__(self, config: EngineConfig) -> None:
        self._config = config
        self._params = config.params
        self._exe_path: str | None = None

    def _find_exe(self) -> str:
        """Find the piper CLI executable."""
        if self._exe_path:
            return self._exe_path

        # Check common locations
        candidates = [
            Path("~/Library/Piper/piper").expanduser(),
            Path("/usr/local/bin/piper"),
            Path("/usr/bin/piper"),
            Path("~/.local/bin/piper"),
        ]

        for candidate in candidates:
            if candidate.exists():
                self._exe_path = str(candidate)
                return self._exe_path

        # Try PATH lookup
        result = subprocess.run(["which", "piper"], capture_output=True, text=True)
        if result.returncode == 0:
            self._exe_path = result.stdout.strip()
            return self._exe_path

        raise RuntimeError(
            "piper not found. Install: pip install piper-tts or download from "
            "https://github.com/rhasspy/piper/releases"
        )

    def speak(self, text: str, output_path: str, **kwargs) -> SynthesisResult:
        """Synthesize text to speech using Piper TTS."""
        voice = kwargs.get("voice") or self._params.get("voice", "en_US-lessac-medium")
        model_dir = Path(self._params.get("model_dir", "~/.cache/piper")).expanduser()
        sentence_samples = self._params.get("sentence_samples", 100)

        start = time.monotonic()

        # Ensure output has .wav extension
        if not output_path.endswith(".wav"):
            output_path = output_path.rsplit(".", 1)[0] + ".wav"

        exe = self._find_exe()
        model_path = model_dir / f"{voice}.onnx"
        json_path = model_dir / f"{voice}.onnx.json"

        cmd = [
            exe,
            "--model", str(model_path),
            "--json",
            "--output_file", output_path,
        ]



        if json_path.exists():
            cmd.extend(["--config", str(json_path)])

        try:
            result = subprocess.run(
                cmd,
                input=text,
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Piper failed: {result.stderr}")
        except FileNotFoundError:
            raise RuntimeError(
                f"Piper model not found at {model_path}. "
                f"Download voices from: https://github.com/rhasspy/piper/releases"
            )

        duration = time.monotonic() - start

        return SynthesisResult(
            audio_path=output_path,
            duration=duration,
            engine=self.name,
            text_length=len(text),
        )

    def list_voices(self) -> list[dict]:
        """List Piper voices (sample — full list via `piper --voices`)."""
        return [
            {"name": "en_US-lessac-medium", "language": "en", "gender": "male", "note": "Deep male voice, medium quality"},
            {"name": "en_US-lessac-low", "language": "en", "gender": "male", "note": "Low male voice"},
            {"name": "en_US-amy-medium", "language": "en", "gender": "female", "note": "Female voice, medium quality"},
            {"name": "en_US-kathleen-low", "language": "en", "gender": "female", "note": "Low female voice"},
            {"name": "en_GB-cori-medium", "language": "en-GB", "gender": "female", "note": "British female"},
            {"name": "de_DE-eva-x_low", "language": "de", "gender": "female", "note": "German female, very low pitch"},
            {"name": "es_ES-maria-medium", "language": "es", "gender": "female", "note": "Spanish female"},
            {"name": "fr_FR-siwis-medium", "language": "fr", "gender": "female", "note": "French female"},
            {"name": "it_IT-riccardo-low", "language": "it", "gender": "male", "note": "Italian male, low pitch"},
        ]

    def swap_config(self, config: EngineConfig) -> None:
        """Hot-swap: no restart needed for Piper."""
        self._config = config
        self._params = config.params

    def health_check(self) -> bool:
        """Check if piper is installed and voice model exists."""
        try:
            exe = self._find_exe()
            voice = self._params.get("voice", "en_US-lessac-medium")
            model_dir = Path(self._params.get("model_dir", "~/.cache/piper")).expanduser()
            model_path = model_dir / f"{voice}.onnx"
            return model_path.exists()
        except Exception:
            return False
