"""Audio utilities — format conversion, silence trimming, validation."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Literal


def convert_to_wav(
    audio_path: str | Path,
    output_path: str | Path | None = None,
    sample_rate: int = 16000,
    mono: bool = True,
) -> str:
    """Convert any audio file to WAV at 16kHz mono PCM.

    Uses ffmpeg. Raises RuntimeError if ffmpeg is not available.
    """
    audio_path = Path(audio_path)
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
    output_path = str(output_path)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(audio_path),
        "-ar", str(sample_rate),
    ]
    if mono:
        cmd.extend(["-ac", "1"])
    cmd.extend(["-acodec", "pcm_s16le", output_path])

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg conversion failed: {result.stderr.strip()}")
    return output_path


def get_audio_duration(audio_path: str | Path) -> float:
    """Get duration of an audio file in seconds using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr.strip()}")
    try:
        return float(result.stdout.strip())
    except ValueError:
        raise RuntimeError(f"Could not parse duration from ffprobe output: {result.stdout}")


def validate_audio_file(audio_path: str | Path) -> tuple[bool, str]:
    """Check if a file is a valid audio file.

    Returns (is_valid, reason).
    """
    path = Path(audio_path)
    if not path.exists():
        return False, f"File not found: {audio_path}"
    if path.stat().st_size == 0:
        return False, "File is empty"
    # Basic extension check
    ext = path.suffix.lower()
    if ext not in (".wav", ".mp3", ".ogg", ".flac", ".m4a", ".wma", ".aac"):
        return False, f"Unsupported audio format: {ext}"
    return True, "ok"


def trim_silence(
    audio_path: str | Path,
    output_path: str | Path,
    threshold_db: float = -40.0,
    min_duration_ms: int = 200,
) -> str:
    """Trim leading and trailing silence from audio using ffmpeg silenceremove."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(audio_path),
        "-af", f"silenceremove=start_periods=1:start_duration=0.05:"
               f"start_threshold={threshold_db}dB:detection=peak,"
               f"stop_periods=-1:stop_duration={min_duration_ms}ms:"
               f"stop_threshold={threshold_db}dB:detection=peak",
        "-acodec", "pcm_s16le",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"trim_silence failed: {result.stderr.strip()}")
    return str(output_path)


def bytes_to_audio_format(
    data: bytes,
    format: Literal["wav", "mp3", "ogg", "flac"],
) -> bytes:
    """Convert raw audio bytes to a specific format using ffmpeg pipe."""
    raise NotImplementedError("bytes_to_audio_format not yet implemented")
