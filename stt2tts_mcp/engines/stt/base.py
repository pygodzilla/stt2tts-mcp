"""STT and TTS engine abstractions."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class EngineConfig:
    """Hot-swappable engine configuration loaded from config.yaml."""

    name: str
    enabled: bool
    params: dict[str, Any]


@dataclass
class TranscriptionResult:
    """Result from an STT engine."""

    text: str
    language: str | None
    duration: float | None
    engine: str


@dataclass
class SynthesisResult:
    """Result from a TTS engine."""

    audio_path: str | None
    duration: float | None
    engine: str
    text_length: int


class STTEngine(ABC):
    """Abstract base class for all STT engines."""

    name: str
    supports_streaming: bool = False

    @abstractmethod
    def transcribe(self, audio_path: str, **kwargs) -> TranscriptionResult:
        """Transcribe an audio file to text."""
        ...

    def swap_config(self, config: EngineConfig) -> None:
        """Hot-swap engine configuration without restart."""
        ...

    @abstractmethod
    def list_models(self) -> list[dict]:
        """List available models for this engine."""
        ...

    def health_check(self) -> bool:
        """Check if the engine is reachable/operational."""
        return True


class TTSEngine(ABC):
    """Abstract base class for all TTS engines."""

    name: str

    @abstractmethod
    def speak(self, text: str, output_path: str, **kwargs) -> SynthesisResult:
        """Synthesize text to speech and save to output_path."""
        ...

    @abstractmethod
    def list_voices(self) -> list[dict]:
        """List available voices for this engine."""
        ...

    def swap_config(self, config: EngineConfig) -> None:
        """Hot-swap engine configuration without restart."""
        ...

    def health_check(self) -> bool:
        """Check if the engine is reachable/operational."""
        return True
