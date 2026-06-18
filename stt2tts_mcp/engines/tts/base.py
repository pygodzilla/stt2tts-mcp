"""TTS engine base — re-exports shared types and defines TTSEngine."""

from abc import ABC, abstractmethod
from typing import Any

from stt2tts_mcp.engines.stt.base import EngineConfig, SynthesisResult


class TTSEngine(ABC):
    """Abstract base class for all TTS engines."""

    name: str

    def __init__(self, config: EngineConfig) -> None:
        self._config = config
        self._params: dict[str, Any] = config.params

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
        self._config = config
        self._params = config.params

    def health_check(self) -> bool:
        """Check if the engine is reachable/operational."""
        return True


__all__ = ["EngineConfig", "SynthesisResult", "TTSEngine"]
