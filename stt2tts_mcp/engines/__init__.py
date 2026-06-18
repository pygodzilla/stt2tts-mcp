"""STT and TTS engine abstractions."""

from stt2tts_mcp.engines.stt.base import EngineConfig, STTEngine, TranscriptionResult
from stt2tts_mcp.engines.tts.base import SynthesisResult, TTSEngine
from stt2tts_mcp.engines.stt import STT_ENGINES
from stt2tts_mcp.engines.tts import TTS_ENGINES

__all__ = [
    "EngineConfig",
    "STTEngine",
    "TTSEngine",
    "TranscriptionResult",
    "SynthesisResult",
    "STT_ENGINES",
    "TTS_ENGINES",
]
