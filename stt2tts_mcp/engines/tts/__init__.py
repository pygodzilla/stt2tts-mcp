"""TTS engine implementations."""

from .piper import PiperTTSEngine
from .kokoro import KokoroTTSEngine
from .coqui import CoquiTTSEngine
from .openai_api import OpenAITTSEngine
from .ollama import OllamaTTSEngine
from .lmstudio import LMStudioTTSEngine

TTS_ENGINES = {
    "piper": PiperTTSEngine,
    "kokoro": KokoroTTSEngine,
    "coqui": CoquiTTSEngine,
    "openai_api": OpenAITTSEngine,
    "ollama": OllamaTTSEngine,
    "lmstudio": LMStudioTTSEngine,
}

__all__ = [
    "PiperTTSEngine",
    "KokoroTTSEngine",
    "CoquiTTSEngine",
    "OpenAITTSEngine",
    "OllamaTTSEngine",
    "LMStudioTTSEngine",
    "TTS_ENGINES",
]
