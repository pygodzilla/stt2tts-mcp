"""STT engine implementations."""

from .faster_whisper import FasterWhisperEngine
from .sherpa_onnx import SherpaOnnxEngine
from .openai_api import OpenAISTTEngine
from .ollama import OllamaSTTEngine
from .lmstudio import LMStudioSTTEngine

STT_ENGINES = {
    "faster_whisper": FasterWhisperEngine,
    "sherpa_onnx": SherpaOnnxEngine,
    "openai_api": OpenAISTTEngine,
    "ollama": OllamaSTTEngine,
    "lmstudio": LMStudioSTTEngine,
}

__all__ = [
    "FasterWhisperEngine",
    "SherpaOnnxEngine",
    "OpenAISTTEngine",
    "OllamaSTTEngine",
    "LMStudioSTTEngine",
    "STT_ENGINES",
]
