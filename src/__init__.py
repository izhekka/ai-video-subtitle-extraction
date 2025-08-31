"""
AI Video Subtitle Extraction Agent

An intelligent Python agent that automatically extracts subtitles from MP4 videos
using advanced speech recognition and AI technologies.
"""

__version__ = "1.0.0"
__author__ = "AI Video Subtitle Agent"
__description__ = "Automatic subtitle extraction from videos using AI"

# Import core modules that don't require heavy dependencies
from .utils import *
from .subtitle_generator import SubtitleGenerator
from .language_detector import LanguageDetector

# Import modules that require heavy dependencies (optional)
try:
    from .video_processor import VideoProcessor
    from .audio_processor import AudioProcessor
    from .speech_recognition import SpeechRecognizer
    __all__ = [
        "VideoProcessor",
        "AudioProcessor",
        "SpeechRecognizer",
        "SubtitleGenerator",
        "LanguageDetector"
    ]
except ImportError:
    # If heavy dependencies are not available, only export core modules
    __all__ = [
        "SubtitleGenerator",
        "LanguageDetector"
    ]
