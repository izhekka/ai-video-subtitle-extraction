"""
Utility functions for the AI Video Subtitle Extraction Agent.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def setup_logging(level: str = "INFO", debug: bool = False) -> logging.Logger:
    """Setup logging configuration."""
    if debug:
        level = "DEBUG"

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('subtitle_agent.log')
        ]
    )
    return logging.getLogger(__name__)

def validate_video_file(file_path: str) -> bool:
    """Validate if the file is a supported video format."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Video file not found: {file_path}")

    supported_formats = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
    file_ext = Path(file_path).suffix.lower()

    if file_ext not in supported_formats:
        raise ValueError(f"Unsupported video format: {file_ext}. Supported formats: {supported_formats}")

    return True

def validate_language_code(language: str) -> bool:
    """Validate language code format."""
    # Common language codes (ISO 639-1)
    supported_languages = {
        'en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh', 'ar', 'hi',
        'bn', 'ur', 'tr', 'nl', 'pl', 'sv', 'da', 'no', 'fi', 'cs', 'sk', 'hu',
        'ro', 'bg', 'hr', 'sr', 'sl', 'et', 'lv', 'lt', 'mt', 'ga', 'cy', 'eu',
        'ca', 'gl', 'is', 'fo', 'sq', 'mk', 'bs', 'me', 'sq', 'mk', 'bs', 'me'
    }

    if language.lower() not in supported_languages:
        raise ValueError(f"Unsupported language: {language}")

    return True

def get_config() -> Dict[str, Any]:
    """Get configuration from environment variables."""
    return {
        'whisper_model_size': os.getenv('WHISPER_MODEL_SIZE', 'base'),
        'whisper_device': os.getenv('WHISPER_DEVICE', 'cpu'),
        'audio_sample_rate': int(os.getenv('AUDIO_SAMPLE_RATE', '16000')),
        'audio_chunk_length': int(os.getenv('AUDIO_CHUNK_LENGTH', '30')),
        'default_output_format': os.getenv('DEFAULT_OUTPUT_FORMAT', 'srt'),
        'default_language': os.getenv('DEFAULT_LANGUAGE', 'en'),
        'debug': os.getenv('DEBUG', 'false').lower() == 'true'
    }

def format_time(seconds: float) -> str:
    """Format seconds to SRT timestamp format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millisecs = int((seconds % 1) * 1000)

    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"

def format_time_vtt(seconds: float) -> str:
    """Format seconds to VTT timestamp format (HH:MM:SS.mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millisecs = int((seconds % 1) * 1000)

    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millisecs:03d}"

def ensure_directory(directory: str) -> None:
    """Ensure directory exists, create if it doesn't."""
    Path(directory).mkdir(parents=True, exist_ok=True)

def get_file_info(file_path: str) -> Dict[str, Any]:
    """Get basic file information."""
    path = Path(file_path)
    stat = path.stat()

    return {
        'name': path.name,
        'stem': path.stem,
        'suffix': path.suffix,
        'size_mb': stat.st_size / (1024 * 1024),
        'modified': stat.st_mtime,
        'exists': path.exists()
    }

def save_json(data: Dict[str, Any], file_path: str) -> None:
    """Save data to JSON file."""
    ensure_directory(os.path.dirname(file_path))
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_json(file_path: str) -> Dict[str, Any]:
    """Load data from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_supported_formats() -> List[str]:
    """Get list of supported subtitle output formats."""
    return ['srt', 'vtt', 'txt', 'json']

def get_supported_languages() -> List[str]:
    """Get list of supported languages."""
    return [
        'en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh', 'ar', 'hi',
        'bn', 'ur', 'tr', 'nl', 'pl', 'sv', 'da', 'no', 'fi', 'cs', 'sk', 'hu',
        'ro', 'bg', 'hr', 'sr', 'sl', 'et', 'lv', 'lt', 'mt', 'ga', 'cy', 'eu',
        'ca', 'gl', 'is', 'fo', 'sq', 'mk', 'bs', 'me'
    ]
