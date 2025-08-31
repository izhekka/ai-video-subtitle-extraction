# AI Video Subtitle Extraction Agent

An intelligent Python agent that automatically extracts subtitles from MP4 videos using advanced speech recognition and AI technologies.

## Features

- 🎥 **Video Processing**: Supports various video formats with automatic format detection
- 🗣️ **Speech Recognition**: Uses OpenAI Whisper for high-accuracy speech-to-text conversion
- 🌍 **Multilingual Support**: Automatic language detection and support for 99+ languages
- ⏱️ **Timestamp Extraction**: Precise subtitle timing with frame-accurate synchronization
- 📝 **Multiple Formats**: Export subtitles in SRT, VTT, and TXT formats
- 🔧 **Batch Processing**: Process multiple videos simultaneously
- 🎯 **Quality Optimization**: Audio preprocessing for better recognition accuracy

## Technology Stack

- **Video Processing**: MoviePy, OpenCV
- **Audio Processing**: Librosa, Pydub
- **Speech Recognition**: OpenAI Whisper
- **AI Framework**: PyTorch, Transformers
- **Subtitle Processing**: PySRT, WebVTT
- **Language Detection**: LangDetect, Google Translate

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd ai-video-subtitle-extraction
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install FFmpeg** (required for video processing):
   - **macOS**: `brew install ffmpeg`
   - **Ubuntu**: `sudo apt install ffmpeg`
   - **Windows**: Download from [FFmpeg website](https://ffmpeg.org/download.html)

## Usage

### Basic Usage

```bash
# Extract subtitles from a single video
python main.py extract --video path/to/video.mp4 --language en

# Extract with automatic language detection
python main.py extract --video path/to/video.mp4 --auto-detect

# Process multiple videos
python main.py batch --input-dir videos/ --output-dir subtitles/
```

### Advanced Usage

```bash
# Extract with custom model size (faster/slower)
python main.py extract --video video.mp4 --model-size base

# Export in specific format
python main.py extract --video video.mp4 --format srt

# Translate subtitles
python main.py translate --input subtitles.srt --target-language es
```

## Configuration

Create a `.env` file for custom settings:

```env
# Whisper model settings
WHISPER_MODEL_SIZE=base
WHISPER_DEVICE=cpu

# Audio processing
AUDIO_SAMPLE_RATE=16000
AUDIO_CHUNK_LENGTH=30

# Output settings
DEFAULT_OUTPUT_FORMAT=srt
DEFAULT_LANGUAGE=en
```

## Project Structure

```
video-subtitles/
├── src/
│   ├── __init__.py
│   ├── main.py              # CLI entry point
│   ├── video_processor.py   # Video processing logic
│   ├── audio_processor.py   # Audio extraction and processing
│   ├── speech_recognition.py # Speech-to-text conversion
│   ├── subtitle_generator.py # Subtitle creation and formatting
│   ├── language_detector.py  # Language detection and translation
│   └── utils.py             # Utility functions
├── tests/
│   └── test_*.py
├── examples/
│   └── sample_videos/
├── requirements.txt
├── README.md
└── .env.example
```

## API Reference

### Main Classes

- `VideoProcessor`: Handles video loading and frame extraction
- `AudioProcessor`: Manages audio extraction and preprocessing
- `SpeechRecognizer`: Performs speech-to-text conversion
- `SubtitleGenerator`: Creates and formats subtitle files
- `LanguageDetector`: Detects and translates languages

### Key Methods

```python
from src.video_processor import VideoProcessor
from src.speech_recognition import SpeechRecognizer

# Process video and extract subtitles
processor = VideoProcessor("video.mp4")
audio = processor.extract_audio()

recognizer = SpeechRecognizer()
subtitles = recognizer.transcribe(audio, language="en")
```

## Performance Tips

1. **Model Size**: Use `tiny` or `base` for faster processing, `large` for better accuracy
2. **GPU Acceleration**: Install CUDA for faster processing on NVIDIA GPUs
3. **Batch Processing**: Process multiple videos simultaneously for efficiency
4. **Audio Quality**: Ensure good audio quality for better recognition results

## Troubleshooting

### Common Issues

1. **FFmpeg not found**: Install FFmpeg and ensure it's in your PATH
2. **CUDA errors**: Install PyTorch with CUDA support or use CPU-only version
3. **Memory issues**: Use smaller model sizes or process shorter video segments
4. **Audio quality**: Check video audio track quality and consider preprocessing

### Debug Mode

```bash
python main.py extract --video video.mp4 --debug
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- OpenAI for the Whisper model
- MoviePy team for video processing capabilities
- Hugging Face for transformers library
