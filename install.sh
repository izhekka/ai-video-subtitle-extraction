#!/bin/bash

# AI Video Subtitle Extraction Agent - Installation Script

echo "=========================================="
echo "AI Video Subtitle Extraction Agent"
echo "Installation Script"
echo "=========================================="

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.8"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "❌ Python version $PYTHON_VERSION is too old. Please install Python 3.8 or higher."
    exit 1
fi

echo "✅ Python $PYTHON_VERSION detected"

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip."
    exit 1
fi

echo "✅ pip3 detected"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Check if FFmpeg is installed
if ! command -v ffmpeg &> /dev/null; then
    echo "⚠️  FFmpeg is not installed. This is required for video processing."
    echo "Please install FFmpeg:"
    echo "  macOS: brew install ffmpeg"
    echo "  Ubuntu: sudo apt install ffmpeg"
    echo "  Windows: Download from https://ffmpeg.org/download.html"
else
    echo "✅ FFmpeg detected"
fi

# Test installation
echo "🧪 Testing installation..."
python3 -c "
import sys
sys.path.insert(0, 'src')
try:
    from src.utils import setup_logging, get_config
    from src.subtitle_generator import SubtitleGenerator
    from src.language_detector import LanguageDetector
    print('✅ Core modules imported successfully')
except ImportError as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ Installation completed successfully!"
    echo "=========================================="
    echo ""
    echo "To use the subtitle extraction agent:"
    echo ""
    echo "1. Activate the virtual environment:"
    echo "   source venv/bin/activate"
    echo ""
    echo "2. Run the agent:"
    echo "   python main.py --help"
    echo ""
    echo "3. Extract subtitles from a video:"
    echo "   python main.py extract video.mp4 --auto-detect"
    echo ""
    echo "4. For batch processing:"
    echo "   python main.py batch videos/ --language en"
    echo ""
    echo "5. To translate subtitles:"
    echo "   python main.py translate subtitles.srt --target-language es"
    echo ""
    echo "For more information, see the README.md file."
else
    echo ""
    echo "❌ Installation failed. Please check the error messages above."
    exit 1
fi
