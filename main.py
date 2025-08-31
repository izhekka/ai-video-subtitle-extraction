#!/usr/bin/env python3
"""
AI Video Subtitle Extraction Agent - Main Entry Point

Extract subtitles from MP4 videos using advanced AI speech recognition.
"""

import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.main import cli

if __name__ == '__main__':
    cli()
