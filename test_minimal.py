#!/usr/bin/env python3
"""
Minimal test script for the AI Video Subtitle Extraction Agent.
Tests core functionality without importing heavy dependencies.
"""

import sys
import os
import tempfile
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_utils():
    """Test utility functions."""
    print("Testing utility functions...")

    try:
        from src.utils import (
            format_time,
            format_time_vtt,
            get_supported_formats,
            get_supported_languages,
            get_config
        )

        # Test time formatting
        assert format_time(0) == "00:00:00,000"
        assert format_time(61.5) == "00:01:01,500"
        assert format_time(3661.123) == "01:01:01,123"

        # Test VTT time formatting
        assert format_time_vtt(0) == "00:00:00.000"
        assert format_time_vtt(61.5) == "00:01:01.500"
        assert format_time_vtt(3661.123) == "01:01:01.123"

        # Test supported formats
        formats = get_supported_formats()
        assert 'srt' in formats
        assert 'vtt' in formats
        assert 'txt' in formats
        assert 'json' in formats

        # Test supported languages
        languages = get_supported_languages()
        assert 'en' in languages
        assert 'es' in languages
        assert 'fr' in languages

        # Test config
        config = get_config()
        assert 'whisper_model_size' in config
        assert 'default_output_format' in config

        print("✅ Utility functions test passed")
        return True

    except Exception as e:
        print(f"❌ Utility functions test failed: {e}")
        return False

def test_subtitle_generator():
    """Test subtitle generator functionality."""
    print("Testing subtitle generator...")

    try:
        from src.subtitle_generator import SubtitleGenerator

        generator = SubtitleGenerator()

        # Test data
        test_segments = [
            {
                'start': 0.0,
                'end': 2.5,
                'text': 'Hello, world!',
                'confidence': 0.9
            },
            {
                'start': 2.5,
                'end': 5.0,
                'text': 'This is a test.',
                'confidence': 0.8
            }
        ]

        # Test SRT generation
        with tempfile.NamedTemporaryFile(suffix='.srt', delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            result_path = generator.generate_subtitles(
                test_segments,
                temp_path,
                'srt'
            )

            # Check file was created
            assert os.path.exists(result_path)

            # Check file content
            with open(result_path, 'r', encoding='utf-8') as f:
                content = f.read()

            assert 'Hello, world!' in content
            assert 'This is a test.' in content

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

        # Test validation
        validation = generator.validate_subtitles(test_segments)
        assert validation['valid'] == True
        assert validation['total_segments'] == 2

        print("✅ Subtitle generator test passed")
        return True

    except Exception as e:
        print(f"❌ Subtitle generator test failed: {e}")
        return False

def test_language_detector():
    """Test language detector functionality."""
    print("Testing language detector...")

    try:
        from src.language_detector import LanguageDetector

        detector = LanguageDetector()

        # Test language name retrieval
        assert detector.get_language_name('en') == 'english'
        assert detector.get_language_name('es') == 'spanish'
        assert detector.get_language_name('fr') == 'french'

        # Test language validation
        assert detector.validate_language('en') == True
        assert detector.validate_language('es') == True
        assert detector.validate_language('invalid') == False

        # Test supported languages
        supported = detector.get_supported_languages()
        assert 'en' in supported
        assert 'es' in supported
        assert 'fr' in supported

        # Test language detection (basic)
        lang, confidence = detector.detect_language("Hello, this is English text.")
        assert isinstance(lang, str)
        assert isinstance(confidence, float)
        assert confidence > 0

        # Test statistics
        test_segments = [
            {'text': 'Hello world'},
            {'text': 'This is a test'},
            {'text': 'Another segment'}
        ]

        stats = detector.get_language_statistics(test_segments)
        assert stats['total_segments'] == 3
        assert stats['total_characters'] > 0
        assert stats['total_words'] > 0

        print("✅ Language detector test passed")
        return True

    except Exception as e:
        print(f"❌ Language detector test failed: {e}")
        return False

def test_project_structure():
    """Test project structure and file organization."""
    print("Testing project structure...")

    try:
        # Check required files exist
        required_files = [
            'main.py',
            'requirements.txt',
            'README.md',
            'setup.py',
            'install.sh',
            'src/__init__.py',
            'src/utils.py',
            'src/video_processor.py',
            'src/audio_processor.py',
            'src/speech_recognition.py',
            'src/subtitle_generator.py',
            'src/language_detector.py',
            'src/main.py',
            'tests/test_basic.py',
            'examples/example_usage.py'
        ]

        for file_path in required_files:
            assert os.path.exists(file_path), f"Missing file: {file_path}"

        # Check directory structure
        required_dirs = [
            'src',
            'tests',
            'examples',
            'examples/sample_videos'
        ]

        for dir_path in required_dirs:
            assert os.path.isdir(dir_path), f"Missing directory: {dir_path}"

        print("✅ Project structure test passed")
        return True

    except Exception as e:
        print(f"❌ Project structure test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("AI Video Subtitle Extraction Agent - Minimal Tests")
    print("=" * 60)

    tests = [
        test_utils,
        test_subtitle_generator,
        test_language_detector,
        test_project_structure
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")

    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    print("=" * 60)

    if passed == total:
        print("🎉 All tests passed! The core functionality is working correctly.")
        print("\nTo install dependencies and use the full functionality:")
        print("1. Run: ./install.sh")
        print("2. Activate virtual environment: source venv/bin/activate")
        print("3. Use the agent: python main.py --help")
        print("\nExample usage:")
        print("  python main.py extract video.mp4 --auto-detect")
        print("  python main.py batch videos/ --language en")
        print("  python main.py translate subtitles.srt --target-language es")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
