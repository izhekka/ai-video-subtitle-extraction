"""
Basic unit tests for the AI Video Subtitle Extraction Agent.
"""

import unittest
import tempfile
import os
from pathlib import Path

# Add src to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.utils import (
    validate_video_file,
    validate_language_code,
    get_config,
    format_time,
    format_time_vtt
)
from src.subtitle_generator import SubtitleGenerator
from src.language_detector import LanguageDetector


class TestUtils(unittest.TestCase):
    """Test utility functions."""

    def test_format_time(self):
        """Test SRT time formatting."""
        # Test various time values
        self.assertEqual(format_time(0), "00:00:00,000")
        self.assertEqual(format_time(61.5), "00:01:01,500")
        self.assertEqual(format_time(3661.123), "01:01:01,123")

    def test_format_time_vtt(self):
        """Test VTT time formatting."""
        # Test various time values
        self.assertEqual(format_time_vtt(0), "00:00:00.000")
        self.assertEqual(format_time_vtt(61.5), "00:01:01.500")
        self.assertEqual(format_time_vtt(3661.123), "01:01:01.123")

    def test_validate_language_code(self):
        """Test language code validation."""
        # Valid languages
        self.assertTrue(validate_language_code('en'))
        self.assertTrue(validate_language_code('es'))
        self.assertTrue(validate_language_code('fr'))

        # Invalid languages
        with self.assertRaises(ValueError):
            validate_language_code('invalid')
        with self.assertRaises(ValueError):
            validate_language_code('xx')

    def test_get_config(self):
        """Test configuration loading."""
        config = get_config()

        # Check required keys
        required_keys = [
            'whisper_model_size',
            'whisper_device',
            'audio_sample_rate',
            'default_output_format',
            'default_language'
        ]

        for key in required_keys:
            self.assertIn(key, config)


class TestSubtitleGenerator(unittest.TestCase):
    """Test subtitle generation functionality."""

    def setUp(self):
        """Set up test data."""
        self.generator = SubtitleGenerator()
        self.test_segments = [
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

    def test_generate_srt(self):
        """Test SRT subtitle generation."""
        with tempfile.NamedTemporaryFile(suffix='.srt', delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            result_path = self.generator.generate_subtitles(
                self.test_segments,
                temp_path,
                'srt'
            )

            # Check file was created
            self.assertTrue(os.path.exists(result_path))

            # Check file content
            with open(result_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Should contain subtitle content
            self.assertIn('Hello, world!', content)
            self.assertIn('This is a test.', content)

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_generate_txt(self):
        """Test TXT subtitle generation."""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            result_path = self.generator.generate_subtitles(
                self.test_segments,
                temp_path,
                'txt'
            )

            # Check file was created
            self.assertTrue(os.path.exists(result_path))

            # Check file content
            with open(result_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Should contain subtitle content
            self.assertIn('Hello, world!', content)
            self.assertIn('This is a test.', content)

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_validate_subtitles(self):
        """Test subtitle validation."""
        # Valid segments
        validation = self.generator.validate_subtitles(self.test_segments)
        self.assertTrue(validation['valid'])
        self.assertEqual(validation['total_segments'], 2)

        # Invalid segments (missing fields)
        invalid_segments = [
            {'start': 0.0, 'end': 2.5},  # Missing text
            {'start': 2.5, 'text': 'Test'}  # Missing end
        ]

        validation = self.generator.validate_subtitles(invalid_segments)
        self.assertFalse(validation['valid'])
        self.assertGreater(len(validation['issues']), 0)


class TestLanguageDetector(unittest.TestCase):
    """Test language detection functionality."""

    def setUp(self):
        """Set up test data."""
        self.detector = LanguageDetector()

    def test_detect_language(self):
        """Test language detection."""
        # Test English text
        lang, confidence = self.detector.detect_language("Hello, this is English text.")
        self.assertIsInstance(lang, str)
        self.assertIsInstance(confidence, float)
        self.assertGreater(confidence, 0)

    def test_get_language_name(self):
        """Test language name retrieval."""
        # Test known languages
        self.assertEqual(self.detector.get_language_name('en'), 'english')
        self.assertEqual(self.detector.get_language_name('es'), 'spanish')
        self.assertEqual(self.detector.get_language_name('fr'), 'french')

        # Test unknown language
        self.assertEqual(self.detector.get_language_name('xx'), 'xx')

    def test_validate_language(self):
        """Test language validation."""
        # Valid languages
        self.assertTrue(self.detector.validate_language('en'))
        self.assertTrue(self.detector.validate_language('es'))
        self.assertTrue(self.detector.validate_language('fr'))

        # Invalid languages
        self.assertFalse(self.detector.validate_language('invalid'))
        self.assertFalse(self.detector.validate_language('xx'))

    def test_get_language_statistics(self):
        """Test language statistics calculation."""
        test_segments = [
            {'text': 'Hello world'},
            {'text': 'This is a test'},
            {'text': 'Another segment'}
        ]

        stats = self.detector.get_language_statistics(test_segments)

        # Check required fields
        required_fields = [
            'total_segments',
            'total_characters',
            'total_words',
            'detected_language'
        ]

        for field in required_fields:
            self.assertIn(field, stats)

        # Check values
        self.assertEqual(stats['total_segments'], 3)
        self.assertGreater(stats['total_characters'], 0)
        self.assertGreater(stats['total_words'], 0)


if __name__ == '__main__':
    unittest.main()
