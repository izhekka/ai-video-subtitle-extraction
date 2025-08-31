"""
Language detection and translation module for subtitle processing.
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Any
import json

from langdetect import detect, detect_langs, LangDetectException
from googletrans import Translator, LANGUAGES

from .utils import setup_logging, validate_language_code

logger = setup_logging()

class LanguageDetector:
    """Handles language detection and translation for subtitles."""

    def __init__(self):
        """Initialize LanguageDetector."""
        self.translator = Translator()
        self.supported_languages = LANGUAGES
        self.language_codes = list(LANGUAGES.keys())

    def detect_language(self, text: str) -> Tuple[str, float]:
        """
        Detect the language of text.

        Args:
            text: Text to analyze

        Returns:
            Tuple of (language_code, confidence)
        """
        try:
            logger.info("Detecting language from text...")

            # Use langdetect for language detection
            detected_lang = detect(text)
            confidence = 0.8  # Default confidence for langdetect

            # Get confidence scores for all detected languages
            try:
                lang_scores = detect_langs(text)
                if lang_scores:
                    # Get the highest confidence score
                    best_match = max(lang_scores, key=lambda x: x.prob)
                    detected_lang = best_match.lang
                    confidence = best_match.prob
            except LangDetectException:
                pass

            logger.info(f"Detected language: {detected_lang} (confidence: {confidence:.3f})")
            return detected_lang, confidence

        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            return 'en', 0.0

    def detect_language_from_segments(self, segments: List[Dict[str, Any]]) -> Tuple[str, float]:
        """
        Detect language from subtitle segments.

        Args:
            segments: List of subtitle segments

        Returns:
            Tuple of (language_code, confidence)
        """
        try:
            logger.info("Detecting language from subtitle segments...")

            # Combine text from all segments
            combined_text = " ".join([seg.get('text', '') for seg in segments])

            if not combined_text.strip():
                logger.warning("No text found in segments for language detection")
                return 'en', 0.0

            return self.detect_language(combined_text)

        except Exception as e:
            logger.error(f"Language detection from segments failed: {e}")
            return 'en', 0.0

    def translate_text(self, text: str, target_language: str,
                      source_language: Optional[str] = None) -> str:
        """
        Translate text to target language.

        Args:
            text: Text to translate
            target_language: Target language code
            source_language: Source language code (auto-detect if None)

        Returns:
            Translated text
        """
        try:
            # Validate target language
            validate_language_code(target_language)

            logger.info(f"Translating text to {target_language}...")

            # Perform translation
            if source_language:
                result = self.translator.translate(
                    text,
                    dest=target_language,
                    src=source_language
                )
            else:
                result = self.translator.translate(text, dest=target_language)

            translated_text = result.text

            logger.info(f"Translation completed: {len(text)} chars -> {len(translated_text)} chars")
            return translated_text

        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return text

    def translate_segments(self, segments: List[Dict[str, Any]],
                          target_language: str,
                          source_language: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Translate subtitle segments to target language.

        Args:
            segments: List of subtitle segments
            target_language: Target language code
            source_language: Source language code (auto-detect if None)

        Returns:
            Translated segments
        """
        try:
            logger.info(f"Translating {len(segments)} segments to {target_language}...")

            translated_segments = []

            for i, segment in enumerate(segments):
                original_text = segment.get('text', '')

                if not original_text.strip():
                    # Skip empty segments
                    translated_segments.append(segment)
                    continue

                try:
                    # Translate text
                    translated_text = self.translate_text(
                        original_text,
                        target_language,
                        source_language
                    )

                    # Create translated segment
                    translated_segment = segment.copy()
                    translated_segment['text'] = translated_text
                    translated_segment['original_text'] = original_text
                    translated_segment['target_language'] = target_language

                    translated_segments.append(translated_segment)

                    # Log progress
                    if (i + 1) % 10 == 0:
                        logger.info(f"Translated {i + 1}/{len(segments)} segments")

                except Exception as e:
                    logger.error(f"Failed to translate segment {i}: {e}")
                    # Keep original segment if translation fails
                    translated_segments.append(segment)

            logger.info(f"Translation completed: {len(translated_segments)} segments")
            return translated_segments

        except Exception as e:
            logger.error(f"Segment translation failed: {e}")
            return segments

    def batch_translate(self, segments: List[Dict[str, Any]],
                       target_languages: List[str],
                       source_language: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Translate segments to multiple target languages.

        Args:
            segments: List of subtitle segments
            target_languages: List of target language codes
            source_language: Source language code (auto-detect if None)

        Returns:
            Dictionary mapping language codes to translated segments
        """
        try:
            logger.info(f"Batch translating to {len(target_languages)} languages...")

            results = {}

            for target_lang in target_languages:
                try:
                    translated = self.translate_segments(segments, target_lang, source_language)
                    results[target_lang] = translated
                    logger.info(f"Completed translation to {target_lang}")
                except Exception as e:
                    logger.error(f"Failed to translate to {target_lang}: {e}")
                    results[target_lang] = segments  # Keep original if translation fails

            return results

        except Exception as e:
            logger.error(f"Batch translation failed: {e}")
            return {lang: segments for lang in target_languages}

    def get_language_name(self, language_code: str) -> str:
        """
        Get language name from language code.

        Args:
            language_code: Language code

        Returns:
            Language name
        """
        return self.supported_languages.get(language_code, language_code)

    def get_supported_languages(self) -> Dict[str, str]:
        """
        Get dictionary of supported languages.

        Returns:
            Dictionary mapping language codes to language names
        """
        return self.supported_languages.copy()

    def validate_language(self, language_code: str) -> bool:
        """
        Validate if language code is supported.

        Args:
            language_code: Language code to validate

        Returns:
            True if supported, False otherwise
        """
        return language_code in self.supported_languages

    def get_language_statistics(self, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get language statistics from segments.

        Args:
            segments: List of subtitle segments

        Returns:
            Language statistics
        """
        try:
            # Combine all text
            all_text = " ".join([seg.get('text', '') for seg in segments])

            if not all_text.strip():
                return {
                    'total_segments': len(segments),
                    'total_characters': 0,
                    'total_words': 0,
                    'detected_language': 'unknown',
                    'confidence': 0.0
                }

            # Detect language
            detected_lang, confidence = self.detect_language(all_text)

            # Count characters and words
            total_chars = len(all_text)
            total_words = len(all_text.split())

            return {
                'total_segments': len(segments),
                'total_characters': total_chars,
                'total_words': total_words,
                'detected_language': detected_lang,
                'language_name': self.get_language_name(detected_lang),
                'confidence': confidence,
                'average_chars_per_segment': total_chars / len(segments) if segments else 0,
                'average_words_per_segment': total_words / len(segments) if segments else 0
            }

        except Exception as e:
            logger.error(f"Failed to get language statistics: {e}")
            return {}

    def create_multilingual_subtitles(self, segments: List[Dict[str, Any]],
                                     target_languages: List[str],
                                     output_dir: str,
                                     base_filename: str,
                                     format_type: str = 'srt') -> Dict[str, str]:
        """
        Create subtitle files in multiple languages.

        Args:
            segments: List of subtitle segments
            target_languages: List of target language codes
            output_dir: Output directory
            base_filename: Base filename for subtitle files
            format_type: Subtitle format

        Returns:
            Dictionary mapping language codes to file paths
        """
        try:
            logger.info(f"Creating multilingual subtitles for {len(target_languages)} languages...")

            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)

            # Translate to all target languages
            translated_results = self.batch_translate(segments, target_languages)

            # Generate subtitle files
            from .subtitle_generator import SubtitleGenerator
            generator = SubtitleGenerator()

            file_paths = {}

            for lang_code, translated_segments in translated_results.items():
                # Create filename
                lang_name = self.get_language_name(lang_code)
                filename = f"{base_filename}_{lang_code}.{format_type}"
                output_path = os.path.join(output_dir, filename)

                # Generate subtitle file
                try:
                    generator.generate_subtitles(
                        translated_segments,
                        output_path,
                        format_type,
                        lang_code
                    )
                    file_paths[lang_code] = output_path
                    logger.info(f"Created subtitle file: {output_path}")
                except Exception as e:
                    logger.error(f"Failed to create subtitle file for {lang_code}: {e}")

            return file_paths

        except Exception as e:
            logger.error(f"Failed to create multilingual subtitles: {e}")
            return {}

    def detect_language_changes(self, segments: List[Dict[str, Any]],
                               window_size: int = 5) -> List[Dict[str, Any]]:
        """
        Detect language changes within subtitle segments.

        Args:
            segments: List of subtitle segments
            window_size: Size of sliding window for language detection

        Returns:
            List of language change points
        """
        try:
            logger.info("Detecting language changes...")

            language_changes = []

            for i in range(len(segments) - window_size + 1):
                # Get window of segments
                window_segments = segments[i:i + window_size]
                window_text = " ".join([seg.get('text', '') for seg in window_segments])

                if not window_text.strip():
                    continue

                # Detect language for this window
                detected_lang, confidence = self.detect_language(window_text)

                # Check if this is a language change
                if i > 0:
                    prev_window = segments[i-1:i-1+window_size]
                    prev_text = " ".join([seg.get('text', '') for seg in prev_window])

                    if prev_text.strip():
                        prev_lang, _ = self.detect_language(prev_text)

                        if detected_lang != prev_lang and confidence > 0.7:
                            change_point = {
                                'segment_index': i,
                                'timestamp': segments[i].get('start', 0),
                                'from_language': prev_lang,
                                'to_language': detected_lang,
                                'confidence': confidence
                            }
                            language_changes.append(change_point)

            logger.info(f"Detected {len(language_changes)} language changes")
            return language_changes

        except Exception as e:
            logger.error(f"Language change detection failed: {e}")
            return []

    def get_translation_quality_metrics(self, original_segments: List[Dict[str, Any]],
                                       translated_segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate translation quality metrics.

        Args:
            original_segments: Original subtitle segments
            translated_segments: Translated subtitle segments

        Returns:
            Translation quality metrics
        """
        try:
            if len(original_segments) != len(translated_segments):
                logger.warning("Segment count mismatch for quality metrics")
                return {}

            total_original_chars = sum(len(seg.get('text', '')) for seg in original_segments)
            total_translated_chars = sum(len(seg.get('text', '')) for seg in translated_segments)

            # Calculate character ratio
            char_ratio = total_translated_chars / total_original_chars if total_original_chars > 0 else 0

            # Count empty translations
            empty_translations = sum(1 for seg in translated_segments if not seg.get('text', '').strip())

            metrics = {
                'total_segments': len(original_segments),
                'total_original_characters': total_original_chars,
                'total_translated_characters': total_translated_chars,
                'character_ratio': char_ratio,
                'empty_translations': empty_translations,
                'translation_coverage': (len(original_segments) - empty_translations) / len(original_segments) if original_segments else 0
            }

            return metrics

        except Exception as e:
            logger.error(f"Failed to calculate translation quality metrics: {e}")
            return {}
