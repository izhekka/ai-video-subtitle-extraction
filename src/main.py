"""
Main CLI interface for the AI Video Subtitle Extraction Agent.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, List

import click
from tqdm import tqdm

from .video_processor import VideoProcessor
from .audio_processor import AudioProcessor
from .speech_recognition import SpeechRecognizer
from .subtitle_generator import SubtitleGenerator
from .language_detector import LanguageDetector
from .utils import setup_logging, get_config, validate_video_file, get_supported_formats, get_supported_languages

# Setup logging
logger = setup_logging()

@click.group()
@click.version_option(version="1.0.0")
@click.option('--debug', is_flag=True, help='Enable debug logging')
@click.option('--config', type=click.Path(exists=True), help='Configuration file path')
def cli(debug: bool, config: Optional[str]):
    """AI Video Subtitle Extraction Agent - Extract subtitles from videos using AI."""
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")

@cli.command()
@click.argument('video_path', type=click.Path(exists=True))
@click.option('--language', '-l', help='Language code (e.g., en, es, fr)')
@click.option('--auto-detect', is_flag=True, help='Auto-detect language')
@click.option('--output', '-o', type=click.Path(), help='Output subtitle file path')
@click.option('--format', '-f', type=click.Choice(get_supported_formats()),
              default='srt', help='Output subtitle format')
@click.option('--model-size', type=click.Choice(['tiny', 'base', 'small', 'medium', 'large']),
              default='base', help='Whisper model size')
@click.option('--device', type=click.Choice(['cpu', 'cuda', 'auto']),
              default='auto', help='Device to use for processing')
@click.option('--no-audio-processing', is_flag=True, help='Skip audio preprocessing')
@click.option('--no-post-processing', is_flag=True, help='Skip subtitle post-processing')
@click.option('--confidence-threshold', type=float, default=-1.0,
              help='Minimum confidence threshold for segments (Whisper uses log probabilities, typically negative values)')
def extract(video_path: str, language: Optional[str], auto_detect: bool,
           output: Optional[str], format: str, model_size: str, device: str,
           no_audio_processing: bool, no_post_processing: bool, confidence_threshold: float):
    """Extract subtitles from a video file."""

    try:
        logger.info(f"Starting subtitle extraction for: {video_path}")

        # Validate video file
        validate_video_file(video_path)

        # Generate output path if not provided
        if not output:
            video_stem = Path(video_path).stem
            output = f"{video_stem}_subtitles.{format}"

        # Initialize components
        video_processor = VideoProcessor(video_path)
        audio_processor = AudioProcessor()
        speech_recognizer = SpeechRecognizer(model_size=model_size, device=device)
        subtitle_generator = SubtitleGenerator()
        language_detector = LanguageDetector()

        # Extract audio
        logger.info("Extracting audio from video...")
        audio_path = video_processor.extract_audio()

        # Process audio if enabled
        if not no_audio_processing:
            logger.info("Processing audio for better recognition...")
            processed_audio_path = audio_processor.process_audio(audio_path)
        else:
            processed_audio_path = audio_path

        # Detect language if auto-detect is enabled
        if auto_detect:
            logger.info("Auto-detecting language...")
            detected_lang, confidence = speech_recognizer.detect_language(processed_audio_path)
            language = detected_lang
            logger.info(f"Detected language: {language} (confidence: {confidence:.3f})")

        # Transcribe audio
        logger.info("Transcribing audio...")
        transcription_result = speech_recognizer.transcribe_with_timestamps(
            processed_audio_path,
            language=language
        )

        # Post-process transcription if enabled
        if not no_post_processing:
            logger.info("Post-processing transcription...")
            transcription_result = speech_recognizer.post_process_transcription(
                transcription_result
            )

        # Filter segments by confidence
        if confidence_threshold > 0:
            filtered_segments = [
                seg for seg in transcription_result
                if seg.get('confidence', 0) >= confidence_threshold
            ]
            logger.info(f"Filtered {len(transcription_result)} -> {len(filtered_segments)} segments")
            transcription_result = filtered_segments

        # Generate subtitles
        logger.info(f"Generating {format.upper()} subtitles...")
        subtitle_path = subtitle_generator.generate_subtitles(
            transcription_result,
            output,
            format,
            language or 'en'
        )

        # Get statistics
        stats = language_detector.get_language_statistics(transcription_result)

        # Display results
        logger.info("=" * 50)
        logger.info("EXTRACTION COMPLETED")
        logger.info("=" * 50)
        logger.info(f"Video: {video_path}")
        logger.info(f"Subtitles: {subtitle_path}")
        logger.info(f"Format: {format.upper()}")
        logger.info(f"Language: {language or 'auto-detected'}")
        logger.info(f"Segments: {len(transcription_result)}")
        logger.info(f"Total duration: {stats.get('total_words', 0)} words")
        logger.info(f"Confidence threshold: {confidence_threshold}")

        # Cleanup
        video_processor.cleanup()
        speech_recognizer.cleanup()

        # Remove temporary files
        if os.path.exists(audio_path):
            os.remove(audio_path)
        if processed_audio_path != audio_path and os.path.exists(processed_audio_path):
            os.remove(processed_audio_path)

        logger.info("Extraction completed successfully!")

    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        sys.exit(1)

@cli.command()
@click.argument('input_dir', type=click.Path(exists=True, file_okay=False))
@click.option('--output-dir', '-o', type=click.Path(), help='Output directory for subtitles')
@click.option('--language', '-l', help='Language code for all videos')
@click.option('--format', '-f', type=click.Choice(get_supported_formats()),
              default='srt', help='Output subtitle format')
@click.option('--model-size', type=click.Choice(['tiny', 'base', 'small', 'medium', 'large']),
              default='base', help='Whisper model size')
@click.option('--pattern', default='*.mp4', help='File pattern to process')
@click.option('--max-workers', type=int, default=1, help='Maximum number of parallel workers')
def batch(input_dir: str, output_dir: Optional[str], language: Optional[str],
          format: str, model_size: str, pattern: str, max_workers: int):
    """Process multiple video files in batch."""

    try:
        logger.info(f"Starting batch processing: {input_dir}")

        # Find video files
        input_path = Path(input_dir)
        video_files = list(input_path.glob(pattern))

        if not video_files:
            logger.error(f"No video files found matching pattern: {pattern}")
            sys.exit(1)

        logger.info(f"Found {len(video_files)} video files to process")

        # Create output directory
        if not output_dir:
            output_dir = input_path / "subtitles"
        os.makedirs(output_dir, exist_ok=True)

        # Process each video
        successful = 0
        failed = 0

        for video_file in tqdm(video_files, desc="Processing videos"):
            try:
                logger.info(f"Processing: {video_file.name}")

                # Generate output path
                output_file = Path(output_dir) / f"{video_file.stem}_subtitles.{format}"

                # Extract subtitles
                video_processor = VideoProcessor(str(video_file))
                audio_processor = AudioProcessor()
                speech_recognizer = SpeechRecognizer(model_size=model_size)
                subtitle_generator = SubtitleGenerator()

                # Extract and process audio
                audio_path = video_processor.extract_audio()
                processed_audio_path = audio_processor.process_audio(audio_path)

                # Transcribe
                transcription_result = speech_recognizer.transcribe_with_timestamps(
                    processed_audio_path,
                    language=language
                )

                # Post-process
                transcription_result = speech_recognizer.post_process_transcription(
                    transcription_result
                )

                # Generate subtitles
                subtitle_generator.generate_subtitles(
                    transcription_result,
                    str(output_file),
                    format,
                    language or 'en'
                )

                # Cleanup
                video_processor.cleanup()
                speech_recognizer.cleanup()

                if os.path.exists(audio_path):
                    os.remove(audio_path)
                if processed_audio_path != audio_path and os.path.exists(processed_audio_path):
                    os.remove(processed_audio_path)

                successful += 1
                logger.info(f"Completed: {video_file.name}")

            except Exception as e:
                failed += 1
                logger.error(f"Failed to process {video_file.name}: {e}")

        # Summary
        logger.info("=" * 50)
        logger.info("BATCH PROCESSING COMPLETED")
        logger.info("=" * 50)
        logger.info(f"Total files: {len(video_files)}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Output directory: {output_dir}")

    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        sys.exit(1)

@cli.command()
@click.argument('subtitle_file', type=click.Path(exists=True))
@click.option('--target-language', '-t', required=True, help='Target language code')
@click.option('--output', '-o', type=click.Path(), help='Output subtitle file path')
@click.option('--format', '-f', type=click.Choice(get_supported_formats()),
              default='srt', help='Output subtitle format')
def translate(subtitle_file: str, target_language: str, output: Optional[str], format: str):
    """Translate existing subtitle file to another language."""

    try:
        logger.info(f"Translating subtitle file: {subtitle_file}")

        # Generate output path
        if not output:
            subtitle_stem = Path(subtitle_file).stem
            output = f"{subtitle_stem}_{target_language}.{format}"

        # Load subtitles
        subtitle_generator = SubtitleGenerator()
        segments = subtitle_generator.load_subtitles(subtitle_file)

        # Translate
        language_detector = LanguageDetector()
        translated_segments = language_detector.translate_segments(
            segments,
            target_language
        )

        # Generate translated subtitles
        subtitle_generator.generate_subtitles(
            translated_segments,
            output,
            format,
            target_language
        )

        logger.info(f"Translation completed: {output}")

    except Exception as e:
        logger.error(f"Translation failed: {e}")
        sys.exit(1)

@cli.command()
@click.argument('subtitle_file', type=click.Path(exists=True))
def validate(subtitle_file: str):
    """Validate subtitle file for common issues."""

    try:
        logger.info(f"Validating subtitle file: {subtitle_file}")

        # Load and validate subtitles
        subtitle_generator = SubtitleGenerator()
        segments = subtitle_generator.load_subtitles(subtitle_file)

        validation_result = subtitle_generator.validate_subtitles(segments)

        # Display results
        logger.info("=" * 50)
        logger.info("VALIDATION RESULTS")
        logger.info("=" * 50)
        logger.info(f"File: {subtitle_file}")
        logger.info(f"Valid: {validation_result['valid']}")
        logger.info(f"Total segments: {validation_result['total_segments']}")

        if validation_result['issues']:
            logger.error("ISSUES FOUND:")
            for issue in validation_result['issues']:
                logger.error(f"  - {issue}")

        if validation_result['warnings']:
            logger.warning("WARNINGS:")
            for warning in validation_result['warnings']:
                logger.warning(f"  - {warning}")

        if validation_result['valid']:
            logger.info("✓ Subtitle file is valid!")
        else:
            logger.error("✗ Subtitle file has issues!")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        sys.exit(1)

@cli.command()
def languages():
    """List supported languages."""

    try:
        language_detector = LanguageDetector()
        supported_languages = language_detector.get_supported_languages()

        logger.info("=" * 50)
        logger.info("SUPPORTED LANGUAGES")
        logger.info("=" * 50)

        for code, name in sorted(supported_languages.items()):
            logger.info(f"{code:5} - {name}")

        logger.info(f"\nTotal: {len(supported_languages)} languages")

    except Exception as e:
        logger.error(f"Failed to list languages: {e}")
        sys.exit(1)

@cli.command()
def formats():
    """List supported subtitle formats."""

    try:
        supported_formats = get_supported_formats()

        logger.info("=" * 50)
        logger.info("SUPPORTED SUBTITLE FORMATS")
        logger.info("=" * 50)

        for format_type in supported_formats:
            logger.info(f"  - {format_type.upper()}")

    except Exception as e:
        logger.error(f"Failed to list formats: {e}")
        sys.exit(1)

@cli.command()
def info():
    """Show system information and configuration."""

    try:
        config = get_config()

        logger.info("=" * 50)
        logger.info("SYSTEM INFORMATION")
        logger.info("=" * 50)

        logger.info("Configuration:")
        for key, value in config.items():
            logger.info(f"  {key}: {value}")

        logger.info("\nSupported formats:")
        for format_type in get_supported_formats():
            logger.info(f"  - {format_type.upper()}")

        logger.info(f"\nSupported languages: {len(get_supported_languages())}")

    except Exception as e:
        logger.error(f"Failed to show info: {e}")
        sys.exit(1)

if __name__ == '__main__':
    cli()
