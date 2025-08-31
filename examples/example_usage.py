#!/usr/bin/env python3
"""
Example usage of the AI Video Subtitle Extraction Agent.

This script demonstrates how to use the agent programmatically
to extract subtitles from video files.
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.video_processor import VideoProcessor
from src.audio_processor import AudioProcessor
from src.speech_recognition import SpeechRecognizer
from src.subtitle_generator import SubtitleGenerator
from src.language_detector import LanguageDetector
from src.utils import setup_logging

def main():
    """Main example function."""
    logger = setup_logging()

    # Example video file path (replace with your video file)
    video_path = "sample_video.mp4"  # Change this to your video file

    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        logger.info("Please place a video file in the examples directory and update the path")
        return

    try:
        logger.info("=" * 60)
        logger.info("AI VIDEO SUBTITLE EXTRACTION EXAMPLE")
        logger.info("=" * 60)

        # Step 1: Initialize components
        logger.info("1. Initializing components...")
        video_processor = VideoProcessor(video_path)
        audio_processor = AudioProcessor()
        speech_recognizer = SpeechRecognizer(model_size="base")  # Use base model for speed
        subtitle_generator = SubtitleGenerator()
        language_detector = LanguageDetector()

        # Step 2: Extract audio from video
        logger.info("2. Extracting audio from video...")
        audio_path = video_processor.extract_audio()
        logger.info(f"Audio extracted to: {audio_path}")

        # Step 3: Process audio for better recognition
        logger.info("3. Processing audio...")
        processed_audio_path = audio_processor.process_audio(audio_path)
        logger.info(f"Audio processed: {processed_audio_path}")

        # Step 4: Detect language (optional)
        logger.info("4. Detecting language...")
        detected_lang, confidence = speech_recognizer.detect_language(processed_audio_path)
        logger.info(f"Detected language: {detected_lang} (confidence: {confidence:.3f})")

        # Step 5: Transcribe audio
        logger.info("5. Transcribing audio...")
        transcription_result = speech_recognizer.transcribe_with_timestamps(
            processed_audio_path,
            language=detected_lang
        )
        logger.info(f"Transcription completed: {len(transcription_result)} segments")

        # Step 6: Post-process transcription
        logger.info("6. Post-processing transcription...")
        processed_segments = speech_recognizer.post_process_transcription(
            transcription_result
        )

        # Step 7: Generate subtitles in multiple formats
        logger.info("7. Generating subtitle files...")

        # SRT format
        srt_path = f"{Path(video_path).stem}_subtitles.srt"
        subtitle_generator.generate_subtitles(
            processed_segments,
            srt_path,
            'srt',
            detected_lang
        )
        logger.info(f"SRT subtitles: {srt_path}")

        # VTT format
        vtt_path = f"{Path(video_path).stem}_subtitles.vtt"
        subtitle_generator.generate_subtitles(
            processed_segments,
            vtt_path,
            'vtt',
            detected_lang
        )
        logger.info(f"VTT subtitles: {vtt_path}")

        # TXT format
        txt_path = f"{Path(video_path).stem}_subtitles.txt"
        subtitle_generator.generate_subtitles(
            processed_segments,
            txt_path,
            'txt',
            detected_lang
        )
        logger.info(f"TXT subtitles: {txt_path}")

        # Step 8: Get statistics
        logger.info("8. Generating statistics...")
        stats = language_detector.get_language_statistics(processed_segments)

        # Step 9: Validate subtitles
        logger.info("9. Validating subtitles...")
        validation = subtitle_generator.validate_subtitles(processed_segments)

        # Step 10: Display results
        logger.info("=" * 60)
        logger.info("EXTRACTION RESULTS")
        logger.info("=" * 60)
        logger.info(f"Video file: {video_path}")
        logger.info(f"Detected language: {detected_lang} ({language_detector.get_language_name(detected_lang)})")
        logger.info(f"Language confidence: {confidence:.3f}")
        logger.info(f"Total segments: {len(processed_segments)}")
        logger.info(f"Total words: {stats.get('total_words', 0)}")
        logger.info(f"Total characters: {stats.get('total_characters', 0)}")
        logger.info(f"Average words per segment: {stats.get('average_words_per_segment', 0):.1f}")
        logger.info(f"Subtitle validation: {'✓ Valid' if validation['valid'] else '✗ Invalid'}")

        if validation['issues']:
            logger.warning("Issues found:")
            for issue in validation['issues']:
                logger.warning(f"  - {issue}")

        logger.info("\nGenerated subtitle files:")
        logger.info(f"  - {srt_path}")
        logger.info(f"  - {vtt_path}")
        logger.info(f"  - {txt_path}")

        # Step 11: Optional - Translate to another language
        logger.info("\n10. Optional: Translating to Spanish...")
        try:
            translated_segments = language_detector.translate_segments(
                processed_segments,
                'es',  # Spanish
                detected_lang
            )

            spanish_srt_path = f"{Path(video_path).stem}_subtitles_es.srt"
            subtitle_generator.generate_subtitles(
                translated_segments,
                spanish_srt_path,
                'srt',
                'es'
            )
            logger.info(f"Spanish subtitles: {spanish_srt_path}")

        except Exception as e:
            logger.warning(f"Translation failed: {e}")

        # Cleanup
        logger.info("\n11. Cleaning up...")
        video_processor.cleanup()
        speech_recognizer.cleanup()

        # Remove temporary files
        if os.path.exists(audio_path):
            os.remove(audio_path)
        if processed_audio_path != audio_path and os.path.exists(processed_audio_path):
            os.remove(processed_audio_path)

        logger.info("=" * 60)
        logger.info("EXTRACTION COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        import traceback
        traceback.print_exc()

def simple_extraction_example():
    """Simple extraction example with minimal code."""
    logger = setup_logging()

    video_path = "sample_video.mp4"

    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return

    try:
        logger.info("Simple extraction example...")

        # Initialize components
        video_processor = VideoProcessor(video_path)
        audio_processor = AudioProcessor()
        speech_recognizer = SpeechRecognizer(model_size="tiny")  # Fastest model
        subtitle_generator = SubtitleGenerator()

        # Extract and process audio
        audio_path = video_processor.extract_audio()
        processed_audio_path = audio_processor.process_audio(audio_path)

        # Transcribe with auto language detection
        transcription_result = speech_recognizer.transcribe_with_timestamps(
            processed_audio_path
        )

        # Generate SRT subtitles
        output_path = f"{Path(video_path).stem}_simple.srt"
        subtitle_generator.generate_subtitles(
            transcription_result,
            output_path,
            'srt'
        )

        logger.info(f"Simple extraction completed: {output_path}")

        # Cleanup
        video_processor.cleanup()
        speech_recognizer.cleanup()

        if os.path.exists(audio_path):
            os.remove(audio_path)
        if processed_audio_path != audio_path and os.path.exists(processed_audio_path):
            os.remove(processed_audio_path)

    except Exception as e:
        logger.error(f"Simple extraction failed: {e}")

if __name__ == "__main__":
    print("AI Video Subtitle Extraction Agent - Example Usage")
    print("=" * 60)
    print("1. Full extraction example")
    print("2. Simple extraction example")
    print("3. Exit")

    choice = input("\nSelect an option (1-3): ").strip()

    if choice == "1":
        main()
    elif choice == "2":
        simple_extraction_example()
    elif choice == "3":
        print("Goodbye!")
    else:
        print("Invalid choice. Running full example...")
        main()
