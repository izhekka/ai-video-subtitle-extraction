"""
Speech recognition module using OpenAI Whisper for high-accuracy transcription.
"""

import os
import logging
import tempfile
from typing import Dict, List, Optional, Tuple, Any
import whisper
import torch
import numpy as np

from .utils import setup_logging, get_config

logger = setup_logging()

class SpeechRecognizer:
    """Handles speech-to-text conversion using OpenAI Whisper."""

    def __init__(self, model_size: str = "base", device: str = "auto"):
        """
        Initialize SpeechRecognizer with Whisper model.

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
            device: Device to use (cpu, cuda, auto)
        """
        self.model_size = model_size
        self.device = self._get_device(device)
        self.model = None
        self.config = get_config()

        # Override config with parameters
        if model_size != self.config['whisper_model_size']:
            self.config['whisper_model_size'] = model_size
        if device != "auto":
            self.config['whisper_device'] = device

        self._load_model()

    def _get_device(self, device: str) -> str:
        """Determine the best device to use."""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                logger.info("CUDA available, using GPU")
            else:
                device = "cpu"
                logger.info("CUDA not available, using CPU")
        return device

    def _load_model(self) -> None:
        """Load Whisper model."""
        try:
            logger.info(f"Loading Whisper model: {self.model_size}")
            self.model = whisper.load_model(self.model_size, device=self.device)
            logger.info(f"Whisper model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise

    def transcribe(self, audio_path: str, language: Optional[str] = None,
                   task: str = "transcribe",
                   word_timestamps: bool = True,
                   **kwargs) -> Dict[str, Any]:
        """
        Transcribe audio file using Whisper.

        Args:
            audio_path: Path to audio file
            language: Language code (None for auto-detection)
            task: Task type (transcribe or translate)
            word_timestamps: Whether to include word-level timestamps
            **kwargs: Additional Whisper parameters

        Returns:
            Transcription result dictionary
        """
        if not self.model:
            raise ValueError("Whisper model not loaded")

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            logger.info(f"Transcribing audio: {audio_path}")
            logger.info(f"Language: {language or 'auto-detect'}, Task: {task}")

            # Prepare transcription options
            options = {
                "task": task,
                "word_timestamps": word_timestamps,
                "verbose": False,
                **kwargs
            }

            # Add language if specified
            if language:
                options["language"] = language

            # Perform transcription
            result = self.model.transcribe(audio_path, **options)

            logger.info(f"Transcription completed. Segments: {len(result.get('segments', []))}")
            return result

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise

    def transcribe_with_timestamps(self, audio_path: str,
                                  language: Optional[str] = None,
                                  segment_duration: float = 30.0) -> List[Dict[str, Any]]:
        """
        Transcribe audio with precise timestamp segmentation.

        Args:
            audio_path: Path to audio file
            language: Language code (None for auto-detection)
            segment_duration: Duration of each segment in seconds

        Returns:
            List of transcription segments with timestamps
        """
        try:
            logger.info(f"Transcribing with timestamps: {audio_path}")

            # Transcribe with word timestamps
            result = self.transcribe(
                audio_path,
                language=language,
                word_timestamps=True
            )

            # Process segments
            segments = result.get('segments', [])
            processed_segments = []

            for segment in segments:
                processed_segment = {
                    'start': segment['start'],
                    'end': segment['end'],
                    'text': segment['text'].strip(),
                    'words': segment.get('words', []),
                    'confidence': segment.get('avg_logprob', 0.0)
                }
                processed_segments.append(processed_segment)

            logger.info(f"Processed {len(processed_segments)} segments")
            return processed_segments

        except Exception as e:
            logger.error(f"Timestamp transcription failed: {e}")
            raise

    def detect_language(self, audio_path: str) -> Tuple[str, float]:
        """
        Detect the language of the audio.

        Args:
            audio_path: Path to audio file

        Returns:
            Tuple of (language_code, confidence)
        """
        if not self.model:
            raise ValueError("Whisper model not loaded")

        try:
            logger.info(f"Detecting language: {audio_path}")

            # Load audio and detect language
            audio = whisper.load_audio(audio_path)
            audio = whisper.pad_or_trim(audio)

            # Log mel spectrogram and move to the same device as the model
            mel = whisper.log_mel_spectrogram(audio).to(self.model.device)

            # Detect language
            _, probs = self.model.detect_language(mel)
            detected_lang = max(probs, key=probs.get)
            confidence = float(probs[detected_lang])

            logger.info(f"Detected language: {detected_lang} (confidence: {confidence:.3f})")
            return detected_lang, confidence

        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            raise

    def transcribe_segments(self, audio_path: str,
                           segment_duration: float = 30.0,
                           language: Optional[str] = None,
                           overlap: float = 2.0) -> List[Dict[str, Any]]:
        """
        Transcribe audio in overlapping segments for better accuracy.

        Args:
            audio_path: Path to audio file
            segment_duration: Duration of each segment in seconds
            language: Language code (None for auto-detection)
            overlap: Overlap between segments in seconds

        Returns:
            List of transcription segments
        """
        try:
            logger.info(f"Transcribing in segments: {audio_path}")

            # Load audio
            audio = whisper.load_audio(audio_path)
            duration = len(audio) / whisper.SAMPLE_RATE

            segments = []
            current_time = 0.0

            while current_time < duration:
                # Calculate segment boundaries
                start_time = max(0, current_time - overlap)
                end_time = min(duration, current_time + segment_duration)

                # Extract segment
                start_sample = int(start_time * whisper.SAMPLE_RATE)
                end_sample = int(end_time * whisper.SAMPLE_RATE)
                segment_audio = audio[start_sample:end_sample]

                # Save segment to temporary file
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    temp_path = temp_file.name
                    whisper.save_audio(temp_path, segment_audio, whisper.SAMPLE_RATE)

                try:
                    # Transcribe segment
                    segment_result = self.transcribe(
                        temp_path,
                        language=language,
                        word_timestamps=False
                    )

                    # Adjust timestamps
                    segment_text = segment_result.get('text', '').strip()
                    if segment_text:
                        segments.append({
                            'start': start_time,
                            'end': end_time,
                            'text': segment_text,
                            'confidence': segment_result.get('avg_logprob', 0.0)
                        })

                finally:
                    # Clean up temporary file
                    os.unlink(temp_path)

                current_time += segment_duration - overlap

            logger.info(f"Segment transcription completed: {len(segments)} segments")
            return segments

        except Exception as e:
            logger.error(f"Segment transcription failed: {e}")
            raise

    def get_transcription_confidence(self, result: Dict[str, Any]) -> float:
        """
        Calculate overall transcription confidence.

        Args:
            result: Transcription result from Whisper

        Returns:
            Average confidence score
        """
        try:
            segments = result.get('segments', [])
            if not segments:
                return 0.0

            # Calculate average log probability
            confidences = [seg.get('avg_logprob', 0.0) for seg in segments]
            avg_confidence = np.mean(confidences)

            # Convert log probability to confidence (0-1)
            confidence = np.exp(avg_confidence)

            return confidence

        except Exception as e:
            logger.error(f"Failed to calculate confidence: {e}")
            return 0.0

    def post_process_transcription(self, segments: List[Dict[str, Any]],
                                  remove_filler_words: bool = True,
                                  capitalize_sentences: bool = True,
                                  fix_punctuation: bool = True) -> List[Dict[str, Any]]:
        """
        Post-process transcription segments.

        Args:
            segments: List of transcription segments
            remove_filler_words: Whether to remove common filler words
            capitalize_sentences: Whether to capitalize sentence starts
            fix_punctuation: Whether to fix punctuation

        Returns:
            Processed segments
        """
        try:
            logger.info("Post-processing transcription...")

            processed_segments = []

            for segment in segments:
                text = segment['text']

                # Remove filler words
                if remove_filler_words:
                    text = self._remove_filler_words(text)

                # Fix punctuation
                if fix_punctuation:
                    text = self._fix_punctuation(text)

                # Capitalize sentences
                if capitalize_sentences:
                    text = self._capitalize_sentences(text)

                # Update segment
                processed_segment = segment.copy()
                processed_segment['text'] = text
                processed_segments.append(processed_segment)

            logger.info("Post-processing completed")
            return processed_segments

        except Exception as e:
            logger.error(f"Post-processing failed: {e}")
            return segments

    def _remove_filler_words(self, text: str) -> str:
        """Remove common filler words from text."""
        filler_words = {
            'um', 'uh', 'ah', 'er', 'mm', 'hmm', 'like', 'you know', 'i mean',
            'sort of', 'kind of', 'basically', 'actually', 'literally'
        }

        words = text.split()
        filtered_words = [word for word in words if word.lower() not in filler_words]

        return ' '.join(filtered_words)

    def _fix_punctuation(self, text: str) -> str:
        """Fix common punctuation issues."""
        import re

        # Fix spacing around punctuation
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        text = re.sub(r'([.,!?;:])\s*([A-Z])', r'\1 \2', text)

        # Ensure proper sentence endings
        if text and not text[-1] in '.!?':
            text += '.'

        return text

    def _capitalize_sentences(self, text: str) -> str:
        """Capitalize the first letter of each sentence."""
        import re

        # Split into sentences and capitalize
        sentences = re.split(r'([.!?]+)\s+', text)
        capitalized = []

        for i, sentence in enumerate(sentences):
            if sentence and sentence[0].isalpha():
                capitalized.append(sentence.capitalize())
            else:
                capitalized.append(sentence)

        return ''.join(capitalized)

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            'model_size': self.model_size,
            'device': self.device,
            'config': self.config,
            'model_loaded': self.model is not None
        }

    def cleanup(self) -> None:
        """Clean up resources."""
        if self.model:
            del self.model
            self.model = None
            logger.debug("Whisper model unloaded")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
