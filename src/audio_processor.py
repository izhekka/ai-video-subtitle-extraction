"""
Audio processing module for preprocessing audio before speech recognition.
"""

import os
import logging
from typing import Optional, Tuple, List
import numpy as np
import librosa
import librosa.display
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt

from .utils import setup_logging

logger = setup_logging()

class AudioProcessor:
    """Handles audio preprocessing and optimization for speech recognition."""

    def __init__(self, sample_rate: int = 16000):
        """
        Initialize AudioProcessor.

        Args:
            sample_rate: Target sample rate for audio processing
        """
        self.sample_rate = sample_rate
        self.audio_data = None
        self.original_sample_rate = None

    def load_audio(self, audio_path: str) -> np.ndarray:
        """
        Load audio file and resample if necessary.

        Args:
            audio_path: Path to audio file

        Returns:
            Audio data as numpy array
        """
        try:
            logger.info(f"Loading audio: {audio_path}")

            # Load audio with librosa
            audio_data, original_sr = librosa.load(audio_path, sr=None)
            self.original_sample_rate = original_sr

            # Resample if necessary
            if original_sr != self.sample_rate:
                logger.info(f"Resampling from {original_sr}Hz to {self.sample_rate}Hz")
                audio_data = librosa.resample(audio_data, orig_sr=original_sr, target_sr=self.sample_rate)

            self.audio_data = audio_data
            logger.info(f"Audio loaded: {len(audio_data)} samples, {len(audio_data)/self.sample_rate:.2f}s")

            return audio_data

        except Exception as e:
            logger.error(f"Failed to load audio: {e}")
            raise

    def normalize_audio(self, audio_data: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Normalize audio to prevent clipping and improve quality.

        Args:
            audio_data: Audio data to normalize (uses self.audio_data if None)

        Returns:
            Normalized audio data
        """
        if audio_data is None:
            audio_data = self.audio_data

        if audio_data is None:
            raise ValueError("No audio data available")

        try:
            # Normalize to [-1, 1] range
            normalized = librosa.util.normalize(audio_data)
            logger.debug("Audio normalized")
            return normalized

        except Exception as e:
            logger.error(f"Failed to normalize audio: {e}")
            raise

    def remove_noise(self, audio_data: Optional[np.ndarray] = None,
                    noise_reduction_strength: float = 0.1) -> np.ndarray:
        """
        Apply noise reduction to audio.

        Args:
            audio_data: Audio data to process (uses self.audio_data if None)
            noise_reduction_strength: Strength of noise reduction (0.0 to 1.0)

        Returns:
            Noise-reduced audio data
        """
        if audio_data is None:
            audio_data = self.audio_data

        if audio_data is None:
            raise ValueError("No audio data available")

        try:
            logger.info("Applying noise reduction...")

            # Apply spectral gating (simple noise reduction)
            # Calculate noise profile from first 1 second
            noise_sample = audio_data[:self.sample_rate]
            noise_profile = np.mean(np.abs(noise_sample))

            # Apply threshold-based noise reduction
            threshold = noise_profile * noise_reduction_strength
            cleaned_audio = np.where(np.abs(audio_data) < threshold, 0, audio_data)

            logger.debug(f"Noise reduction applied with threshold: {threshold:.6f}")
            return cleaned_audio

        except Exception as e:
            logger.error(f"Failed to remove noise: {e}")
            return audio_data

    def apply_high_pass_filter(self, audio_data: Optional[np.ndarray] = None,
                              cutoff_freq: float = 80.0) -> np.ndarray:
        """
        Apply high-pass filter to remove low-frequency noise.

        Args:
            audio_data: Audio data to filter (uses self.audio_data if None)
            cutoff_freq: Cutoff frequency in Hz

        Returns:
            Filtered audio data
        """
        if audio_data is None:
            audio_data = self.audio_data

        if audio_data is None:
            raise ValueError("No audio data available")

        try:
            logger.info(f"Applying high-pass filter (cutoff: {cutoff_freq}Hz)...")

            # Design high-pass filter
            nyquist = self.sample_rate / 2
            normalized_cutoff = cutoff_freq / nyquist
            b, a = signal.butter(4, normalized_cutoff, btype='high')

            # Apply filter
            filtered_audio = signal.filtfilt(b, a, audio_data)

            logger.debug("High-pass filter applied")
            return filtered_audio

        except Exception as e:
            logger.error(f"Failed to apply high-pass filter: {e}")
            return audio_data

    def enhance_speech(self, audio_data: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply speech enhancement techniques.

        Args:
            audio_data: Audio data to enhance (uses self.audio_data if None)

        Returns:
            Enhanced audio data
        """
        if audio_data is None:
            audio_data = self.audio_data

        if audio_data is None:
            raise ValueError("No audio data available")

        try:
            logger.info("Applying speech enhancement...")

            # Apply pre-emphasis filter to boost high frequencies
            pre_emphasis = 0.97
            emphasized_audio = np.append(audio_data[0], audio_data[1:] - pre_emphasis * audio_data[:-1])

            # Apply spectral subtraction for noise reduction
            # This is a simplified version - more advanced methods can be used
            stft = librosa.stft(emphasized_audio)
            magnitude = np.abs(stft)

            # Estimate noise from first few frames
            noise_estimate = np.mean(magnitude[:, :10], axis=1, keepdims=True)

            # Apply spectral subtraction
            enhanced_magnitude = np.maximum(magnitude - 0.5 * noise_estimate, 0.1 * magnitude)

            # Reconstruct signal
            enhanced_stft = enhanced_magnitude * np.exp(1j * np.angle(stft))
            enhanced_audio = librosa.istft(enhanced_stft)

            logger.debug("Speech enhancement applied")
            return enhanced_audio

        except Exception as e:
            logger.error(f"Failed to enhance speech: {e}")
            return audio_data

    def detect_speech_segments(self, audio_data: Optional[np.ndarray] = None,
                              min_silence_duration: float = 0.5,
                              speech_threshold: float = 0.01) -> List[Tuple[float, float]]:
        """
        Detect speech segments in audio.

        Args:
            audio_data: Audio data to analyze (uses self.audio_data if None)
            min_silence_duration: Minimum silence duration to split segments
            speech_threshold: Threshold for speech detection

        Returns:
            List of (start_time, end_time) tuples for speech segments
        """
        if audio_data is None:
            audio_data = self.audio_data

        if audio_data is None:
            raise ValueError("No audio data available")

        try:
            logger.info("Detecting speech segments...")

            # Calculate energy envelope
            frame_length = int(0.025 * self.sample_rate)  # 25ms frames
            hop_length = int(0.010 * self.sample_rate)    # 10ms hop

            # Calculate RMS energy
            rms = librosa.feature.rms(y=audio_data, frame_length=frame_length, hop_length=hop_length)[0]

            # Apply threshold to detect speech
            speech_mask = rms > speech_threshold

            # Find speech segments
            segments = []
            in_speech = False
            start_time = 0

            for i, is_speech in enumerate(speech_mask):
                time = i * hop_length / self.sample_rate

                if is_speech and not in_speech:
                    # Start of speech segment
                    start_time = time
                    in_speech = True
                elif not is_speech and in_speech:
                    # End of speech segment
                    if time - start_time >= min_silence_duration:
                        segments.append((start_time, time))
                    in_speech = False

            # Handle case where speech continues to end
            if in_speech:
                segments.append((start_time, len(audio_data) / self.sample_rate))

            logger.info(f"Detected {len(segments)} speech segments")
            return segments

        except Exception as e:
            logger.error(f"Failed to detect speech segments: {e}")
            return []

    def get_audio_features(self, audio_data: Optional[np.ndarray] = None) -> dict:
        """
        Extract audio features for analysis.

        Args:
            audio_data: Audio data to analyze (uses self.audio_data if None)

        Returns:
            Dictionary of audio features
        """
        if audio_data is None:
            audio_data = self.audio_data

        if audio_data is None:
            raise ValueError("No audio data available")

        try:
            features = {}

            # Basic statistics
            features['duration'] = len(audio_data) / self.sample_rate
            features['rms_energy'] = np.sqrt(np.mean(audio_data**2))
            features['peak_amplitude'] = np.max(np.abs(audio_data))

            # Spectral features
            stft = librosa.stft(audio_data)
            magnitude = np.abs(stft)

            features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(S=magnitude))
            features['spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(S=magnitude))
            features['spectral_rolloff'] = np.mean(librosa.feature.spectral_rolloff(S=magnitude))

            # MFCC features
            mfcc = librosa.feature.mfcc(y=audio_data, sr=self.sample_rate, n_mfcc=13)
            features['mfcc_mean'] = np.mean(mfcc, axis=1).tolist()
            features['mfcc_std'] = np.std(mfcc, axis=1).tolist()

            # Zero crossing rate
            features['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(audio_data))

            logger.debug(f"Extracted {len(features)} audio features")
            return features

        except Exception as e:
            logger.error(f"Failed to extract audio features: {e}")
            return {}

    def save_audio(self, audio_data: np.ndarray, output_path: str) -> None:
        """
        Save processed audio to file.

        Args:
            audio_data: Audio data to save
            output_path: Output file path
        """
        try:
            logger.info(f"Saving audio to: {output_path}")

            # Ensure output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir:  # Only create directory if path is not empty
                os.makedirs(output_dir, exist_ok=True)

            # Save as WAV file
            wavfile.write(output_path, self.sample_rate, audio_data.astype(np.float32))

            logger.info(f"Audio saved successfully: {output_path}")

        except Exception as e:
            logger.error(f"Failed to save audio: {e}")
            raise

    def plot_audio(self, audio_data: Optional[np.ndarray] = None,
                   output_path: Optional[str] = None) -> None:
        """
        Create audio visualization plots.

        Args:
            audio_data: Audio data to plot (uses self.audio_data if None)
            output_path: Path to save plot (optional)
        """
        if audio_data is None:
            audio_data = self.audio_data

        if audio_data is None:
            raise ValueError("No audio data available")

        try:
            fig, axes = plt.subplots(3, 1, figsize=(12, 8))

            # Waveform
            time = np.linspace(0, len(audio_data) / self.sample_rate, len(audio_data))
            axes[0].plot(time, audio_data)
            axes[0].set_title('Waveform')
            axes[0].set_xlabel('Time (s)')
            axes[0].set_ylabel('Amplitude')

            # Spectrogram
            D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)
            librosa.display.specshow(D, sr=self.sample_rate, x_axis='time', y_axis='log', ax=axes[1])
            axes[1].set_title('Spectrogram')

            # Mel spectrogram
            mel_spect = librosa.feature.melspectrogram(y=audio_data, sr=self.sample_rate)
            mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)
            librosa.display.specshow(mel_spect_db, sr=self.sample_rate, x_axis='time', y_axis='mel', ax=axes[2])
            axes[2].set_title('Mel Spectrogram')

            plt.tight_layout()

            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"Audio plot saved: {output_path}")
            else:
                plt.show()

        except Exception as e:
            logger.error(f"Failed to create audio plot: {e}")

    def process_audio(self, audio_path: str, output_path: Optional[str] = None,
                     apply_noise_reduction: bool = True,
                     apply_enhancement: bool = True,
                     apply_filter: bool = True) -> str:
        """
        Complete audio processing pipeline.

        Args:
            audio_path: Input audio file path
            output_path: Output audio file path (optional)
            apply_noise_reduction: Whether to apply noise reduction
            apply_enhancement: Whether to apply speech enhancement
            apply_filter: Whether to apply high-pass filter

        Returns:
            Path to processed audio file
        """
        try:
            logger.info("Starting audio processing pipeline...")

            # Load audio
            audio_data = self.load_audio(audio_path)

            # Normalize
            audio_data = self.normalize_audio(audio_data)

            # Apply processing steps
            if apply_filter:
                audio_data = self.apply_high_pass_filter(audio_data)

            if apply_noise_reduction:
                audio_data = self.remove_noise(audio_data)

            if apply_enhancement:
                audio_data = self.enhance_speech(audio_data)

            # Final normalization
            audio_data = self.normalize_audio(audio_data)

            # Save processed audio
            if output_path is None:
                base_path = os.path.splitext(audio_path)[0]
                output_path = f"{base_path}_processed.wav"

            self.save_audio(audio_data, output_path)

            logger.info("Audio processing pipeline completed")
            return output_path

        except Exception as e:
            logger.error(f"Audio processing pipeline failed: {e}")
            raise
