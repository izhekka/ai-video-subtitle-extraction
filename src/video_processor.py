"""
Video processing module for extracting audio and metadata from video files.
"""

import os
import logging
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

import cv2
import numpy as np
from moviepy import VideoFileClip, AudioFileClip
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_audio

from .utils import validate_video_file, get_file_info, setup_logging

logger = setup_logging()

class VideoProcessor:
    """Handles video file processing, validation, and audio extraction."""

    def __init__(self, video_path: str):
        """
        Initialize VideoProcessor with video file path.

        Args:
            video_path: Path to the video file
        """
        self.video_path = video_path
        self.video_clip = None
        self.audio_path = None
        self.metadata = {}

        # Validate video file
        validate_video_file(video_path)
        self._load_video()

    def _load_video(self) -> None:
        """Load video file and extract metadata."""
        try:
            logger.info(f"Loading video: {self.video_path}")
            self.video_clip = VideoFileClip(self.video_path)
            self._extract_metadata()
            logger.info(f"Video loaded successfully. Duration: {self.video_clip.duration:.2f}s")
        except Exception as e:
            logger.error(f"Failed to load video: {e}")
            raise

    def _extract_metadata(self) -> None:
        """Extract video metadata."""
        if not self.video_clip:
            return

        # Get file info
        file_info = get_file_info(self.video_path)

        # Get video properties
        self.metadata = {
            'file_info': file_info,
            'duration': self.video_clip.duration,
            'fps': self.video_clip.fps,
            'size': self.video_clip.size,  # (width, height)
            'has_audio': self.video_clip.audio is not None,
            'audio_fps': self.video_clip.audio.fps if self.video_clip.audio else None,
            'audio_nchannels': self.video_clip.audio.nchannels if self.video_clip.audio else None
        }

        logger.debug(f"Video metadata: {self.metadata}")

    def extract_audio(self, output_path: Optional[str] = None,
                     audio_format: str = 'wav',
                     sample_rate: int = 16000) -> str:
        """
        Extract audio from video file.

        Args:
            output_path: Path for output audio file (optional)
            audio_format: Audio format (wav, mp3, etc.)
            sample_rate: Audio sample rate

        Returns:
            Path to extracted audio file
        """
        if not self.video_clip or not self.video_clip.audio:
            raise ValueError("Video has no audio track")

        if output_path is None:
            # Generate output path
            video_stem = Path(self.video_path).stem
            output_path = f"{video_stem}_audio.{audio_format}"

        try:
            logger.info(f"Extracting audio to: {output_path}")

            # Extract audio using MoviePy
            audio_clip = self.video_clip.audio
            audio_clip.write_audiofile(
                output_path,
                fps=sample_rate,
                logger=None
            )

            self.audio_path = output_path
            logger.info(f"Audio extraction completed: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Failed to extract audio: {e}")
            raise

    def get_video_frames(self, start_time: float = 0,
                        end_time: Optional[float] = None,
                        frame_rate: Optional[float] = None) -> np.ndarray:
        """
        Extract video frames for analysis.

        Args:
            start_time: Start time in seconds
            end_time: End time in seconds (None for end of video)
            frame_rate: Frame rate for extraction (None for original)

        Returns:
            Array of video frames
        """
        if not self.video_clip:
            raise ValueError("Video not loaded")

        if end_time is None:
            end_time = self.video_clip.duration

        try:
            # Extract subclip
            subclip = self.video_clip.subclip(start_time, end_time)

            # Set frame rate if specified
            if frame_rate:
                subclip = subclip.set_fps(frame_rate)

            # Convert to numpy array
            frames = []
            for frame in subclip.iter_frames():
                frames.append(frame)

            return np.array(frames)

        except Exception as e:
            logger.error(f"Failed to extract frames: {e}")
            raise

    def get_video_info(self) -> Dict[str, Any]:
        """Get comprehensive video information."""
        return {
            'path': self.video_path,
            'metadata': self.metadata,
            'audio_path': self.audio_path
        }

    def detect_scenes(self, threshold: float = 30.0) -> list:
        """
        Detect scene changes in the video.

        Args:
            threshold: Threshold for scene detection

        Returns:
            List of scene timestamps
        """
        if not self.video_clip:
            raise ValueError("Video not loaded")

        try:
            logger.info("Detecting scenes...")

            # Use OpenCV for scene detection
            cap = cv2.VideoCapture(self.video_path)
            scenes = []
            prev_frame = None
            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if prev_frame is not None:
                    # Calculate frame difference
                    diff = cv2.absdiff(prev_frame, frame)
                    mean_diff = np.mean(diff)

                    if mean_diff > threshold:
                        timestamp = frame_count / self.video_clip.fps
                        scenes.append(timestamp)

                prev_frame = frame.copy()
                frame_count += 1

            cap.release()
            logger.info(f"Detected {len(scenes)} scenes")
            return scenes

        except Exception as e:
            logger.error(f"Failed to detect scenes: {e}")
            return []

    def get_audio_segments(self, segment_duration: float = 30.0) -> list:
        """
        Split audio into segments for processing.

        Args:
            segment_duration: Duration of each segment in seconds

        Returns:
            List of audio segment file paths
        """
        if not self.audio_path:
            raise ValueError("Audio not extracted")

        try:
            logger.info(f"Splitting audio into {segment_duration}s segments...")

            audio_clip = AudioFileClip(self.audio_path)
            segments = []

            for i in range(0, int(audio_clip.duration), int(segment_duration)):
                start_time = i
                end_time = min(i + segment_duration, audio_clip.duration)

                # Create segment
                segment = audio_clip.subclip(start_time, end_time)
                segment_path = f"{Path(self.audio_path).stem}_segment_{i//int(segment_duration)}.wav"

                segment.write_audiofile(segment_path, verbose=False, logger=None)
                segments.append({
                    'path': segment_path,
                    'start_time': start_time,
                    'end_time': end_time
                })

            logger.info(f"Created {len(segments)} audio segments")
            return segments

        except Exception as e:
            logger.error(f"Failed to create audio segments: {e}")
            raise

    def cleanup(self) -> None:
        """Clean up resources."""
        if self.video_clip:
            self.video_clip.close()

        # Remove temporary audio file if it exists
        if self.audio_path and os.path.exists(self.audio_path):
            try:
                os.remove(self.audio_path)
                logger.debug(f"Removed temporary audio file: {self.audio_path}")
            except Exception as e:
                logger.warning(f"Failed to remove temporary audio file: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
