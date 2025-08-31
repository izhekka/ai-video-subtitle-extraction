"""
Subtitle generation module for creating subtitle files in various formats.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

import pysrt
from webvtt import WebVTT, Caption

from .utils import setup_logging, format_time, format_time_vtt, ensure_directory

logger = setup_logging()

class SubtitleGenerator:
    """Handles generation of subtitle files in various formats."""

    def __init__(self):
        """Initialize SubtitleGenerator."""
        self.supported_formats = ['srt', 'vtt', 'txt', 'json']

    def generate_subtitles(self, segments: List[Dict[str, Any]],
                          output_path: str,
                          format_type: str = 'srt',
                          language: str = 'en',
                          encoding: str = 'utf-8') -> str:
        """
        Generate subtitle file from transcription segments.

        Args:
            segments: List of transcription segments
            output_path: Output file path
            format_type: Subtitle format (srt, vtt, txt, json)
            language: Language code
            encoding: File encoding

        Returns:
            Path to generated subtitle file
        """
        if format_type not in self.supported_formats:
            raise ValueError(f"Unsupported format: {format_type}. Supported: {self.supported_formats}")

        try:
            logger.info(f"Generating {format_type.upper()} subtitles: {output_path}")

            # Ensure output directory exists
            ensure_directory(os.path.dirname(output_path))

            if format_type == 'srt':
                return self._generate_srt(segments, output_path, encoding)
            elif format_type == 'vtt':
                return self._generate_vtt(segments, output_path, encoding)
            elif format_type == 'txt':
                return self._generate_txt(segments, output_path, encoding)
            elif format_type == 'json':
                return self._generate_json(segments, output_path, encoding)

        except Exception as e:
            logger.error(f"Failed to generate subtitles: {e}")
            raise

    def _generate_srt(self, segments: List[Dict[str, Any]],
                     output_path: str, encoding: str) -> str:
        """Generate SRT subtitle file."""
        try:
            subs = pysrt.SubRipFile()

            for i, segment in enumerate(segments, 1):
                # Convert seconds to milliseconds for pysrt
                start_ms = int(segment['start'] * 1000)
                end_ms = int(segment['end'] * 1000)
                text = segment['text']

                # Create SubRipTime objects using milliseconds
                start_time = pysrt.SubRipTime(milliseconds=start_ms)
                end_time = pysrt.SubRipTime(milliseconds=end_ms)

                sub = pysrt.SubRipItem(i, start_time, end_time, text)
                subs.append(sub)

            # Save SRT file
            subs.save(output_path, encoding=encoding)
            logger.info(f"SRT subtitles saved: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Failed to generate SRT: {e}")
            raise

    def _generate_vtt(self, segments: List[Dict[str, Any]],
                     output_path: str, encoding: str) -> str:
        """Generate WebVTT subtitle file."""
        try:
            vtt = WebVTT()

            for segment in segments:
                start_time = format_time_vtt(segment['start'])
                end_time = format_time_vtt(segment['end'])
                text = segment['text']

                caption = Caption(start_time, end_time, text)
                vtt.captions.append(caption)

            # Save VTT file
            with open(output_path, 'w', encoding=encoding) as f:
                f.write(vtt.content)

            logger.info(f"VTT subtitles saved: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Failed to generate VTT: {e}")
            raise

    def _generate_txt(self, segments: List[Dict[str, Any]],
                     output_path: str, encoding: str) -> str:
        """Generate plain text subtitle file."""
        try:
            with open(output_path, 'w', encoding=encoding) as f:
                for segment in segments:
                    start_time = format_time(segment['start'])
                    end_time = format_time(segment['end'])
                    text = segment['text']

                    f.write(f"[{start_time} --> {end_time}]\n")
                    f.write(f"{text}\n\n")

            logger.info(f"TXT subtitles saved: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Failed to generate TXT: {e}")
            raise

    def _generate_json(self, segments: List[Dict[str, Any]],
                      output_path: str, encoding: str) -> str:
        """Generate JSON subtitle file."""
        try:
            subtitle_data = {
                'segments': segments,
                'metadata': {
                    'total_segments': len(segments),
                    'total_duration': segments[-1]['end'] if segments else 0,
                    'format': 'json'
                }
            }

            with open(output_path, 'w', encoding=encoding) as f:
                json.dump(subtitle_data, f, indent=2, ensure_ascii=False)

            logger.info(f"JSON subtitles saved: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Failed to generate JSON: {e}")
            raise

    def merge_subtitles(self, subtitle_files: List[str],
                       output_path: str,
                       format_type: str = 'srt') -> str:
        """
        Merge multiple subtitle files into one.

        Args:
            subtitle_files: List of subtitle file paths
            output_path: Output file path
            format_type: Output format

        Returns:
            Path to merged subtitle file
        """
        try:
            logger.info(f"Merging {len(subtitle_files)} subtitle files...")

            all_segments = []

            for file_path in subtitle_files:
                if not os.path.exists(file_path):
                    logger.warning(f"Subtitle file not found: {file_path}")
                    continue

                segments = self.load_subtitles(file_path)
                all_segments.extend(segments)

            # Sort segments by start time
            all_segments.sort(key=lambda x: x['start'])

            # Generate merged subtitles
            return self.generate_subtitles(all_segments, output_path, format_type)

        except Exception as e:
            logger.error(f"Failed to merge subtitles: {e}")
            raise

    def load_subtitles(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load subtitles from file.

        Args:
            file_path: Path to subtitle file

        Returns:
            List of subtitle segments
        """
        try:
            file_ext = Path(file_path).suffix.lower()

            if file_ext == '.srt':
                return self._load_srt(file_path)
            elif file_ext == '.vtt':
                return self._load_vtt(file_path)
            elif file_ext == '.json':
                return self._load_json(file_path)
            else:
                raise ValueError(f"Unsupported subtitle format: {file_ext}")

        except Exception as e:
            logger.error(f"Failed to load subtitles: {e}")
            raise

    def _load_srt(self, file_path: str) -> List[Dict[str, Any]]:
        """Load SRT subtitle file."""
        try:
            subs = pysrt.open(file_path)
            segments = []

            for sub in subs:
                segment = {
                    'start': sub.start.ordinal / 1000.0,  # Convert to seconds
                    'end': sub.end.ordinal / 1000.0,
                    'text': sub.text,
                    'index': sub.index
                }
                segments.append(segment)

            return segments

        except Exception as e:
            logger.error(f"Failed to load SRT: {e}")
            raise

    def _load_vtt(self, file_path: str) -> List[Dict[str, Any]]:
        """Load WebVTT subtitle file."""
        try:
            vtt = WebVTT().read(file_path)
            segments = []

            for i, caption in enumerate(vtt.captions, 1):
                # Parse timestamps
                start_time = self._parse_vtt_time(caption.start)
                end_time = self._parse_vtt_time(caption.end)

                segment = {
                    'start': start_time,
                    'end': end_time,
                    'text': caption.text,
                    'index': i
                }
                segments.append(segment)

            return segments

        except Exception as e:
            logger.error(f"Failed to load VTT: {e}")
            raise

    def _load_json(self, file_path: str) -> List[Dict[str, Any]]:
        """Load JSON subtitle file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            return data.get('segments', [])

        except Exception as e:
            logger.error(f"Failed to load JSON: {e}")
            raise

    def _parse_vtt_time(self, time_str: str) -> float:
        """Parse VTT timestamp to seconds."""
        try:
            # VTT format: HH:MM:SS.mmm
            parts = time_str.split(':')
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])

            return hours * 3600 + minutes * 60 + seconds

        except Exception as e:
            logger.error(f"Failed to parse VTT time: {time_str}")
            return 0.0

    def adjust_timing(self, segments: List[Dict[str, Any]],
                     offset: float) -> List[Dict[str, Any]]:
        """
        Adjust timing of subtitle segments.

        Args:
            segments: List of subtitle segments
            offset: Time offset in seconds (positive for delay, negative for advance)

        Returns:
            Adjusted segments
        """
        try:
            logger.info(f"Adjusting timing by {offset:.3f}s")

            adjusted_segments = []

            for segment in segments:
                adjusted_segment = segment.copy()
                adjusted_segment['start'] = max(0, segment['start'] + offset)
                adjusted_segment['end'] = max(0, segment['end'] + offset)
                adjusted_segments.append(adjusted_segment)

            return adjusted_segments

        except Exception as e:
            logger.error(f"Failed to adjust timing: {e}")
            return segments

    def split_long_subtitles(self, segments: List[Dict[str, Any]],
                            max_duration: float = 5.0,
                            max_chars: int = 100) -> List[Dict[str, Any]]:
        """
        Split long subtitle segments into shorter ones.

        Args:
            segments: List of subtitle segments
            max_duration: Maximum duration for each segment
            max_chars: Maximum characters per segment

        Returns:
            Split segments
        """
        try:
            logger.info("Splitting long subtitle segments...")

            split_segments = []

            for segment in segments:
                duration = segment['end'] - segment['start']
                text = segment['text']

                # Check if splitting is needed
                if duration <= max_duration and len(text) <= max_chars:
                    split_segments.append(segment)
                    continue

                # Split by duration
                if duration > max_duration:
                    num_splits = int(duration / max_duration) + 1
                    split_duration = duration / num_splits

                    for i in range(num_splits):
                        start_time = segment['start'] + i * split_duration
                        end_time = segment['start'] + (i + 1) * split_duration

                        split_segment = segment.copy()
                        split_segment['start'] = start_time
                        split_segment['end'] = end_time
                        split_segments.append(split_segment)

                # Split by text length
                elif len(text) > max_chars:
                    words = text.split()
                    current_text = ""
                    current_start = segment['start']

                    for word in words:
                        test_text = current_text + " " + word if current_text else word

                        if len(test_text) <= max_chars:
                            current_text = test_text
                        else:
                            # Create segment for current text
                            if current_text:
                                split_segment = segment.copy()
                                split_segment['start'] = current_start
                                split_segment['end'] = segment['end']  # Approximate
                                split_segment['text'] = current_text.strip()
                                split_segments.append(split_segment)

                            current_text = word
                            current_start = segment['start']  # Approximate

                    # Add remaining text
                    if current_text:
                        split_segment = segment.copy()
                        split_segment['start'] = current_start
                        split_segment['end'] = segment['end']
                        split_segment['text'] = current_text.strip()
                        split_segments.append(split_segment)

            logger.info(f"Split {len(segments)} segments into {len(split_segments)} segments")
            return split_segments

        except Exception as e:
            logger.error(f"Failed to split subtitles: {e}")
            return segments

    def validate_subtitles(self, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate subtitle segments for common issues.

        Args:
            segments: List of subtitle segments

        Returns:
            Validation results
        """
        try:
            issues = []
            warnings = []

            for i, segment in enumerate(segments):
                # Check required fields
                if 'start' not in segment or 'end' not in segment or 'text' not in segment:
                    issues.append(f"Segment {i}: Missing required fields")

                # Check timing
                if segment.get('start', 0) < 0:
                    issues.append(f"Segment {i}: Negative start time")

                if segment.get('end', 0) < segment.get('start', 0):
                    issues.append(f"Segment {i}: End time before start time")

                # Check text
                text = segment.get('text', '')
                if not text.strip():
                    issues.append(f"Segment {i}: Empty text")

                if len(text) > 200:
                    warnings.append(f"Segment {i}: Very long text ({len(text)} chars)")

                # Check duration
                duration = segment.get('end', 0) - segment.get('start', 0)
                if duration > 10:
                    warnings.append(f"Segment {i}: Very long duration ({duration:.1f}s)")
                elif duration < 0.5:
                    warnings.append(f"Segment {i}: Very short duration ({duration:.1f}s)")

            # Check overlaps
            for i in range(len(segments) - 1):
                current_end = segments[i].get('end', 0)
                next_start = segments[i + 1].get('start', 0)

                if current_end > next_start:
                    issues.append(f"Segments {i} and {i+1}: Overlapping timing")

            return {
                'valid': len(issues) == 0,
                'issues': issues,
                'warnings': warnings,
                'total_segments': len(segments)
            }

        except Exception as e:
            logger.error(f"Failed to validate subtitles: {e}")
            return {'valid': False, 'issues': [str(e)], 'warnings': [], 'total_segments': 0}
