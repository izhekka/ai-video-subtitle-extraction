#!/usr/bin/env python3
"""
Setup script for AI Video Subtitle Extraction Agent.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ai-video-subtitles",
    version="1.0.0",
    author="AI Video Subtitle Agent",
    author_email="",
    description="An intelligent Python agent that automatically extracts subtitles from MP4 videos using advanced speech recognition and AI technologies.",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Multimedia :: Video",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    entry_points={
        "console_scripts": [
            "subtitle-agent=src.main:cli",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="video subtitles speech recognition ai whisper moviepy opencv",
    project_urls={
        "Bug Reports": "",
        "Source": "",
        "Documentation": "",
    },
)
