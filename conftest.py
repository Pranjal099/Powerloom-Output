"""Pytest configuration for Powerloom Output tests.

Ensures 'src' directory is automatically in the Python module search path.
"""

from pathlib import Path
import sys

# Ensure src/ is on sys.path for test discovery and execution
SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
