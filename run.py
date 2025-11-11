"""run.py

Convenience script to run the emotion detection application.

Developed by Aravind
"""

import sys
import os

# Add emotion_detector to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'emotion_detector'))

from emotion_detector.app import main

if __name__ == "__main__":
    sys.exit(main())
