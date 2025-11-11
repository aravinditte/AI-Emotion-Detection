"""AI Emotion Detection Package

Real-time emotion detection with cyberpunk HUD overlay.

Developed by Aravind
"""

__version__ = "1.0.0"
__author__ = "Aravind"

from .detector import FaceDetector
from .emotion_analyzer import EmotionAnalyzer
from .hud_renderer import HUDRenderer

__all__ = ['FaceDetector', 'EmotionAnalyzer', 'HUDRenderer']
