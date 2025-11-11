"""emotion_analyzer.py

Emotion classification engine with multiple detection modes.
Supports heuristic-based detection, ONNX models, and DeepFace integration.

Developed by Aravind
"""

import time
import numpy as np
import cv2
import os
from typing import List, Tuple, Dict
from collections import deque, defaultdict
import threading

# Optional imports for deep learning
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    ort = None

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    DeepFace = None


class EmotionAnalyzer:
    """Emotion detection and analysis engine.
    
    Supports multiple detection modes:
    - Heuristic: Fast geometric feature analysis
    - ONNX: Deep learning via ONNX runtime
    - DeepFace: Full deep learning pipeline
    
    Features temporal smoothing and persona mapping.
    """

    # Cyberpunk-themed persona mappings
    PERSONA_MAP = {
        'joy': 'Circuit Dancer',
        'surprise': 'Neon Explorer',
        'anger': 'Cyber Warrior',
        'sadness': 'Digital Wanderer',
        'confused': 'Quantum Thinker',
        'happy': 'Code Dreamer',
        'excited': 'Energy Surger',
        'fear': 'Shadow Watcher',
        'disgust': 'System Skeptic',
        'neutral': 'Data Observer'
    }

    def __init__(self, mode: str = 'heuristic'):
        """Initialize emotion analyzer.
        
        Args:
            mode: Detection mode - 'heuristic', 'onnx', or 'deepface'
        """
        self.mode = mode
        self.dl_backend = 'onnx'  # Default DL backend
        self.use_dl = False  # DL disabled by default
        
        # Temporal smoothing parameters
        self.history = deque(maxlen=10)
        self.smoothing_factor = 0.85
        self.bbox_lerp = 0.20
        
        # Animation state
        self.current_label = 'neutral'
        self.display_alpha = 1.0
        self.last_change_time = time.time()
        
        # ONNX session cache
        self._onnx_session = None
        self._onnx_lock = threading.Lock()
        self._model_path = 'assets/emotion_model.onnx'
        self._model_url = 'https://github.com/onnx/models/raw/main/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx'

    def analyze_emotion(self, landmarks: List[Tuple[int, int]], 
                       frame_shape: Tuple[int, int]) -> Tuple[str, float, str, float]:
        """Analyze emotion from facial landmarks.
        
        Args:
            landmarks: List of (x, y) facial landmark coordinates
            frame_shape: (height, width) of the frame
            
        Returns:
            Tuple of (emotion_label, confidence, persona, display_alpha)
        """
        if not landmarks:
            return self._get_neutral_result()

        # Use heuristic detection
        emotion, confidence = self._heuristic_detection(landmarks, frame_shape)
        
        # Add to history for temporal smoothing
        self.history.append((emotion, confidence))
        
        # Compute smoothed emotion
        smoothed_emotion, smoothed_conf = self._apply_temporal_smoothing()
        
        # Map to persona
        persona = self.PERSONA_MAP.get(smoothed_emotion.lower(), self.PERSONA_MAP['neutral'])
        
        # Update animation state
        alpha = self._update_animation_state(smoothed_emotion)
        
        return smoothed_emotion, smoothed_conf, persona, alpha

    def analyze_with_dl(self, frame: np.ndarray, 
                       bbox: Tuple[int, int, int, int]) -> Tuple[str, float, str, float]:
        """Analyze emotion using deep learning.
        
        Args:
            frame: BGR image frame
            bbox: (x, y, width, height) face bounding box
            
        Returns:
            Tuple of (emotion_label, confidence, persona, display_alpha)
        """
        try:
            if self.dl_backend == 'onnx' and ONNX_AVAILABLE:
                emotion, conf = self._onnx_inference(frame, bbox)
            elif self.dl_backend == 'deepface' and DEEPFACE_AVAILABLE:
                emotion, conf = self._deepface_inference(frame, bbox)
            else:
                emotion, conf = 'neutral', 0.5
        except Exception as e:
            print(f"DL inference error: {e}")
            emotion, conf = 'neutral', 0.5
        
        persona = self.PERSONA_MAP.get(emotion.lower(), self.PERSONA_MAP['neutral'])
        alpha = self._update_animation_state(emotion)
        
        return emotion, conf, persona, alpha

    def _heuristic_detection(self, landmarks: List[Tuple[int, int]], 
                            frame_shape: Tuple[int, int]) -> Tuple[str, float]:
        """Fast heuristic-based emotion detection.
        
        Analyzes geometric features like eye aspect ratio, mouth shape,
        and facial proportions to determine emotion.
        """
        height, width = frame_shape
        
        # Get key landmark points (MediaPipe indices)
        # Mouth corners
        left_mouth = landmarks[61] if len(landmarks) > 61 else landmarks[0]
        right_mouth = landmarks[291] if len(landmarks) > 291 else landmarks[0]
        # Lip points
        upper_lip = landmarks[13] if len(landmarks) > 13 else landmarks[0]
        lower_lip = landmarks[14] if len(landmarks) > 14 else landmarks[0]
        
        # Eye points
        left_eye_top = landmarks[159] if len(landmarks) > 159 else landmarks[0]
        left_eye_bottom = landmarks[145] if len(landmarks) > 145 else landmarks[0]
        right_eye_top = landmarks[386] if len(landmarks) > 386 else landmarks[0]
        right_eye_bottom = landmarks[374] if len(landmarks) > 374 else landmarks[0]
        
        # Calculate distances
        def euclidean_dist(p1, p2):
            return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        
        # Mouth metrics
        mouth_width = euclidean_dist(left_mouth, right_mouth)
        mouth_height = euclidean_dist(upper_lip, lower_lip)
        
        # Eye metrics
        left_eye_height = abs(left_eye_top[1] - left_eye_bottom[1])
        right_eye_height = abs(right_eye_top[1] - right_eye_bottom[1])
        eye_aspect = (left_eye_height + right_eye_height) / 2.0
        
        # Normalize by face height
        norm_mouth_width = mouth_width / (height + 1e-6)
        norm_mouth_height = mouth_height / (height + 1e-6)
        norm_eye_aspect = eye_aspect / (height + 1e-6)
        
        # Compute emotion scores
        smile_score = norm_mouth_width - 0.75 * norm_mouth_height
        open_mouth = norm_mouth_height * 3.5
        surprise_score = open_mouth + (norm_eye_aspect * 2.5)
        squint_factor = 1.0 - norm_eye_aspect
        
        # Emotion classification logic
        emotion = 'neutral'
        confidence = 0.5
        
        if smile_score > 0.04 and norm_eye_aspect < 0.022:
            emotion = 'happy'
            confidence = min(0.95, 0.5 + smile_score * 10)
        elif surprise_score > 0.15:
            emotion = 'surprise'
            confidence = min(0.95, 0.4 + (surprise_score - 0.15) * 4)
        elif squint_factor > 0.05 and norm_mouth_height < 0.015:
            emotion = 'anger'
            confidence = min(0.90, 0.3 + squint_factor * 7)
        elif norm_mouth_height > 0.09 and smile_score > 0.025:
            emotion = 'excited'
            confidence = min(0.95, 0.4 + (norm_mouth_height - 0.09) * 6)
        elif norm_mouth_height > 0.07 and smile_score < 0.015:
            emotion = 'sadness'
            confidence = min(0.85, 0.25 + (norm_mouth_height - 0.07) * 5)
        elif abs(norm_eye_aspect - 0.035) > 0.025:
            emotion = 'confused'
            confidence = min(0.85, 0.35 + abs(norm_eye_aspect - 0.035) * 12)
        
        return emotion, float(max(0.0, min(1.0, confidence)))

    def _onnx_inference(self, frame: np.ndarray, 
                       bbox: Tuple[int, int, int, int]) -> Tuple[str, float]:
        """Perform inference using ONNX model."""
        if not ONNX_AVAILABLE:
            raise RuntimeError("ONNX runtime not available")
        
        # Ensure model exists
        if not os.path.exists(self._model_path):
            self._download_onnx_model()
        
        # Extract face region
        x, y, w, h = bbox
        H, W = frame.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(W, x + w), min(H, y + h)
        face = frame[y1:y2, x1:x2]
        
        if face.size == 0:
            return 'neutral', 0.5
        
        # Preprocess for FER+ model (grayscale 64x64)
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (64, 64))
        normalized = (resized / 255.0).astype(np.float32)
        input_tensor = normalized[None, None, :, :]
        
        # Load or get cached session
        with self._onnx_lock:
            if self._onnx_session is None:
                self._onnx_session = ort.InferenceSession(
                    self._model_path, 
                    providers=['CPUExecutionProvider']
                )
        
        # Run inference
        input_name = self._onnx_session.get_inputs()[0].name
        outputs = self._onnx_session.run(None, {input_name: input_tensor})
        
        # Process outputs
        logits = np.array(outputs[0]).flatten()
        probs = self._softmax(logits)
        
        # FER+ emotion labels
        fer_labels = ['neutral', 'happiness', 'surprise', 'sadness', 
                     'anger', 'disgust', 'fear', 'contempt']
        
        idx = int(np.argmax(probs))
        fer_emotion = fer_labels[idx] if idx < len(fer_labels) else 'neutral'
        confidence = float(probs[idx])
        
        # Map to our emotion labels
        emotion_map = {
            'neutral': 'neutral',
            'happiness': 'happy',
            'surprise': 'surprise',
            'sadness': 'sadness',
            'anger': 'anger',
            'disgust': 'disgust',
            'fear': 'confused',
            'contempt': 'neutral'
        }
        
        emotion = emotion_map.get(fer_emotion, 'neutral')
        return emotion, confidence

    def _deepface_inference(self, frame: np.ndarray, 
                           bbox: Tuple[int, int, int, int]) -> Tuple[str, float]:
        """Perform inference using DeepFace."""
        if not DEEPFACE_AVAILABLE:
            raise RuntimeError("DeepFace not available")
        
        # Extract face region
        x, y, w, h = bbox
        H, W = frame.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(W, x + w), min(H, y + h)
        face = frame[y1:y2, x1:x2]
        
        if face.size == 0:
            return 'neutral', 0.5
        
        try:
            result = DeepFace.analyze(face, actions=['emotion'], 
                                     enforce_detection=False)
            
            if isinstance(result, list):
                result = result[0]
            
            emotions = result.get('emotion', {})
            if not emotions:
                return 'neutral', 0.5
            
            # Get top emotion
            top_emotion = max(emotions.items(), key=lambda x: x[1])
            emotion_label, emotion_score = top_emotion
            
            # Normalize confidence
            confidence = emotion_score / 100.0 if emotion_score > 1 else emotion_score
            
            # Map DeepFace labels to our labels
            label_map = {
                'happy': 'joy',
                'sad': 'sadness',
                'angry': 'anger',
                'surprise': 'surprise',
                'neutral': 'neutral',
                'disgust': 'disgust',
                'fear': 'fear'
            }
            
            mapped_emotion = label_map.get(emotion_label.lower(), 'neutral')
            return mapped_emotion, float(confidence)
            
        except Exception as e:
            print(f"DeepFace analysis error: {e}")
            return 'neutral', 0.5

    def _apply_temporal_smoothing(self) -> Tuple[str, float]:
        """Apply temporal smoothing to emotion predictions."""
        if not self.history:
            return 'neutral', 0.5
        
        # Aggregate scores with recency weighting
        emotion_scores = defaultdict(float)
        weight = 1.0
        total_weight = 0.0
        
        for emotion, confidence in reversed(self.history):
            emotion_scores[emotion] += confidence * weight
            total_weight += weight
            weight *= self.smoothing_factor
        
        # Select emotion with highest weighted score
        best_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
        smoothed_confidence = emotion_scores[best_emotion] / max(total_weight, 1e-6)
        
        return best_emotion, float(smoothed_confidence)

    def _update_animation_state(self, emotion: str) -> float:
        """Update fade animation state for emotion label changes."""
        if emotion != self.current_label:
            self.current_label = emotion
            self.last_change_time = time.time()
            self.display_alpha = 0.0
        
        # Fade in over 0.3 seconds
        elapsed = time.time() - self.last_change_time
        self.display_alpha = min(1.0, elapsed / 0.3)
        
        return self.display_alpha

    def _get_neutral_result(self) -> Tuple[str, float, str, float]:
        """Return neutral emotion result."""
        return 'neutral', 0.5, self.PERSONA_MAP['neutral'], 1.0

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax probabilities."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / (np.sum(exp_x) + 1e-9)

    def _download_onnx_model(self):
        """Download ONNX model if not present."""
        os.makedirs(os.path.dirname(self._model_path), exist_ok=True)
        
        print(f"Downloading ONNX emotion model...")
        try:
            import requests
            response = requests.get(self._model_url, stream=True, timeout=60)
            response.raise_for_status()
            
            with open(self._model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            print(f"Model downloaded to {self._model_path}")
        except Exception as e:
            print(f"Failed to download ONNX model: {e}")
            raise RuntimeError(f"Could not acquire ONNX model: {e}")


if __name__ == "__main__":
    # Test the emotion analyzer
    print("Testing Emotion Analyzer...")
    analyzer = EmotionAnalyzer(mode='heuristic')
    print(f"Mode: {analyzer.mode}")
    print(f"Available backends: ONNX={ONNX_AVAILABLE}, DeepFace={DEEPFACE_AVAILABLE}")
    print("Test complete")
