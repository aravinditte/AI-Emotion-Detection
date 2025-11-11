"""detector.py

Face detection and landmark extraction using MediaPipe Face Mesh.
Provides a clean interface for detecting faces and extracting facial landmarks.

Developed by Aravind
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, List, Tuple


class FaceDetector:
    """Face detection wrapper around MediaPipe Face Mesh.
    
    Detects faces in video frames and extracts facial landmarks for emotion analysis.
    
    Attributes:
        max_faces (int): Maximum number of faces to detect
        face_mesh: MediaPipe Face Mesh solution instance
    """

    def __init__(self, max_faces: int = 1, refine_landmarks: bool = True):
        """Initialize the face detector.
        
        Args:
            max_faces: Maximum number of faces to detect per frame
            refine_landmarks: Whether to refine landmarks around eyes and lips
        """
        self.max_faces = max_faces
        self.mp_face_mesh = mp.solutions.face_mesh
        
        # Initialize Face Mesh with optimized parameters
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def detect_face(self, frame: np.ndarray) -> Tuple[Optional[Tuple[int, int, int, int]], List[Tuple[int, int]]]:
        """Detect face and extract landmarks from a frame.
        
        Args:
            frame: BGR image from webcam
            
        Returns:
            Tuple containing:
                - bbox: (x, y, width, height) or None if no face detected
                - landmarks: List of (x, y) landmark coordinates in pixel space
        """
        if frame is None or frame.size == 0:
            return None, []

        # Convert BGR to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = frame.shape[:2]
        
        try:
            results = self.face_mesh.process(frame_rgb)
        except Exception as e:
            print(f"Face detection error: {e}")
            return None, []

        # Check if any faces were detected
        if not results.multi_face_landmarks:
            return None, []

        # Process first detected face
        face_landmarks = results.multi_face_landmarks[0]

        # Extract landmark coordinates
        landmarks = []
        x_coords = []
        y_coords = []
        
        for landmark in face_landmarks.landmark:
            pixel_x = int(landmark.x * width)
            pixel_y = int(landmark.y * height)
            landmarks.append((pixel_x, pixel_y))
            x_coords.append(pixel_x)
            y_coords.append(pixel_y)

        # Calculate bounding box with padding
        if x_coords and y_coords:
            padding = 15
            x_min = max(min(x_coords) - padding, 0)
            y_min = max(min(y_coords) - padding, 0)
            x_max = min(max(x_coords) + padding, width)
            y_max = min(max(y_coords) + padding, height)
            
            bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
            return bbox, landmarks

        return None, landmarks

    def release(self):
        """Release MediaPipe resources."""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()


if __name__ == "__main__":
    # Test the face detector
    print("Testing Face Detector...")
    cap = cv2.VideoCapture(0)
    detector = FaceDetector()
    
    print("Press ESC to exit")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        bbox, landmarks = detector.detect_face(frame)
        
        if bbox:
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Landmarks: {len(landmarks)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Face Detector Test', frame)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    detector.release()
    print("Test complete")
