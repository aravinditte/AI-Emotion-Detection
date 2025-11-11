"""app.py

Main application for real-time emotion detection with cyberpunk HUD overlay.
Combines face detection, emotion analysis, and visual rendering.

Developed by Aravind
"""

import cv2
import time
import sys
import traceback

try:
    import winsound
    SOUND_AVAILABLE = True
except ImportError:
    SOUND_AVAILABLE = False

from detector import FaceDetector
from emotion_analyzer import EmotionAnalyzer
from hud_renderer import HUDRenderer


def play_shutter_sound():
    """Play camera shutter sound effect (Windows only)."""
    if not SOUND_AVAILABLE:
        return
    
    try:
        winsound.Beep(1100, 75)
        winsound.Beep(1500, 55)
    except Exception:
        pass


def main():
    """Main application loop."""
    print("="*60)
    print("AI Emotion Detection with Cyberpunk HUD")
    print("Developed by Aravind")
    print("="*60)
    print()
    
    # Initialize components
    cap = None
    detector = None
    analyzer = None
    renderer = None
    
    try:
        # Open webcam with default settings (best quality)
        print("Initializing webcam...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("ERROR: Could not open webcam!")
            print("Please ensure:")
            print("  1. A webcam is connected")
            print("  2. No other application is using the camera")
            print("  3. Camera permissions are granted")
            return 1
        
        # Don't set camera properties - use defaults for best quality
        # This prevents overexposure and quality issues
        
        print("Webcam initialized successfully")
        
        # Initialize detector with higher confidence
        print("Loading face detector...")
        detector = FaceDetector(max_faces=1, refine_landmarks=True)
        print("Face detector loaded")
        
        # Initialize emotion analyzer (heuristic mode by default)
        print("Loading emotion analyzer...")
        analyzer = EmotionAnalyzer(mode='heuristic')
        # Use same bbox lerp as reference (0.22)
        analyzer.bbox_lerp = 0.22
        print("Emotion analyzer loaded")
        
        # Initialize HUD renderer
        print("Initializing HUD renderer...")
        renderer = HUDRenderer()
        print("HUD renderer initialized")
        
        print()
        print("Application ready!")
        print()
        print("Controls:")
        print("  S       - Save screenshot")
        print("  +/=     - Increase smoothing")
        print("  -       - Decrease smoothing")
        print("  ]       - Increase bbox tracking speed")
        print("  [       - Decrease bbox tracking speed")
        print("  D       - Toggle deep learning mode")
        print("  M       - Switch DL backend (ONNX/DeepFace)")
        print("  ESC     - Exit")
        print()
        print("Starting real-time detection...")
        print()
        
        # Performance tracking
        fps_smooth = 30.0
        last_time = time.time()
        
        # Display state
        smoothed_bbox = None
        use_dl = False
        dl_backend = analyzer.dl_backend
        
        # Main loop
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("Warning: Failed to read frame, exiting...")
                break
            
            h, w = frame.shape[:2]
            
            # Detect face
            bbox, landmarks = detector.detect_face(frame)
            
            # Smooth bounding box movement
            if bbox:
                if smoothed_bbox is None:
                    smoothed_bbox = bbox
                else:
                    x0, y0, w0, h0 = smoothed_bbox
                    x1, y1, w1, h1 = bbox
                    lerp = analyzer.bbox_lerp
                    
                    smoothed_bbox = (
                        int(x0 + (x1 - x0) * lerp),
                        int(y0 + (y1 - y0) * lerp),
                        int(w0 + (w1 - w0) * lerp),
                        int(h0 + (h1 - h0) * lerp)
                    )
            else:
                smoothed_bbox = None
            
            # Analyze emotion
            emotion = 'neutral'
            confidence = 0.0
            persona = 'Data Observer'
            alpha = 1.0
            
            if use_dl and bbox:
                # Use deep learning
                try:
                    emotion, confidence, persona, alpha = analyzer.analyze_with_dl(frame, bbox)
                except Exception as e:
                    print(f"DL analysis error: {e}")
                    # Fallback to heuristic
                    if landmarks:
                        emotion, confidence, persona, alpha = analyzer.analyze_emotion(landmarks, (h, w))
            elif landmarks:
                # Use heuristic detection
                emotion, confidence, persona, alpha = analyzer.analyze_emotion(landmarks, (h, w))
            
            # Render HUD
            if smoothed_bbox:
                # Draw face box
                frame = renderer.draw_face_box(frame, smoothed_bbox, glow=True, corner_radius=22)
                frame = renderer.draw_emotion_info(frame, emotion, confidence, 
                                                  persona, smoothed_bbox, alpha)
            
            # Draw scanline animation
            frame = renderer.draw_scanline(frame)
            
            # Draw header
            frame = renderer.draw_glitch_header(frame, "AI EMOTION DETECTION SYSTEM")
            
            # Calculate and display FPS
            now = time.time()
            frame_time = now - last_time
            last_time = now
            
            if frame_time > 0:
                fps = 1.0 / frame_time
                fps_smooth = fps_smooth * 0.85 + fps * 0.15
            
            frame = renderer.draw_fps_counter(frame, fps_smooth)
            
            # Draw status panel
            status_lines = [
                f"Smoothing: {analyzer.smoothing_factor:.2f}  BBox Lerp: {analyzer.bbox_lerp:.2f}",
                f"DL Mode: {'ON' if use_dl else 'OFF'}  Backend: {dl_backend}",
                "Controls: +/- smoothing | [/] bbox | D toggle DL | M backend"
            ]
            frame = renderer.draw_status_panel(frame, status_lines)
            
            # Display frame
            cv2.imshow('AI Emotion Detection', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                print("Exiting...")
                break
            
            elif key == ord('s') or key == ord('S'):
                filepath = renderer.save_screenshot(frame)
                print(f"Screenshot saved: {filepath}")
                play_shutter_sound()
            
            elif key == ord('+') or key == ord('='):
                analyzer.smoothing_factor = min(0.98, analyzer.smoothing_factor + 0.03)
                print(f"Smoothing: {analyzer.smoothing_factor:.2f}")
            
            elif key == ord('-'):
                analyzer.smoothing_factor = max(0.5, analyzer.smoothing_factor - 0.03)
                print(f"Smoothing: {analyzer.smoothing_factor:.2f}")
            
            elif key == ord(']'):
                analyzer.bbox_lerp = min(0.9, analyzer.bbox_lerp + 0.05)
                print(f"BBox Lerp: {analyzer.bbox_lerp:.2f}")
            
            elif key == ord('['):
                analyzer.bbox_lerp = max(0.02, analyzer.bbox_lerp - 0.05)
                print(f"BBox Lerp: {analyzer.bbox_lerp:.2f}")
            
            elif key == ord('d') or key == ord('D'):
                use_dl = not use_dl
                print(f"Deep Learning: {'ON' if use_dl else 'OFF'}")
            
            elif key == ord('m') or key == ord('M'):
                dl_backend = 'onnx' if dl_backend == 'deepface' else 'deepface'
                analyzer.dl_backend = dl_backend
                print(f"DL Backend: {dl_backend}")
        
        print()
        print("Application closed successfully")
        return 0
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 0
    
    except Exception as e:
        print("\nFATAL ERROR:")
        print(traceback.format_exc())
        return 1
    
    finally:
        # Cleanup
        print("Cleaning up resources...")
        
        if cap is not None:
            cap.release()
        
        if detector is not None:
            detector.release()
        
        cv2.destroyAllWindows()
        print("Cleanup complete")


if __name__ == "__main__":
    sys.exit(main())
