"""hud_renderer.py

Cyberpunk-style HUD rendering system for the emotion detection overlay.
Provides neon-themed visual elements with animated effects.

Developed by Aravind
"""

import cv2
import numpy as np
import time
import os
from typing import Tuple, List


class HUDRenderer:
    """Cyberpunk HUD renderer with neon effects.
    
    Provides methods for drawing animated UI elements including:
    - Glowing rounded rectangles
    - Scanline animations
    - Glitch-style text
    - Status panels
    - FPS displays
    """

    # Neon color palette (BGR format)
    CYAN_NEON = (220, 240, 255)
    BLUE_NEON = (180, 100, 255)
    PURPLE_NEON = (200, 120, 255)
    PINK_NEON = (180, 100, 255)
    
    def __init__(self):
        """Initialize HUD renderer."""
        self.scanline_y = 0
        self.last_update = time.time()

    def draw_face_box(self, frame: np.ndarray, 
                      bbox: Tuple[int, int, int, int],
                      color: Tuple[int, int, int] = None,
                      thickness: int = 2,
                      corner_radius: int = 25,
                      glow: bool = True) -> np.ndarray:
        """Draw a rounded rectangle with optional glow effect.
        
        Args:
            frame: Input image
            bbox: (x, y, width, height)
            color: BGR color tuple
            thickness: Line thickness
            corner_radius: Radius for rounded corners
            glow: Whether to add glow effect
            
        Returns:
            Frame with drawn rectangle
        """
        if color is None:
            color = self.CYAN_NEON
        
        x, y, w, h = bbox
        overlay = frame.copy()
        
        # Draw rounded rectangle components
        x1, y1, x2, y2 = x, y, x + w, y + h
        r = corner_radius
        
        # Draw straight lines
        cv2.line(overlay, (x1 + r, y1), (x2 - r, y1), color, thickness)
        cv2.line(overlay, (x1 + r, y2), (x2 - r, y2), color, thickness)
        cv2.line(overlay, (x1, y1 + r), (x1, y2 - r), color, thickness)
        cv2.line(overlay, (x2, y1 + r), (x2, y2 - r), color, thickness)
        
        # Draw corner arcs
        cv2.ellipse(overlay, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
        cv2.ellipse(overlay, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
        cv2.ellipse(overlay, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
        cv2.ellipse(overlay, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)
        
        # Add glow effect
        if glow:
            glow_mask = np.zeros_like(frame)
            cv2.rectangle(glow_mask, (x1, y1), (x2, y2), color, -1)
            
            # Multiple blur passes for glow
            for kernel_size, alpha in [(31, 0.15), (19, 0.10), (9, 0.05)]:
                blurred = cv2.GaussianBlur(glow_mask, (kernel_size, kernel_size), 0)
                frame = cv2.addWeighted(frame, 1.0, blurred, alpha, 0)
        
        # Blend overlay
        frame = cv2.addWeighted(frame, 1.0, overlay, 0.85, 0)
        return frame

    def draw_emotion_info(self, frame: np.ndarray,
                         emotion: str,
                         confidence: float,
                         persona: str,
                         bbox: Tuple[int, int, int, int],
                         alpha: float = 1.0) -> np.ndarray:
        """Draw emotion label and persona information.
        
        Args:
            frame: Input image
            emotion: Detected emotion label
            confidence: Confidence score (0-1)
            persona: Persona name
            bbox: Face bounding box (x, y, width, height)
            alpha: Display alpha for fade animation
            
        Returns:
            Frame with drawn text
        """
        x, y, w, h = bbox
        
        # Position above bounding box
        text_x = x
        text_y = max(15, y - 15)
        
        # Create emotion text
        emotion_text = f"{emotion.upper()} ({int(confidence * 100)}%)"
        
        # Add jitter effect during fade-in
        jitter = int(5 * (1 - alpha))
        offset_x = jitter if jitter > 0 else 0
        
        # Draw shadow for depth
        shadow_offset = 3
        cv2.putText(frame, emotion_text, 
                   (text_x + shadow_offset + offset_x, text_y + shadow_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.75, (15, 15, 15), 4, cv2.LINE_AA)
        
        # Draw main text with color based on alpha
        text_color = self.CYAN_NEON if alpha > 0.6 else self.BLUE_NEON
        cv2.putText(frame, emotion_text,
                   (text_x + offset_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.75, text_color, 2, cv2.LINE_AA)
        
        # Draw persona below
        persona_y = text_y + 28
        cv2.putText(frame, f"[{persona}]",
                   (text_x, persona_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.65, self.CYAN_NEON, 1, cv2.LINE_AA)
        
        return frame

    def draw_scanline(self, frame: np.ndarray,
                     color: Tuple[int, int, int] = None,
                     thickness: int = 3) -> np.ndarray:
        """Draw animated horizontal scanline.
        
        Args:
            frame: Input image
            color: Scanline color
            thickness: Line thickness
            
        Returns:
            Frame with drawn scanline
        """
        if color is None:
            color = self.CYAN_NEON
        
        h, w = frame.shape[:2]
        
        # Update scanline position
        now = time.time()
        dt = now - self.last_update
        self.last_update = now
        
        self.scanline_y = (self.scanline_y + int(dt * 200)) % h
        
        overlay = frame.copy()
        
        # Draw main scanline
        cv2.line(overlay, (0, self.scanline_y), (w, self.scanline_y), 
                color, thickness)
        
        # Draw faint trail
        trail_color = tuple(max(0, c // 3) for c in color)
        cv2.line(overlay, (0, self.scanline_y - 3), (w, self.scanline_y - 3),
                trail_color, thickness + 1)
        
        # Blend with low opacity
        frame = cv2.addWeighted(frame, 0.88, overlay, 0.45, 0)
        return frame

    def draw_glitch_header(self, frame: np.ndarray,
                          text: str,
                          position: Tuple[int, int] = (25, 45),
                          base_color: Tuple[int, int, int] = None) -> np.ndarray:
        """Draw glitch-style header text.
        
        Args:
            frame: Input image
            text: Header text
            position: (x, y) position
            base_color: Base text color
            
        Returns:
            Frame with drawn text
        """
        if base_color is None:
            base_color = self.PURPLE_NEON
        
        x, y = position
        
        # Draw shadow layer
        cv2.putText(frame, text, (x - 2, y), cv2.FONT_HERSHEY_DUPLEX, 
                   0.95, (5, 5, 5), 7, cv2.LINE_AA)
        
        # Draw RGB separation glitch effect
        cv2.putText(frame, text, (x + 2, y + 2), cv2.FONT_HERSHEY_DUPLEX,
                   0.95, (255, 50, 120), 2, cv2.LINE_AA)
        cv2.putText(frame, text, (x - 2, y - 2), cv2.FONT_HERSHEY_DUPLEX,
                   0.95, (120, 220, 255), 2, cv2.LINE_AA)
        
        # Draw main text
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_DUPLEX,
                   0.95, base_color, 2, cv2.LINE_AA)
        
        return frame

    def draw_fps_counter(self, frame: np.ndarray,
                        fps: float,
                        position: Tuple[int, int] = (10, 30)) -> np.ndarray:
        """Draw FPS counter.
        
        Args:
            frame: Input image
            fps: Current FPS value
            position: (x, y) position
            
        Returns:
            Frame with FPS display
        """
        text = f"FPS: {int(fps)}"
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                   0.85, self.BLUE_NEON, 2, cv2.LINE_AA)
        return frame

    def draw_status_panel(self, frame: np.ndarray,
                         status_lines: List[str],
                         position: Tuple[int, int] = (12, 65),
                         bg_alpha: float = 0.65) -> np.ndarray:
        """Draw translucent status information panel.
        
        Args:
            frame: Input image
            status_lines: List of status text lines
            position: (x, y) top-left position
            bg_alpha: Background transparency
            
        Returns:
            Frame with status panel
        """
        x, y = position
        
        # Calculate panel dimensions
        panel_width = 340
        line_height = 20
        panel_height = 25 + (line_height * len(status_lines))
        
        # Draw semi-transparent background
        overlay = frame.copy()
        bg_color = (15, 15, 25)
        cv2.rectangle(overlay, (x, y), (x + panel_width, y + panel_height),
                     bg_color, -1)
        frame = cv2.addWeighted(frame, 1.0, overlay, bg_alpha, 0)
        
        # Draw status lines
        for i, line in enumerate(status_lines):
            text_y = y + 20 + (i * line_height)
            cv2.putText(frame, line, (x + 10, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.CYAN_NEON, 1, cv2.LINE_AA)
        
        return frame

    def save_screenshot(self, frame: np.ndarray,
                       output_dir: str = 'assets/screenshots') -> str:
        """Save screenshot with timestamp.
        
        Args:
            frame: Frame to save
            output_dir: Output directory path
            
        Returns:
            Path to saved screenshot
        """
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = int(time.time())
        filename = f"emotion_capture_{timestamp}.png"
        filepath = os.path.join(output_dir, filename)
        
        cv2.imwrite(filepath, frame)
        return filepath


if __name__ == "__main__":
    # Test HUD rendering
    print("Testing HUD Renderer...")
    
    renderer = HUDRenderer()
    cap = cv2.VideoCapture(0)
    start_time = time.time()
    
    print("Press ESC to exit, S to save screenshot")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        h, w = frame.shape[:2]
        
        # Demo: draw face box in center
        bbox = (w // 4, h // 4, w // 2, h // 2)
        frame = renderer.draw_face_box(frame, bbox, glow=True)
        
        # Draw emotion info
        frame = renderer.draw_emotion_info(frame, "Happy", 0.89, 
                                          "Code Dreamer", bbox, alpha=1.0)
        
        # Draw scanline
        frame = renderer.draw_scanline(frame)
        
        # Draw header
        frame = renderer.draw_glitch_header(frame, "EMOTION DETECTION SYSTEM")
        
        # Draw FPS
        fps = 30
        frame = renderer.draw_fps_counter(frame, fps)
        
        # Draw status
        status = [
            "Mode: Heuristic",
            "DL: OFF",
            "Press S for screenshot"
        ]
        frame = renderer.draw_status_panel(frame, status)
        
        cv2.imshow('HUD Renderer Test', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == ord('s') or key == ord('S'):
            path = renderer.save_screenshot(frame)
            print(f"Screenshot saved: {path}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("Test complete")
