"""hud_renderer.py
Clean border-only overlay for neon face box with no global blending.
All glow and effects drawn only where needed. No image graying.
Developed by Aravind
"""
import cv2
import numpy as np
import time
import os
from typing import Tuple, List
class HUDRenderer:
    CYAN_NEON = (220, 240, 255)
    BLUE_NEON = (180, 100, 255)
    PURPLE_NEON = (200, 120, 255)
    def __init__(self):
        self.scanline_y = 0
        self.last_update = time.time()
    def draw_face_box(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], color: Tuple[int, int, int] = None, thickness: int = 2, corner_radius: int = 22, glow: bool = True) -> np.ndarray:
        if color is None:
            color = self.CYAN_NEON
        x, y, w, h = bbox
        x1, y1, x2, y2 = x, y, x + w, y + h
        r = min(corner_radius, w // 4, h // 4)
        # Draw neon border directly
        cv2.line(frame, (x1 + r, y1), (x2 - r, y1), color, thickness)
        cv2.line(frame, (x1 + r, y2), (x2 - r, y2), color, thickness)
        cv2.line(frame, (x1, y1 + r), (x1, y2 - r), color, thickness)
        cv2.line(frame, (x2, y1 + r), (x2, y2 - r), color, thickness)
        cv2.ellipse(frame, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
        cv2.ellipse(frame, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
        cv2.ellipse(frame, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
        cv2.ellipse(frame, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)
        if glow:
            # Optional subtle blur globally
            glow_img = np.zeros_like(frame)
            cv2.line(glow_img, (x1 + r, y1), (x2 - r, y1), color, thickness + 8)
            cv2.line(glow_img, (x1 + r, y2), (x2 - r, y2), color, thickness + 8)
            cv2.line(glow_img, (x1, y1 + r), (x1, y2 - r), color, thickness + 8)
            cv2.line(glow_img, (x2, y1 + r), (x2, y2 - r), color, thickness + 8)
            cv2.ellipse(glow_img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness + 8)
            cv2.ellipse(glow_img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness + 8)
            cv2.ellipse(glow_img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness + 8)
            cv2.ellipse(glow_img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness + 8)
            blurred = cv2.GaussianBlur(glow_img, (45, 45), 0)
            frame = cv2.addWeighted(frame, 0.92, blurred, 0.08, 0)
        return frame
    def draw_emotion_info(self, frame: np.ndarray, emotion: str, confidence: float, persona: str, bbox: Tuple[int, int, int, int], alpha: float = 1.0) -> np.ndarray:
        x, y, w, h = bbox
        text_x = x
        text_y = max(22, y - 18)
        emotion_text = f"{emotion.upper()} ({int(confidence*100)}%)"
        cv2.putText(frame, emotion_text, (text_x+2, text_y+2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 4, cv2.LINE_AA)
        color = self.CYAN_NEON if alpha > 0.6 else self.BLUE_NEON
        cv2.putText(frame, emotion_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        persona_y = text_y + 25
        cv2.putText(frame, f"[{persona}]", (text_x, persona_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.CYAN_NEON, 1, cv2.LINE_AA)
        return frame
    def draw_scanline(self, frame: np.ndarray, color: Tuple[int, int, int] = None, thickness: int = 2) -> np.ndarray:
        if color is None:
            color = self.CYAN_NEON
        h, w = frame.shape[:2]
        now = time.time()
        dt = now - self.last_update
        self.last_update = now
        self.scanline_y = (self.scanline_y + int(dt * 200)) % h
        cv2.line(frame, (0, self.scanline_y), (w, self.scanline_y), color, thickness)
        cv2.line(frame, (0, self.scanline_y-3), (w, self.scanline_y-3), (color[0]//3, color[1]//3, color[2]//3), thickness+1)
        return frame
    def draw_glitch_header(self, frame: np.ndarray, text: str, position: Tuple[int, int] = (25, 45), base_color: Tuple[int, int, int] = None) -> np.ndarray:
        if base_color is None:
            base_color = self.PURPLE_NEON
        x, y = position
        cv2.putText(frame, text, (x-2, y), cv2.FONT_HERSHEY_DUPLEX, 0.9, (5,5,5), 7, cv2.LINE_AA)
        cv2.putText(frame, text, (x+2, y+2), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255,50,120), 2, cv2.LINE_AA)
        cv2.putText(frame, text, (x-2, y-2), cv2.FONT_HERSHEY_DUPLEX, 0.9, (120,220,255), 2, cv2.LINE_AA)
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, 0.9, base_color, 2, cv2.LINE_AA)
        return frame
    def draw_fps_counter(self, frame: np.ndarray, fps: float, position: Tuple[int, int] = (10, 30)) -> np.ndarray:
        text = f"FPS: {int(fps)}"
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.BLUE_NEON, 2, cv2.LINE_AA)
        return frame
    def draw_status_panel(self, frame: np.ndarray, status_lines: List[str], position: Tuple[int, int] = (12, 65), bg_alpha: float = 0.04) -> np.ndarray:
        # Minimal alpha for near-invisible panel
        x, y = position
        panel_width = 340
        line_height = 20
        panel_height = 25 + (line_height * len(status_lines))
        overlay = np.zeros_like(frame)
        bg_color = (15, 15, 25)
        cv2.rectangle(overlay, (x, y), (x + panel_width, y + panel_height), bg_color, -1)
        frame = cv2.addWeighted(frame, 1.0, overlay, bg_alpha, 0)
        for i, line in enumerate(status_lines):
            text_y = y + 20 + (i * line_height)
            cv2.putText(frame, line, (x + 10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.CYAN_NEON, 1, cv2.LINE_AA)
        return frame
    def save_screenshot(self, frame: np.ndarray, output_dir: str = 'assets/screenshots') -> str:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = int(time.time())
        filename = f"emotion_capture_{timestamp}.png"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, frame)
        return filepath
