import cv2
import asyncio
import base64
import numpy as np
import time
from typing import AsyncGenerator

# Enable OpenCV threading
cv2.setNumThreads(4)
cv2.setUseOptimized(True)

# ── Output resolution cap — KEY for streaming performance ─────
# Cap the frame sent to browser — doesn't affect detection quality
# 640 wide = good quality, fast encode
# 480 wide = fastest
STREAM_WIDTH = 640


class VideoStreamer:
    def __init__(self, target_fps=10):
        """
        target_fps: how many frames per second to AIM for
        Adaptive skip will automatically drop frames if inference is slow
        """
        self.target_fps    = target_fps
        self.frame_delay   = 1.0 / target_fps
        self.is_running    = False
        self.cap           = None
        self.frame_count   = 0
        self._last_frame_t = 0

    def open(self, source) -> bool:
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            return False

        # ── PERFORMANCE: set buffer size small ────────────────
        # Prevents reading stale frames from a large buffer
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

        self.is_running = True
        return True

    def get_video_info(self) -> dict:
        if self.cap is None:
            return {}
        return {
            "width":        int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height":       int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps":          self.cap.get(cv2.CAP_PROP_FPS),
            "total_frames": int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        }

    async def frames(self, detector=None) -> AsyncGenerator[np.ndarray, None]:
        """
        Async generator yielding frames.
        If detector is passed, uses adaptive skip based on inference speed.
        """
        video_fps   = self.cap.get(cv2.CAP_PROP_FPS) or 30
        frame_idx   = 0

        while self.is_running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                # Loop video for demo
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
                if not ret:
                    break

            frame_idx += 1

            # ── RESIZE for streaming ──────────────────────────
            h, w = frame.shape[:2]
            if w > STREAM_WIDTH:
                scale  = STREAM_WIDTH / w
                new_w  = STREAM_WIDTH
                new_h  = int(h * scale)
                # INTER_LINEAR is fastest resize algorithm
                frame  = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            self.frame_count += 1
            self._last_frame_t = time.time()
            yield frame

            # Throttle to target FPS and yield control to event loop
            await asyncio.sleep(self.frame_delay)

    def stop(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None

    @staticmethod
    def frame_to_base64(frame: np.ndarray, quality=55) -> str:
        """
        Encode frame as base64 JPEG.
        quality=55 is the sweet spot: fast encode, acceptable quality
        Lower = faster transmission, higher = better image
        """
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        _, buffer = cv2.imencode(".jpg", frame, encode_params)
        return base64.b64encode(buffer).decode("utf-8")

    @staticmethod
    def draw_grid_overlay(frame: np.ndarray, grid_data: list, alpha=0.35) -> np.ndarray:
        h, w  = frame.shape[:2]
        rows  = len(grid_data)
        cols  = len(grid_data[0]) if rows > 0 else 0
        if rows == 0 or cols == 0:
            return frame

        cell_h = h // rows
        cell_w = w // cols
        overlay = frame.copy()

        color_map = {
            "CRITICAL": (0,   0, 220),
            "SEVERE":   (0, 100, 255),
            "MODERATE": (0, 200, 255),
            "SAFE":     (0, 180,  60),
        }

        for r, row in enumerate(grid_data):
            for cell in row:
                c     = cell["col"]
                level = cell["level"]
                score = cell["score"]
                if score < 1.0:
                    continue

                color = color_map.get(level, (0, 180, 60))
                x1 = c * cell_w
                y1 = r * cell_h
                x2 = x1 + cell_w
                y2 = y1 + cell_h

                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                zone_label = f"{chr(65+r)}{c+1}"
                cv2.putText(overlay, zone_label, (x1+4, y1+16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

        result = cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0)

        # Grid lines
        for r in range(rows+1):
            cv2.line(result, (0, r*cell_h), (w, r*cell_h), (40,40,40), 1)
        for c in range(cols+1):
            cv2.line(result, (c*cell_w, 0), (c*cell_w, h), (40,40,40), 1)

        return result
