import cv2
import numpy as np
from ultralytics import YOLO
import os
import threading
import queue
import time

# PyTorch 2.6+ safety
try:
    import torch.serialization as _ts
    import torch.nn.modules.container as _containers
    from ultralytics.nn.tasks import DetectionModel as _UltralyticsDetectionModel
    _ts.add_safe_globals([_UltralyticsDetectionModel, _containers.Sequential])
except Exception:
    pass

# Enable OpenCV threading optimizations
cv2.setNumThreads(4)
cv2.setUseOptimized(True)

DISASTER_CLASSES = {
    "Damaged_buildings": {"weight": 2.8, "color": (0, 50, 255),   "category": "structural"},
    "Damaged_road":      {"weight": 1.8, "color": (0, 120, 255),  "category": "infrastructure"},
    "Damaged_vehicle":   {"weight": 1.5, "color": (0, 165, 255),  "category": "debris"},
    "Debris":            {"weight": 1.6, "color": (0, 200, 255),  "category": "debris"},
    "Flood":             {"weight": 2.2, "color": (200, 100, 0),  "category": "flood"},
    "Injured_person":    {"weight": 3.5, "color": (0, 0, 255),    "category": "casualty"},
    "Uninjured_person":  {"weight": 2.0, "color": (0, 255, 100),  "category": "casualty"},
    "severe":            {"weight": 3.0, "color": (0, 0, 220),    "category": "severity"},
    "moderate":          {"weight": 2.0, "color": (0, 100, 255),  "category": "severity"},
    "minor":             {"weight": 1.0, "color": (0, 200, 180),  "category": "severity"},
    "pothole":           {"weight": 1.2, "color": (50, 150, 255), "category": "infrastructure"},
    "person":            {"weight": 2.5, "color": (0, 255, 100),  "category": "casualty"},
    "car":               {"weight": 1.5, "color": (0, 165, 255),  "category": "debris"},
    "truck":             {"weight": 1.8, "color": (0, 140, 255),  "category": "debris"},
    "bus":               {"weight": 1.8, "color": (0, 140, 255),  "category": "debris"},
}

COCO_FALLBACK = {
    0: "person", 2: "car", 5: "bus", 7: "truck",
    56: "Debris", 57: "Damaged_buildings", 60: "Damaged_road",
}

# ── Inference resolution — KEY performance setting ────────────
# 320 = fastest, acceptable accuracy for demo (~3x faster than 640)
# 416 = balanced
# 640 = slowest, highest accuracy
INFERENCE_SIZE = 320


class DisasterDetector:
    def __init__(self, model_path=None):
        self.using_custom_model = False
        self.class_names = []

        # ── PERFORMANCE: always try nano first ────────────────
        # nano = ~40ms/frame on CPU vs ~150ms for small
        custom_model_path = os.path.join(
            os.path.dirname(__file__), "..", "models", "disaster_model.pt"
        )

        if model_path and os.path.exists(model_path):
            print(f"[Detector] Loading custom model: {model_path}")
            self.model = YOLO(model_path)
            self.using_custom_model = True
        elif os.path.exists(custom_model_path):
            print(f"[Detector] Loading disaster model from /models/")
            self.model = YOLO(custom_model_path)
            self.using_custom_model = True
        else:
            # ── CHANGED: back to nano for CPU performance ─────
            print("[Detector] Using YOLOv8n (nano) — optimized for CPU speed")
            self.model = YOLO("yolov8n.pt")

        # ── PERFORMANCE: set inference image size ─────────────
        self.model.conf = 0.28
        self.model.iou  = 0.45

        if hasattr(self.model, "names"):
            self.class_names = self.model.names

        # ── Async inference thread setup ──────────────────────
        self._frame_queue  = queue.Queue(maxsize=2)
        self._result_queue = queue.Queue(maxsize=2)
        self._running      = True
        self._last_result  = []
        self._infer_thread = threading.Thread(target=self._inference_worker, daemon=True)
        self._infer_thread.start()

        # Timing tracker for adaptive skip
        self.last_inference_ms = 80.0

        print(f"[Detector] Ready — inference_size={INFERENCE_SIZE}, async=True")

    def _inference_worker(self):
        """Background thread: pulls frames, runs YOLO, pushes results"""
        while self._running:
            try:
                frame = self._frame_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            t0 = time.time()

            # ── PERFORMANCE: resize before inference ──────────
            h, w = frame.shape[:2]
            results = self.model(
                frame,
                imgsz=INFERENCE_SIZE,   # run at 320 instead of native res
                verbose=False,
                half=False,             # half precision off (CPU doesn't support fp16)
            )[0]

            detections = []
            frame_area = max(1.0, float(w * h))

            for box in results.boxes:
                cls_id      = int(box.cls[0])
                conf        = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if self.using_custom_model and cls_id < len(self.class_names):
                    class_name = self.class_names[cls_id]
                else:
                    class_name = COCO_FALLBACK.get(cls_id, None)
                    if class_name is None:
                        continue
                    box_area   = max(1.0, float((x2-x1)*(y2-y1)))
                    area_ratio = box_area / frame_area
                    if class_name in ("truck","bus","car") and area_ratio > 0.04:
                        class_name = "Damaged_buildings"

                info = self._get_disaster_info(class_name)
                if info is None:
                    continue

                detections.append({
                    "class_name":    class_name,
                    "disaster_type": class_name,
                    "confidence":    conf,
                    "bbox":          [x1, y1, x2, y2],
                    "weight":        info["weight"],
                    "color":         info["color"],
                    "category":      info["category"],
                })

            elapsed_ms = (time.time() - t0) * 1000
            self.last_inference_ms = elapsed_ms

            # Push result, drop old if full
            try:
                self._result_queue.put_nowait(detections)
            except queue.Full:
                try: self._result_queue.get_nowait()
                except: pass
                self._result_queue.put_nowait(detections)

            self._frame_queue.task_done()

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Non-blocking detect:
        - Submits frame to background inference thread
        - Returns latest available result immediately
        - Never blocks the streaming loop
        """
        # Submit frame (drop if worker is busy)
        try:
            self._frame_queue.put_nowait(frame)
        except queue.Full:
            pass  # worker busy — reuse last result

        # Grab latest result if available
        try:
            self._last_result = self._result_queue.get_nowait()
        except queue.Empty:
            pass  # no new result yet — reuse last

        return self._last_result

    def _get_disaster_info(self, class_name: str) -> dict | None:
        if class_name in DISASTER_CLASSES:
            return DISASTER_CLASSES[class_name]
        for key in DISASTER_CLASSES:
            if key.lower() == class_name.lower():
                return DISASTER_CLASSES[key]
        return None

    def annotate_frame(self, frame: np.ndarray, detections: list[dict]) -> np.ndarray:
        annotated = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            color = det["color"]
            label = f"{det['disaster_type']} {det['confidence']:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1-th-6), (x1+tw+4, y1), color, -1)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, label, (x1+2, y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        return annotated

    def shutdown(self):
        self._running = False
        self._infer_thread.join(timeout=2)
