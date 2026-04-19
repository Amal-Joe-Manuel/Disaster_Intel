import numpy as np
from collections import defaultdict

SEVERITY_LEVELS = {
    "CRITICAL": {"min": 75, "color": "#FF0000", "label": "🔴 CRITICAL", "action": "Deploy rescue immediately"},
    "SEVERE":   {"min": 50, "color": "#FF6600", "label": "🟠 SEVERE",   "action": "High priority response"},
    "MODERATE": {"min": 25, "color": "#FFCC00", "label": "🟡 MODERATE", "action": "Monitor and assess"},
    "SAFE":     {"min": 0,  "color": "#00CC44", "label": "🟢 SAFE",     "action": "All clear"},
}

class ImpactScorer:
    def __init__(self, grid_rows=8, grid_cols=8, decay_factor=0.85):
        """
        grid_rows/cols: how many cells to divide the frame into
        decay_factor: how much previous scores fade per frame (temporal smoothing)
        """
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.decay_factor = decay_factor

        # Persistent score grid — smoothed across frames
        self.score_grid = np.zeros((grid_rows, grid_cols), dtype=float)
        self.frame_count = 0
        self.detection_history = defaultdict(list)

    def update(self, frame_width: int, frame_height: int, detections: list[dict]) -> np.ndarray:
        """
        Update score grid with new detections from a frame.
        Returns normalized score grid (0-100).
        """
        self.frame_count += 1
        cell_w = frame_width / self.grid_cols
        cell_h = frame_height / self.grid_rows
        frame_area = max(1.0, float(frame_width * frame_height))

        # Decay existing scores
        self.score_grid *= self.decay_factor

        # Add new detection scores
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            weight = det["weight"]
            conf = det["confidence"]

            # Box area factor: larger impacted regions should contribute more.
            box_area = max(1.0, float((x2 - x1) * (y2 - y1)))
            area_ratio = min(1.0, box_area / frame_area)  # 0..1

            # Center weighting: small boost for detections closer to frame center.
            cx = (x1 + x2) * 0.5
            cy = (y1 + y2) * 0.5
            nx = (cx / max(1.0, frame_width)) * 2.0 - 1.0   # -1..1
            ny = (cy / max(1.0, frame_height)) * 2.0 - 1.0  # -1..1
            center_dist = (nx * nx + ny * ny) ** 0.5
            center_factor = 1.0 + max(0.0, 1.0 - min(center_dist, 1.0)) * 0.4  # 1.0..1.4

            # Final per-detection magnitude (scaled again below for grid overlap)
            magnitude = weight * conf * (0.5 + 1.5 * area_ratio) * center_factor

            # Find which grid cells the bounding box overlaps
            col_start = max(0, int(x1 / cell_w))
            col_end = min(self.grid_cols - 1, int(x2 / cell_w))
            row_start = max(0, int(y1 / cell_h))
            row_end = min(self.grid_rows - 1, int(y2 / cell_h))

            for r in range(row_start, row_end + 1):
                for c in range(col_start, col_end + 1):
                    self.score_grid[r][c] += magnitude * 10.0

        # Clip and normalize to 0-100
        self.score_grid = np.clip(self.score_grid, 0, 100)
        return self.score_grid.copy()

    def get_severity(self, score: float) -> dict:
        """Return severity level dict for a given score"""
        for level, info in SEVERITY_LEVELS.items():
            if score >= info["min"]:
                return {"level": level, **info}
        return {"level": "SAFE", **SEVERITY_LEVELS["SAFE"]}

    def get_top_zones(self, score_grid: np.ndarray, top_n=5) -> list[dict]:
        """Return top N most impacted grid zones"""
        flat = score_grid.flatten()
        top_indices = np.argsort(flat)[::-1][:top_n]
        zones = []
        for idx in top_indices:
            row = idx // self.grid_cols
            col = idx % self.grid_cols
            score = float(flat[idx])
            if score < 1.0:
                continue
            severity = self.get_severity(score)
            zones.append({
                "zone": f"{chr(65 + row)}{col + 1}",  # e.g. A1, B3
                "row": int(row),
                "col": int(col),
                "score": round(score, 1),
                "severity": severity["level"],
                "color": severity["color"],
                "label": severity["label"],
                "action": severity["action"],
            })
        return zones

    def get_grid_data(self, score_grid: np.ndarray) -> list[list[dict]]:
        """Return full grid as 2D array of cell data for frontend"""
        grid_data = []
        for r in range(self.grid_rows):
            row_data = []
            for c in range(self.grid_cols):
                score = float(score_grid[r][c])
                severity = self.get_severity(score)
                row_data.append({
                    "row": r,
                    "col": c,
                    "score": round(score, 1),
                    "color": severity["color"],
                    "level": severity["level"],
                })
            grid_data.append(row_data)
        return grid_data

    def get_stats(self, score_grid: np.ndarray) -> dict:
        """Return summary statistics"""
        flat = score_grid.flatten()
        critical = int(np.sum(flat >= 75))
        severe = int(np.sum((flat >= 50) & (flat < 75)))
        moderate = int(np.sum((flat >= 25) & (flat < 50)))
        safe = int(np.sum(flat < 25))
        return {
            "critical_zones": critical,
            "severe_zones": severe,
            "moderate_zones": moderate,
            "safe_zones": safe,
            "max_score": round(float(np.max(flat)), 1),
            "avg_score": round(float(np.mean(flat)), 1),
            "frame_count": self.frame_count,
        }

    def get_category_breakdown(self, detections: list[dict]) -> dict:
        """
        Returns count of detections per disaster category.
        Useful for the dashboard to show breakdown by type.
        Categories: structural, infrastructure, casualty, flood, debris, severity
        """
        breakdown = {
            "structural": 0,
            "infrastructure": 0,
            "casualty": 0,
            "flood": 0,
            "debris": 0,
            "severity": 0,
        }
        for det in detections:
            cat = det.get("category", "debris")
            if cat in breakdown:
                breakdown[cat] += 1
        return breakdown

    def reset(self):
        """Reset scorer for new video"""
        self.score_grid = np.zeros((self.grid_rows, self.grid_cols), dtype=float)
        self.frame_count = 0
