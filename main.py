import asyncio
import math
import os
import shutil
import time
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from detector import DisasterDetector
from scorer import ImpactScorer
from streamer import VideoStreamer

app = FastAPI(title="Disaster Intel API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

detector: Optional[DisasterDetector] = None
active_streamers: dict[str, VideoStreamer] = {}
UPLOAD_DIR = Path("../data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.on_event("startup")
async def startup():
    global detector
    print("[Server] Loading YOLO model...")
    detector = DisasterDetector(model_path=None)
    print("[Server] Ready!")


@app.on_event("shutdown")
async def shutdown():
    if detector:
        detector.shutdown()


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": detector is not None,
        "inference_ms": round(detector.last_inference_ms, 1) if detector else 0,
    }


@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    dest = UPLOAD_DIR / file.filename
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"filename": file.filename, "path": str(dest)}


@app.get("/videos")
async def list_videos():
    videos = []
    for ext in ["*.mp4", "*.avi", "*.mov", "*.mkv"]:
        videos.extend([p.name for p in UPLOAD_DIR.glob(ext)])
    return {"videos": videos}


# ── Severity Timeline ─────────────────────────────────────────
class SeverityTimeline:
    def __init__(self, max_points=60):
        self.history   = deque(maxlen=max_points)
        self.start_time = time.time()

    def record(self, score_grid: np.ndarray):
        elapsed = round(time.time() - self.start_time, 1)
        self.history.append({
            "t":   elapsed,
            "max": round(float(np.max(score_grid)), 1),
            "avg": round(float(np.mean(score_grid)), 1),
        })

    def get(self) -> list:
        return list(self.history)


# ── Spread Predictor ──────────────────────────────────────────
class SpreadPredictor:
    def predict_spread(self, score_grid: np.ndarray, detections: list) -> list:
        spread = []
        flood_dets = [d for d in detections if d.get("category") == "flood"]
        fire_dets  = [d for d in detections if d.get("disaster_type") in ("fire","Damaged_buildings")]
        for det in flood_dets[:2]:
            x1,y1,x2,y2 = det["bbox"]
            spread.append({"type":"flood_spread","cx":(x1+x2)//2,"cy":(y1+y2)//2+40,"color":"#FF8800","label":"FLOOD ADVANCE"})
        for det in fire_dets[:2]:
            x1,y1,x2,y2 = det["bbox"]
            spread.append({"type":"fire_spread","cx":(x1+x2)//2+30,"cy":(y1+y2)//2-20,"color":"#FF2244","label":"FIRE ADVANCE"})
        return spread[:4]


# ── SITREP Generator ──────────────────────────────────────────
class SituationReportGenerator:
    def __init__(self, interval=30):
        self.interval       = interval
        self.last_generated = 0
        self.last_report    = "SITREP: Initializing sensors..."

    def should_update(self): return (time.time() - self.last_generated) >= self.interval

    def generate(self, stats, top_zones, detections, frame_count) -> str:
        self.last_generated = time.time()
        now      = time.strftime("%H:%M:%S")
        critical = stats.get("critical_zones", 0)
        severe   = stats.get("severe_zones", 0)
        max_score = stats.get("max_score", 0)
        cats = {"structural":0,"flood":0,"casualty":0,"debris":0}
        for d in detections:
            c = d.get("category","debris")
            if c in cats: cats[c] += 1
        parts = [f"SITREP {now} —"]
        if critical > 0:   parts.append(f"{critical} CRITICAL zone{'s' if critical>1 else ''} active.")
        elif severe > 0:   parts.append(f"Situation SEVERE. {severe} zones require response.")
        elif max_score>10: parts.append("Moderate damage detected. Monitoring.")
        else:              parts.append("No significant threats detected.")
        if top_zones:
            z = top_zones[0]
            parts.append(f"Zone {z['zone']} highest priority (score {z['score']}).")
        if cats["casualty"]>0:  parts.append(f"⚠ {cats['casualty']} casualty detection(s) — prioritize medical.")
        if cats["flood"]>0:     parts.append("Flooding active — evacuation routes may be compromised.")
        if cats["structural"]>0:parts.append(f"{cats['structural']} structural collapse indicator(s) detected.")
        if critical>0:   parts.append("RECOMMEND: Deploy rescue to critical zones immediately.")
        elif severe>0:   parts.append("RECOMMEND: Stage response teams at perimeter.")
        else:            parts.append("RECOMMEND: Continue aerial surveillance.")
        self.last_report = " ".join(parts)
        return self.last_report

    def get_last(self): return self.last_report


# ── Helper ────────────────────────────────────────────────────
def _draw_spread_arrows(frame, spread_zones):
    out = frame.copy()
    for sz in spread_zones:
        cx, cy = sz.get("cx",0), sz.get("cy",0)
        h = sz.get("color","#FF8800").lstrip("#")
        r,g,b = int(h[0:2],16),int(h[2:4],16),int(h[4:6],16)
        bgr = (b,g,r)
        cv2.circle(out,(cx,cy),18,bgr,2)
        cv2.circle(out,(cx,cy),8,bgr,-1)
        cv2.putText(out,sz.get("label","SPREAD"),(cx-30,cy-22),cv2.FONT_HERSHEY_SIMPLEX,0.4,bgr,1)
    return out


def _build_payload(frame_b64, grid_data, top_zones, stats, timeline,
                   spread_zones, sitrep, detections, model_type, extra=None):
    p = {
        "type":        "frame",
        "frame":       frame_b64,
        "grid":        grid_data,
        "zones":       top_zones,
        "stats":       stats,
        "timeline":    timeline,
        "spread_zones":spread_zones,
        "sitrep":      sitrep,
        "detections": [
            {"type": d["disaster_type"], "confidence": round(d["confidence"],2),
             "bbox": d.get("bbox",[]), "category": d.get("category","debris")}
            for d in detections
        ],
        "model_type":  model_type,
    }
    if extra:
        p.update(extra)
    return p


# ── Single stream ─────────────────────────────────────────────
@app.websocket("/ws/stream/{stream_id}")
async def stream_video(websocket: WebSocket, stream_id: str):
    await websocket.accept()
    scorer   = ImpactScorer(grid_rows=8, grid_cols=8)
    # ── PERFORMANCE: target_fps=20, adaptive skip handles the rest
    streamer = VideoStreamer(target_fps=20)
    timeline = SeverityTimeline()
    spreader = SpreadPredictor()
    sitrep   = SituationReportGenerator(interval=30)

    try:
        config = await websocket.receive_json()
        source = config.get("source","")
        if not source.startswith("rtsp://") and not source.startswith("http"):
            sp = str(UPLOAD_DIR / source)
            if not os.path.exists(sp):
                await websocket.send_json({"error": f"File not found: {source}"}); return
            source = sp

        if not streamer.open(source):
            await websocket.send_json({"error": "Could not open video source"}); return

        info = streamer.get_video_info()
        await websocket.send_json({"type":"video_info","data":info})
        active_streamers[stream_id] = streamer
        W, H = info["width"], info["height"]

        # ── Pass detector so streamer can do adaptive skip ────
        async for frame in streamer.frames(detector=detector):
            detections = detector.detect(frame)
            fW, fH    = frame.shape[1], frame.shape[0]
            score_grid = scorer.update(fW, fH, detections)
            grid_data  = scorer.get_grid_data(score_grid)
            top_zones  = scorer.get_top_zones(score_grid, top_n=6)
            stats      = scorer.get_stats(score_grid)
            timeline.record(score_grid)
            spread_zones = spreader.predict_spread(score_grid, detections)
            if sitrep.should_update():
                sitrep.generate(stats, top_zones, detections, stats["frame_count"])
            annotated = streamer.draw_grid_overlay(frame, grid_data, alpha=0.35)
            annotated = detector.annotate_frame(annotated, detections)
            annotated = _draw_spread_arrows(annotated, spread_zones)
            # ── PERFORMANCE: quality=55 for fast transmission ──
            frame_b64 = VideoStreamer.frame_to_base64(annotated, quality=55)
            payload   = _build_payload(
                frame_b64, grid_data, top_zones, stats, timeline.get(),
                spread_zones, sitrep.get_last(), detections,
                "custom" if detector.using_custom_model else "coco_fallback",
                {"inference_ms": round(detector.last_inference_ms, 1)},
            )
            await websocket.send_json(payload)

    except WebSocketDisconnect:
        print(f"[WS] Client {stream_id} disconnected")
    except Exception as e:
        print(f"[WS] Error: {e}")
        try: await websocket.send_json({"error": str(e)})
        except: pass
    finally:
        streamer.stop()
        active_streamers.pop(stream_id, None)


# ── Multi-source stream ───────────────────────────────────────
@app.websocket("/ws/multi/{stream_id}")
async def multi_stream(websocket: WebSocket, stream_id: str):
    await websocket.accept()
    scorer   = ImpactScorer(grid_rows=8, grid_cols=8)
    timeline = SeverityTimeline()
    spreader = SpreadPredictor()
    sitrep   = SituationReportGenerator(interval=30)
    streamer1 = VideoStreamer(target_fps=20)
    streamer2 = VideoStreamer(target_fps=20)

    try:
        config  = await websocket.receive_json()
        def resolve(s):
            if not s.startswith("rtsp://") and not s.startswith("http"):
                p = str(UPLOAD_DIR / s)
                return p if os.path.exists(p) else None
            return s

        src1 = resolve(config.get("source1",""))
        src2 = resolve(config.get("source2",""))
        if not src1: await websocket.send_json({"error":"Source 1 not found"}); return
        if not src2: await websocket.send_json({"error":"Source 2 not found"}); return
        if not streamer1.open(src1): await websocket.send_json({"error":"Cannot open source 1"}); return
        if not streamer2.open(src2): await websocket.send_json({"error":"Cannot open source 2"}); return

        info1 = streamer1.get_video_info()
        await websocket.send_json({"type":"video_info","data":info1})

        gen2 = streamer2.frames(detector=detector)
        frame_count = 0

        async for frame1 in streamer1.frames(detector=detector):
            frame_count += 1
            try:
                frame2 = await asyncio.wait_for(gen2.__anext__(), timeout=0.08)
            except (StopAsyncIteration, asyncio.TimeoutError):
                frame2 = np.zeros_like(frame1)

            dets1 = detector.detect(frame1)
            dets2 = detector.detect(frame2)
            fW, fH = frame1.shape[1], frame1.shape[0]
            score_grid = scorer.update(fW, fH, dets1)
            score_grid = scorer.update(fW, fH, dets2)
            grid_data  = scorer.get_grid_data(score_grid)
            top_zones  = scorer.get_top_zones(score_grid, top_n=6)
            stats      = scorer.get_stats(score_grid)
            stats["frame_count"] = frame_count
            timeline.record(score_grid)
            all_dets     = dets1 + dets2
            spread_zones = spreader.predict_spread(score_grid, all_dets)
            if sitrep.should_update():
                sitrep.generate(stats, top_zones, all_dets, frame_count)

            ann1 = streamer1.draw_grid_overlay(frame1, grid_data, alpha=0.35)
            ann1 = detector.annotate_frame(ann1, dets1)
            ann2 = streamer2.draw_grid_overlay(frame2, grid_data, alpha=0.35)
            ann2 = detector.annotate_frame(ann2, dets2)
            if ann2.shape[0] != ann1.shape[0]:
                scale = ann1.shape[0] / ann2.shape[0]
                ann2  = cv2.resize(ann2, (int(ann2.shape[1]*scale), ann1.shape[0]))
            combined  = np.hstack([ann1, ann2])
            frame_b64 = VideoStreamer.frame_to_base64(combined, quality=50)
            payload   = _build_payload(
                frame_b64, grid_data, top_zones, stats, timeline.get(),
                spread_zones, sitrep.get_last(), all_dets,
                "custom" if detector.using_custom_model else "coco_fallback",
                {"multi_source": True, "source_count": 2,
                 "inference_ms": round(detector.last_inference_ms,1)},
            )
            await websocket.send_json(payload)

    except WebSocketDisconnect: pass
    except Exception as e:
        print(f"[Multi WS] Error: {e}")
        try: await websocket.send_json({"error": str(e)})
        except: pass
    finally:
        streamer1.stop(); streamer2.stop()


# ── Demo mode ─────────────────────────────────────────────────
@app.websocket("/ws/demo/{stream_id}")
async def demo_stream(websocket: WebSocket, stream_id: str):
    await websocket.accept()
    scorer   = ImpactScorer(grid_rows=8, grid_cols=8)
    timeline = SeverityTimeline()
    spreader = SpreadPredictor()
    sitrep   = SituationReportGenerator(interval=15)
    frame_count = 0

    try:
        while True:
            frame_count += 1
            W, H  = 640, 480
            frame = np.zeros((H, W, 3), dtype=np.uint8)
            frame[:] = (20, 20, 30)
            t = frame_count * 0.05
            fire_zones  = [(int(W*0.2+30*math.sin(t)), int(H*0.3), 80, 60),
                           (int(W*0.7+20*math.cos(t*1.3)), int(H*0.5), 60, 50)]
            flood_zones = [(int(W*0.1), int(H*0.7), 150, 80)]
            fake = []
            for (x,y,w,h) in fire_zones:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,80,200),-1)
                fake.append({"disaster_type":"fire","confidence":0.85+0.1*math.sin(t),
                             "bbox":[x,y,x+w,y+h],"weight":3.0,"color":(0,0,255),"category":"structural"})
            for (x,y,w,h) in flood_zones:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(180,100,0),-1)
                fake.append({"disaster_type":"flood","confidence":0.75,
                             "bbox":[x,y,x+w,y+h],"weight":2.0,"color":(255,165,0),"category":"flood"})
            score_grid   = scorer.update(W, H, fake)
            grid_data    = scorer.get_grid_data(score_grid)
            top_zones    = scorer.get_top_zones(score_grid, top_n=6)
            stats        = scorer.get_stats(score_grid)
            stats["frame_count"] = frame_count
            timeline.record(score_grid)
            spread_zones = spreader.predict_spread(score_grid, fake)
            if sitrep.should_update():
                sitrep.generate(stats, top_zones, fake, frame_count)
            annotated = VideoStreamer.draw_grid_overlay(frame, grid_data, alpha=0.4)
            annotated = _draw_spread_arrows(annotated, spread_zones)
            frame_b64 = VideoStreamer.frame_to_base64(annotated, quality=70)
            payload   = _build_payload(
                frame_b64, grid_data, top_zones, stats, timeline.get(),
                spread_zones, sitrep.get_last(), fake, "demo",
            )
            await websocket.send_json(payload)
            await asyncio.sleep(0.05)  # demo runs at ~20fps

    except WebSocketDisconnect: pass
    except Exception as e: print(f"[Demo WS] Error: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
