# 🚨 DISASTER INTEL — AI Disaster Damage Mapping System
### CodeReCET 2026 Hackathon Project

Real-time drone/video feed analysis with AI-powered damage detection and impact heatmap visualization.

---

## 🏗️ Architecture

```
Video/Stream → YOLOv8 Detection → Grid Impact Scorer → FastAPI WebSocket → React Dashboard
```

---

## 📁 Project Structure

```
disaster-intel/
├── backend/
│   ├── main.py          ← FastAPI server + WebSocket endpoints
│   ├── detector.py      ← YOLOv8 inference engine
│   ├── scorer.py        ← Grid-based impact scoring
│   ├── streamer.py      ← Video frame streaming + overlay
│   └── requirements.txt
├── frontend/
│   └── index.html       ← Full dashboard (no build needed)
├── data/
│   └── uploads/         ← Drop your video files here
├── setup.bat            ← One-click Windows setup
└── start_backend.bat    ← Start the server
```

---

## 🚀 Quick Start (Windows)

### Step 1: Open in Cursor
Open the `disaster-intel` folder in Cursor

### Step 2: Setup (run once)
```
Double-click: setup.bat
```
This creates a virtual environment and installs all dependencies.

### Step 3: Start backend
```
Double-click: start_backend.bat
```
Or in Cursor terminal:
```bash
venv\Scripts\activate
cd backend
python main.py
```

### (Optional but recommended) Enable real Fire/Smoke detection
By default, the system uses a generic COCO model (great for people/objects) but **COCO does not have true "fire" or "smoke" classes**.
To enable accurate fire/smoke detection, download a dedicated model once:

```powershell
powershell -ExecutionPolicy Bypass -File .\download_fire_smoke_model.ps1
```

Then run `start_backend.bat` again. If `models\fire_smoke_best.pt` exists, the backend will auto-enable it.

### Step 4: Open the dashboard
```
Open: frontend/index.html in Chrome or Edge
```

### Step 5: Test it
- Click **▶ DEMO** to see a synthetic disaster demo immediately
- Or click **⬆ UPLOAD** to upload your own disaster video
- Watch the heatmap update in real time!

---

## 🎯 How to Get Disaster Videos for Demo

1. **YouTube** → Search: `drone footage earthquake site:youtube.com` → Download with yt-dlp
2. **NOAA** → https://www.nesdis.noaa.gov/imagery
3. **Copernicus EMS** → https://emergency.copernicus.eu/mapping/

---

## 🔥 Impact Scoring Formula

```
Cell Score ≈ Σ (detection_weight × confidence × area_factor × center_factor × 10) × temporal_decay

Weights:
  person in danger    → 3.5  (highest priority)
  fire / smoke        → 3.0
  collapsed building  → 2.5
  flood zone          → 2.0
  debris              → 1.5

Severity:
  75-100  → 🔴 CRITICAL  → Deploy rescue immediately
  50-74   → 🟠 SEVERE    → High priority response
  25-49   → 🟡 MODERATE  → Monitor and assess
  0-24    → 🟢 SAFE      → All clear
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Detection | YOLOv8n (Ultralytics) |
| Backend | FastAPI + WebSockets |
| Streaming | OpenCV + asyncio |
| Frontend | Vanilla JS + CSS |
| Heatmap | Canvas overlay |

---

## 📈 Extending for Hackathon (Bonus Points)

1. **Multi-source**: Add a second WebSocket stream panel in frontend
2. **Fine-tune**: Export YOLO model trained on AIDER disaster dataset
3. **Geo-mapping**: Add Leaflet.js map with GPS coordinates from drone metadata
4. **Report gen**: Add a "Generate Report" button that exports zone data as PDF

---

## 🐛 Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError` | Run `setup.bat` first |
| WebSocket error | Make sure `start_backend.bat` is running |
| Slow FPS | Video is processing on CPU — normal, demo mode is faster |
| No detections | Try a video with people/vehicles visible |

---

## 👥 Team
CodeReCET 2026 — Armada Hackathon
