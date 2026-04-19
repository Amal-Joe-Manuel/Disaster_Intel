@echo off
setlocal EnableDelayedExpansion
echo ============================================
echo  DISASTER INTEL - Starting Backend Server
echo ============================================
echo.
call venv\Scripts\activate
cd backend
if exist "..\models\disaster_model.pt" (
  echo Disaster model detected: ..\models\disaster_model.pt
) else (
  echo No custom disaster model found. Using YOLOv8s COCO fallback.
)
echo Server starting at http://localhost:8000
echo.
echo IMPORTANT: Open frontend\index.html in your browser
echo Then click [DEMO] to test without a video file
echo Or upload your own disaster video
echo.
python main.py
pause
