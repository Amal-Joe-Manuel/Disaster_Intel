@echo off
echo ============================================
echo  DISASTER INTEL - Setup Script (Windows)
echo ============================================
echo.

echo [1/3] Creating virtual environment...
python -m venv venv
call venv\Scripts\activate

echo.
echo [2/3] Installing Python dependencies...
pip install -r backend\requirements.txt

echo.
echo [3/3] Setup complete!
echo.
echo To run the project:
echo   1. Run:  start_backend.bat
echo   2. Open: frontend\index.html in your browser
echo.
pause
