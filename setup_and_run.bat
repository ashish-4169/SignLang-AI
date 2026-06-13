@echo off
echo ==========================================
echo   SignLang AI - Setup (Windows)
echo ==========================================

echo.
echo [1/3] Installing dependencies...
pip install streamlit opencv-python mediapipe numpy Pillow torch torchvision matplotlib scikit-learn

echo.
echo [2/3] Creating folders...
mkdir models 2>nul
mkdir data 2>nul
mkdir static 2>nul

echo.
echo [3/3] Launching app...
echo.
echo   Open in browser: http://localhost:8501
echo.
streamlit run app.py
pause
