@echo off
echo ==========================================
echo   SignLang AI - Setup (Windows)
echo ==========================================

echo.
echo [1/2] Installing dependencies...
pip install -r requirements.txt

echo.
echo [2/2] Launching app...
echo.
echo   Open in browser: http://localhost:8501
echo.
streamlit run app.py
pause
