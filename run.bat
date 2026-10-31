@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
title Multimodal Attendance and Engagement Agent
echo =====================================================================
echo       MULTIMODAL ATTENDANCE AND ENGAGEMENT AGENT (GenAI)
echo =====================================================================
echo.

set PYTHON_PATH="C:\Users\ayush\AppData\Local\Programs\Python\Python311\python.exe"

if exist %PYTHON_PATH% (
    echo [OK] Found Python 3.11 at %PYTHON_PATH%
    %PYTHON_PATH% -m streamlit run app.py
) else (
    echo [INFO] Searching for default python in PATH...
    python -m streamlit run app.py
)

pause
