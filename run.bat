@echo off
chcp 65001 >nul
title SegVision QA Server (BDD100K 19-Classes)

echo ==============================================================
echo   SegVision QA - Khoi Dong Nhanh Web App
echo ==============================================================

cd /d "%~dp0"

IF EXIST ".venv\Scripts\activate.bat" (
    echo [Moi truong] Dang kich hoat moi truong ao .venv...
    call .venv\Scripts\activate.bat
) ELSE IF EXIST "venv\Scripts\activate.bat" (
    echo [Moi truong] Dang kich hoat moi truong ao venv...
    call venv\Scripts\activate.bat
)

python run.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [Loi] Khong the khoi chay. Co the ban chua cai thu vien:
    echo        pip install -r requirements.txt
    pause
)
