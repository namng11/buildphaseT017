#!/usr/bin/env bash
# SegVision QA - Quick Startup Script for macOS / Linux

cd "$(dirname "$0")"

if [ -f ".venv/bin/activate" ]; then
    echo "[Environment] Activating .venv virtual environment..."
    source .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
    echo "[Environment] Activating venv virtual environment..."
    source venv/bin/activate
fi

python3 run.py
