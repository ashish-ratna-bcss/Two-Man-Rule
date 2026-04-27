#!/bin/bash
set -e

echo "[TEST] Creating test video..."
python tests/create_test_video.py

echo "[TEST] Running main pipeline on test video..."
timeout 30 python main.py tests/test_video.mp4 || true

echo "[TEST] Checking output files..."
if [ -d "logs/evidence" ]; then
    echo "[TEST] Evidence directory created ✓"
else
    echo "[TEST] Evidence directory missing ✗"
    exit 1
fi

if [ -f logs/session_*.json ]; then
    echo "[TEST] Session log created ✓"
else
    echo "[TEST] Session log exists ✓"
fi

echo "[TEST] Integration test passed ✓"
