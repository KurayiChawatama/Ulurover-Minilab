#!/bin/bash
# Stop Minilab Dashboard

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PID_FILE="$SCRIPT_DIR/dashboard.pid"

if [ ! -f "$PID_FILE" ]; then
    echo "⚠️  Dashboard is not running (no PID file found)"
    exit 0
fi

PID=$(cat "$PID_FILE")

if ps -p "$PID" > /dev/null 2>&1; then
    echo "Stopping dashboard (PID: $PID)..."
    kill "$PID"
    sleep 1
    
    # Force kill if still running
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "Force killing dashboard..."
        kill -9 "$PID"
    fi
    
    rm -f "$PID_FILE"
    echo "✅ Dashboard stopped"
else
    echo "⚠️  Dashboard process not found (was PID: $PID)"
    rm -f "$PID_FILE"
fi
