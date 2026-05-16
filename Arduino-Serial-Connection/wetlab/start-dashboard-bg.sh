#!/bin/bash
# Start Minilab Dashboard in the background

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PID_FILE="$SCRIPT_DIR/dashboard.pid"
LOG_FILE="$SCRIPT_DIR/dashboard.log"

echo "=== Minilab Dashboard ==="

# Check if already running
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "⚠️  Dashboard is already running (PID: $OLD_PID)"
        echo "   Use ./stop-dashboard.sh to stop it first"
        exit 1
    else
        rm -f "$PID_FILE"
    fi
fi

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate arduino-serial

# Navigate to dashboard directory
cd "$SCRIPT_DIR/dashboard"

# Get local IP address
IP=$(hostname -I | awk '{print $1}')

echo "Starting web interface in background..."
echo ""

# Start Flask app in background
nohup python3 app.py > "$LOG_FILE" 2>&1 &
DASHBOARD_PID=$!

# Save PID
echo $DASHBOARD_PID > "$PID_FILE"

# Wait a moment to check if it started
sleep 2

if ps -p $DASHBOARD_PID > /dev/null 2>&1; then
    echo "✅ Dashboard started successfully!"
    echo "   PID: $DASHBOARD_PID"
    echo ""
    echo "📡 Access the dashboard at:"
    echo "   - Local:   http://localhost:5000"
    echo "   - Network: http://$IP:5000"
    echo ""
    echo "📝 Logs: tail -f $LOG_FILE"
    echo "🛑 Stop:  ./stop-dashboard.sh"
else
    echo "❌ Failed to start dashboard"
    echo "   Check logs: cat $LOG_FILE"
    rm -f "$PID_FILE"
    exit 1
fi
