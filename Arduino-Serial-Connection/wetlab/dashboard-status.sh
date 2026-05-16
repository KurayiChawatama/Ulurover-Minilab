#!/bin/bash
# Check Minilab Dashboard status

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PID_FILE="$SCRIPT_DIR/dashboard.pid"
LOG_FILE="$SCRIPT_DIR/dashboard.log"

echo "=== Dashboard Status ==="
echo ""

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "✅ Running (PID: $PID)"
        
        # Get local IP
        IP=$(hostname -I | awk '{print $1}')
        
        echo ""
        echo "📡 Access at:"
        echo "   - Local:   http://localhost:5000"
        echo "   - Network: http://$IP:5000"
        
        # Check if port is actually listening
        if lsof -i :5000 > /dev/null 2>&1; then
            echo ""
            echo "🔌 Port 5000: Listening"
        else
            echo ""
            echo "⚠️  Port 5000: Not listening (server may be starting up)"
        fi
        
        # Show last few log lines
        if [ -f "$LOG_FILE" ]; then
            echo ""
            echo "📝 Recent logs:"
            tail -5 "$LOG_FILE" | sed 's/^/   /'
        fi
    else
        echo "❌ Not running (stale PID file)"
        rm -f "$PID_FILE"
    fi
else
    echo "❌ Not running"
    
    # Check if something else is on port 5000
    if lsof -i :5000 > /dev/null 2>&1; then
        echo ""
        echo "⚠️  Warning: Port 5000 in use by another process:"
        lsof -i :5000 | sed 's/^/   /'
    fi
fi

echo ""
echo "Commands:"
echo "  Start:  ./start-dashboard-bg.sh"
echo "  Stop:   ./stop-dashboard.sh"
echo "  Logs:   tail -f $LOG_FILE"
