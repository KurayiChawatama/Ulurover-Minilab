#!/bin/bash
# Minilab Dashboard Startup Script

echo "=== Minilab Dashboard ==="
echo "Starting web interface for MQ-135 sensor control..."
echo ""

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate arduino-serial

# Check if Flask is installed
if ! python -c "import flask" 2>/dev/null; then
    echo "Installing Flask..."
    pip install flask
fi

# Navigate to dashboard directory
cd "$(dirname "$0")/dashboard"

# Get local IP address
IP=$(hostname -I | awk '{print $1}')

echo ""
echo "✅ Dashboard is starting..."
echo "📡 Access the dashboard at:"
echo "   - Local:   http://localhost:5000"
echo "   - Network: http://$IP:5000"
echo ""
echo "Press Ctrl+C to stop the server"
echo "================================"
echo ""

# Start Flask app
python app.py
