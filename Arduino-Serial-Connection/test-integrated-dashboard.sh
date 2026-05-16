#!/bin/bash
# Test script for the wetlab and weather-station dashboard

echo "================================"
echo "Testing Minilab Dashboard"
echo "================================"
echo ""

# Test Arduino connections
echo "1. Testing Arduino Serial Ports..."
echo ""

echo "   Weather Station (ttyACM0):"
python3 -c "
import serial, time
try:
    ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
    time.sleep(0.5)
    for i in range(3):
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line and 'Date' not in line and 'SYSTEM' not in line:
                print('      ✓ Data:', line[:60] + '...')
                break
        time.sleep(0.5)
    ser.close()
    print('      ✓ Weather Station: OK')
except Exception as e:
    print('      ✗ Error:', e)
"
echo ""

echo "   Wetlab Nano (ttyACM1):"
python3 -c "
import serial, time
try:
    ser = serial.Serial('/dev/ttyACM1', 9600, timeout=1)
    time.sleep(0.5)
    for i in range(3):
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line and 'Seconds' not in line:
                print('      ✓ Data:', line[:60] + '...')
                break
        time.sleep(0.5)
    ser.close()
    print('      ✓ Wetlab Sensors: OK')
except Exception as e:
    print('      ✗ Error:', e)
"
echo ""

# Test Dashboard APIs
echo "2. Testing Dashboard APIs..."
echo ""

echo "   Wetlab Status:"
curl -s --max-time 3 http://localhost:5000/api/status | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print('      ✓ Status:', 'OK' if data.get('success') else 'FAIL')
    print('      - Ports:', data.get('available_ports', []))
except Exception as e:
    print('      ✗ Error:', e)
"
echo ""

echo "   Weather Station Status:"
curl -s --max-time 3 http://localhost:5000/api/weather/status | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print('      ✓ Status:', 'OK' if data.get('success') else 'FAIL')
    print('      - Connected:', data.get('connected', False))
except Exception as e:
    print('      ✗ Error:', e)
"
echo ""

echo "   Port Management:"
curl -s --max-time 3 http://localhost:5000/api/ports/list | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print('      ✓ Status:', 'OK' if data.get('success') else 'FAIL')
    print('      - Total Ports:', data.get('total_ports', 0))
    for port_info in data.get('ports', []):
        port = port_info['port']
        mq = '✓' if port_info['mq135_active'] else ' '
        wx = '✓' if port_info['weather_active'] else ' '
        print(f'      - {port}: Wetlab[{mq}] Weather[{wx}]')
except Exception as e:
    print('      ✗ Error:', e)
"
echo ""

# Test System Stats
echo "3. Testing System Monitor..."
curl -s --max-time 3 http://localhost:5000/api/system/stats | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if data.get('success'):
        print('      ✓ CPU Temp:', data.get('cpu_temp'), '°C')
        print('      ✓ RAM:', data.get('ram_used'), '/', data.get('ram_total'), 'GB')
    else:
        print('      ✗ System stats unavailable')
except Exception as e:
    print('      ✗ Error:', e)
"
echo ""

echo "================================"
echo "Dashboard URL: http://localhost:5000"
echo "Network URL: http://$(hostname -I | awk '{print $1}'):5000"
echo "================================"
echo ""
echo "Available API Endpoints:"
echo "  Wetlab:"
echo "    POST /api/live/start"
echo "    POST /api/live/stop"
echo "    GET  /api/live/data"
echo ""
echo "  Weather Station:"
echo "    POST /api/weather/live/start"
echo "    POST /api/weather/live/stop"
echo "    GET  /api/weather/live/data"
echo "    POST /api/weather/record/start"
echo ""
echo "  Port Management:"
echo "    GET  /api/ports/list"
echo "    POST /api/ports/restart"
echo ""
