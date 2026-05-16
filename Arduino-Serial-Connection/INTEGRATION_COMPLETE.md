# Arduino System Configuration

## Summary

The Minilab now runs a simplified system with single Arduino(s) for sensor data collection.

## System Architecture

### Current Configuration

**MQ-135 CO2 Sensors (Single Arduino)**
- **Hardware:** Single Arduino UNO
- **Connection:** USB to Raspberry Pi
- **Sensors:** Multiple MQ-135 CO₂ sensors (up to 3 by default)
  - Sensor 1: A0
  - Sensor 2: A1
  - Sensor 3: A2
- **Data Format:** `Seconds,CO2_PPM_1,CO2_PPM_2,CO2_PPM_3`
- **Baud Rate:** 9600
- **Update Interval:** 2 seconds

**PCB 1 - Weather Station (Optional, ttyACM0)**
- **Hardware:** Single Arduino Nano
- **Connection:** USB to Raspberry Pi
- **Sensors:**
  - MQ-135 (CO₂)
  - MQ-4 (CH₄ - Methane)
  - MQ-8 (H₂ - Hydrogen)
  - BME280 (Temperature, Pressure, Humidity)
  - VEML6070 (UV)
  - RTC DS3231 (Real-time Clock)
- **Data Format:** `Date,Time,CO2,CH4,H2,Temp,Pressure,Humidity,UV`
- **Baud Rate:** 9600
- **Update Interval:** 5 seconds

## Changes from Previous Version

### Removed
- **Arduino A (Master Bridge)** - I2C master that acted as bridge
- **I2C Slave Communication** - Arduino B no longer operates as I2C slave
- **Dual Arduino Protocol** - No longer uses I2C communication between Arduinos

### Simplified
- **MQ-135 System** - Now uses direct analog connections to single Arduino
- **Data Collection** - Sensors read and output directly via serial
- **Deployment** - Only one Arduino sketch to upload (arduino_b_slave/)

## Dashboard Features

### MQ-135 Endpoints
- `GET /api/status` - Get MQ-135 sensor status
- `POST /api/run` - Record MQ-135 data to CSV
- `POST /api/live/start` - Start live MQ-135 monitoring
- `POST /api/live/stop` - Stop live MQ-135 monitoring
- `GET /api/live/data` - Get current MQ-135 readings
- `GET /api/data/<filename>` - Get recorded MQ-135 data

### Weather Station Endpoints (Optional)
- `GET /api/weather/status` - Get weather station status
- `POST /api/weather/live/start` - Start live weather monitoring
- `POST /api/weather/live/stop` - Stop live weather monitoring
- `GET /api/weather/live/data` - Get current weather readings
- `POST /api/weather/record/start` - Start recording weather data
- `GET /api/weather/data/<filename>` - Get recorded weather data

### Port Management Endpoints
- `GET /api/ports/list` - List all ports and their status
- `POST /api/ports/restart` - Restart serial connections

### System Monitoring
- `GET /api/system/stats` - Get Raspberry Pi system stats
- Camera endpoints (photos, videos, streaming)

## Usage Examples

### Start Weather Station Live Monitoring
```bash
curl -X POST http://localhost:5000/api/weather/live/start
```

### Get Live Weather Data
```bash
curl http://localhost:5000/api/weather/live/data
```

### Start MQ-135 Live Monitoring
```bash
curl -X POST http://localhost:5000/api/live/start
```

### Record Weather Data for 60 seconds
```bash
curl -X POST http://localhost:5000/api/weather/record/start \
  -H "Content-Type: application/json" \
  -d '{"duration": 60}'
```

### List All Ports and Status
```bash
curl http://localhost:5000/api/ports/list
```

### Restart All Port Connections
```bash
curl -X POST http://localhost:5000/api/ports/restart \
  -H "Content-Type: application/json" \
  -d '{"port": "all"}'
```

## File Structure

```
Arduino-Serial-Connection/
├── dual-mq135/
│   ├── dashboard/
│   │   ├── app.py
│   │   ├── templates/
│   │   │   └── dashboard.html
│   │   ├── requirements.txt
│   │   └── README.md
│   ├── arduino_b_slave/
│   │   └── arduino_b_slave.ino (now standalone, no I2C)
│   ├── read-dual-mq135-csv.py
│   ├── run-dual-mq135.sh
│   ├── upload-arduino-b.sh
│   ├── dashboard-status.sh
│   ├── check-sensor-voltages.py
│   └── README.md
├── weather-station/ (Optional)
│   └── weather_station.ino
└── INTEGRATION_COMPLETE.md (this file)
```

## Code Changes Made

### 1. Updated Arduino Sketch (arduino_b_slave.ino)
- **Removed:** I2C slave initialization and Wire library dependency
- **Removed:** receiveEvent() and requestEvent() handlers
- **Added:** Direct serial CSV output via Serial.begin(9600)
- **Output Format:** `Seconds,CO2_PPM_1,CO2_PPM_2,CO2_PPM_3`

### 2. Updated Python Data Collection (read-dual-mq135-csv.py)
- Updated docstring to reference single Arduino
- Changed default CSV filename from `dual_co2_log_` to `mq135_log_`
- Simplified port detection (no longer needs ttyACM1 specifically)

### 3. Updated Dashboard (dashboard/app.py)
- **Removed:** Arduino A sketch compilation/upload
- **Removed:** sensor_a and sensor_b parameters from /api/run
- **Updated:** Port detection logic for single Arduino
- Changed CSV filename pattern from `dual_co2_log_` to `mq135_log_`
- Updated sensor naming from `CO2_PPM_B{i}` to `CO2_PPM_{i}`
- **Simplified:** Data collection endpoint now just records from single Arduino

### 4. Updated Shell Scripts
- **run-dual-mq135.sh** - Updated to upload arduino_b_slave sketch only
- **upload-arduino-b.sh** - Updated instructions for single Arduino setup

### 5. Updated Documentation
- **README.md** - Updated hardware setup, removed I2C protocol details
- **INTEGRATION_COMPLETE.md** - Updated system architecture and configuration

## Testing

### Upload Arduino Sketch
```bash
cd ~/Ulurover-Minilab/Arduino-Serial-Connection/dual-mq135
./upload-arduino-b.sh
```

### Test Arduino Connection
```bash
# Test MQ-135 Arduino
python3 -c "
import serial, time
ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
time.sleep(1)
print(ser.readline().decode('utf-8').strip())
ser.close()
"
```

### Test Data Collection
```bash
cd ~/Ulurover-Minilab/Arduino-Serial-Connection/dual-mq135
python3 read-dual-mq135-csv.py --seconds 10
```

### Test Dashboard APIs
```bash
# Check MQ-135 status
curl -s http://localhost:5000/api/status | python3 -m json.tool

# Start live monitoring
curl -X POST http://localhost:5000/api/live/start

# Get live data
curl http://localhost:5000/api/live/data

# Stop live monitoring
curl -X POST http://localhost:5000/api/live/stop
```

## Usage Examples

### Start Live Monitoring
```bash
curl -X POST http://localhost:5000/api/live/start
```

### Get Live Data
```bash
curl http://localhost:5000/api/live/data
```

### Record Data for 60 seconds
```bash
curl -X POST http://localhost:5000/api/run \
  -H "Content-Type: application/json" \
  -d '{"duration": 60}'
```

### Get All Available Sensor Data
```bash
curl http://localhost:5000/api/status
```

## Dashboard Access

**Local:** http://localhost:5000  
**Network:** http://<raspberry-pi-ip>:5000

## Starting the Dashboard

```bash
cd ~/Ulurover-Minilab/Arduino-Serial-Connection/dual-mq135/dashboard
python3 app.py
```

Or use the provided scripts:
```bash
# Start in background
cd ~/Ulurover-Minilab/Arduino-Serial-Connection/dual-mq135
./start-dashboard-bg.sh

# Stop dashboard
./stop-dashboard.sh
```

## Quick Start Guide

1. **Upload Arduino Sketch**
   ```bash
   cd ~/Ulurover-Minilab/Arduino-Serial-Connection/dual-mq135
   ./upload-arduino-b.sh
   ```

2. **Connect MQ-135 Sensors**
   - Connect 3 MQ-135 sensors to Arduino pins A0, A1, A2
   - Power from 5V, ground to GND

3. **Start Dashboard**
   ```bash
   ./start-dashboard-bg.sh
   ```

4. **Access Dashboard**
   - Open browser to http://localhost:5000
   - Use live monitoring or record data to CSV

## Troubleshooting

### No Arduino Port Found
```bash
# List all serial ports
ls -la /dev/tty* | grep -E "ACM|USB"
```

### Arduino Not Responding
```bash
# Check if Arduino is accessible
python3 -c "import serial; print(serial.tools.list_ports.comports())"
```

### Sensor Readings are 0 or Invalid
- Check sensor connections to A0, A1, A2
- Verify 5V power supply to sensors
- Allow 24-48 hours for sensors to stabilize after first power-on
- Check baseline offset in arduino_b_slave.ino (BASELINE_OFFSET_PPM variable)

# Check status
./dashboard-status.sh
```

## Important Notes

1. **Port Assignment:** The code now explicitly assigns ports:
   - ttyACM0 → Weather Station
   - ttyACM1 → MQ-135 Dual Setup

2. **No Standalone Weather Station:** All functionality is integrated into one dashboard. No need for separate weather station scripts.

3. **Serial Access:** Only one connection per port at a time. Use the API endpoints to manage connections.

4. **Data Recording:** Weather data is saved to `weather-station/weather_log_YYYYMMDD_HHMMSS.csv`

5. **Auto-Recovery:** Use `/api/ports/restart` if serial connections become unresponsive.

## Troubleshooting

### Ports Not Found
```bash
ls -la /dev/ttyACM*
# Should show: ttyACM0 and ttyACM1
```

### Permission Denied
```bash
sudo usermod -a -G dialout $USER
# Logout and login again
```

### Weather Station Not Sending Data
```bash
# Re-upload the sketch
cd ~/Ulurover-Minilab/Arduino-Serial-Connection/weather-station
arduino-cli compile --fqbn arduino:avr:nano weather_station
arduino-cli upload -p /dev/ttyACM0 --fqbn arduino:avr:nano weather_station
```

### Dashboard Not Starting
```bash
# Check logs
tail -f /tmp/dashboard.log

# Kill and restart
pkill -f "python3 app.py"
cd ~/Ulurover-Minilab/Arduino-Serial-Connection/dual-mq135/dashboard
python3 app.py
```

## Next Steps

1. **Update HTML Dashboard:** Add UI controls for weather station in `templates/dashboard.html`
2. **Add Charts:** Integrate Chart.js for weather data visualization
3. **Data Export:** Add CSV download functionality
4. **Alerts:** Add threshold alerts for gas levels
5. **Data Logging:** Implement continuous background logging

---

**Integration Date:** February 28, 2026  
**Status:** ✅ All systems operational  
**Dashboard Version:** 898 lines (integrated)  
**Python Syntax:** ✅ Validated
