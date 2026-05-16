# Minilab Dashboard

Browser-based interface for controlling and visualizing the wetlab MQ-135 sensors and the separate weather station.

## Features

- 🎛️ **Web Control Panel**: Run sensor data collection with customizable parameters
- 📊 **Real-time Visualization**: Interactive graphs of CO₂ PPM readings  
- 🔴 **Live Monitoring**: Watch reaction rates in real-time with auto-updating graphs
- 📁 **File Management**: View and plot historical data files
- 🔌 **Auto-detection**: Automatically finds Arduino USB port
- 🌐 **Network Access**: Access from any device on your local network
- 📷 **Camera Integration**: Capture visual documentation with Camera NoIR V2
  - Live camera stream preview
  - Photo capture for experiment documentation
  - Timed video recording (1-300 seconds)
  - Media gallery with download/delete options
- 💻 **System Monitoring**: Real-time Raspberry Pi stats
  - CPU temperature with color warnings
  - Per-core CPU usage visualization
  - RAM and Swap usage monitoring
  - Top processes by CPU usage

## Dashboard Layout (Optimized for 24-27" monitors)

The dashboard uses a **3-column layout** at the top with a full-width graph at the bottom:

```
┌────────────────┬────────────────┬────────────────┐
│  Gas Sensors   │    Camera      │ System Monitor │
│  - Collection  │  - Stream      │ - CPU Temp     │
│  - Live Feed   │  - Photo       │ - CPU Cores    │
│  - File List   │  - Video       │ - RAM/Swap     │
│                │  - Gallery     │ - Processes    │
├────────────────┴────────────────┴────────────────┤
│        Gas Monitor Visualization (Graph)         │
│              (Full width, 380px tall)             │
└───────────────────────────────────────────────────┘
```

**Responsive Design:**
- ≥1600px: 3 columns (optimal for 24-27" monitors)
- 1000-1600px: 2 columns
- <1000px: single column stack

## Current Setup

**Configuration**: Wetlab Nano with 3 MQ-135 sensors plus a separate weather-station Nano
- **Wetlab Nano**: 3 sensors connected directly on A5, A4, A3
- **Weather Station**: Separate Arduino Nano with MQ gas sensors, BME280, VEML6070, DS3231, and SD logging

## Quick Start

1. **Upload the Wetlab Nano first:**
   ```bash
   cd /home/raspberrypi/Ulurover-Minilab/Ulurover-Minilab/Arduino-Serial-Connection/dual-mq135
   ./upload-arduino-b.sh
   ```

2. **Start the dashboard:**
   ```bash
   ./start-dashboard.sh
   ```

3. **Access the dashboard:**
   - Open your browser to `http://localhost:5000`
   - Or from another device: `http://<raspberry-pi-ip>:5000`

## Usage Modes

### 📊 Recorded Mode
- Set duration (5-3600 seconds)
- Click "Record Data Collection"
- Data saved as timestamped CSV
- Automatic graph display after collection

### 🔴 Live Monitor Mode  
- Click "Start Live Feed"
- Watch real-time sensor readings
- Graph updates every 0.5 seconds
- Keeps last 100 data points
- Perfect for monitoring reaction progress
- Click "Stop Live Feed" when done

## Parameters

- **Duration**: How long to record data (5-3600 seconds) - for recorded mode only
- **Wetlab Sensors**: Number of wetlab gas sensors (1-10, default: 3)

## Camera Controls (NEW)

### 📹 Live Stream
1. Click "Start Stream" to view live camera feed
2. Stream uses continuous still captures at ~2-3 fps
3. Lower framerate but more reliable than video streaming
4. Click "Stop Stream" when done

### 📸 Photo Capture
1. Start the camera stream first
2. Click "Capture Photo"
3. Photo saved as 1920x1080 JPEG
4. Appears in gallery automatically

### 🎥 Video Recording
1. Start the camera stream first
2. Set duration (1-300 seconds)
3. Click "Record Video"
4. Recording indicator shows while active
5. Video saved as H.264 (1920x1080 @ 30fps)
6. Appears in gallery when complete

### 📁 Media Gallery
- **Photos**: Click thumbnail to view full size
- **Videos**: Click to download (.h264 format)
- **Delete**: Hover and click × button
- **Refresh**: Updates gallery with new captures

**Media Storage**: `media/photos/` and `media/videos/` in dashboard directory

**Camera Used**: Raspberry Pi Camera NoIR V2 (NOIR = No Infrared filter)

**Commands Used**:
- Photo: `rpicam-still -n -o <file> -t 0` (immediate capture)
- Video: `rpicam-vid -n -o <file> -t <ms>` (timed recording)
- Stream: Continuous `rpicam-still` captures at 2-3fps

**Note**: Raspberry Pi OS Bookworm uses `rpicam-*` commands (not `libcamera-*`)

**Video Conversion**: Videos are saved in H.264 format. To convert to MP4:
```bash
ffmpeg -i video_TIMESTAMP.h264 -c:v copy output.mp4
```

## Data Format

CSV files contain:
```
Seconds,CO2_PPM_B1,CO2_PPM_B2
0,2.76,0.94
2,2.66,0.92
...
```

## Architecture

- **Backend**: Flask (Python) with pyserial for live monitoring
- **Frontend**: HTML/CSS/JavaScript
- **Graphing**: Plotly.js for interactive charts
- **Data Collection**: Real-time serial communication with Arduino

## Notes

- Wetlab Nano must be uploaded separately before use
- Live monitoring uses the wetlab controller already flashed on the board
- Live feed stops automatically when loading recorded data
- CSV files saved in dual-mq135 directory with timestamps

This is a simple prototype designed for your software team to expand upon.
