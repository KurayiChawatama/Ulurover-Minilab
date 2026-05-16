#!/bin/bash
# Wetlab Nano Data Collection Runner
# Uploads the wetlab Nano sketch and runs the serial logger for gas sensors

source ~/miniconda3/etc/profile.d/conda.sh
conda activate arduino-serial

# Auto-detect Arduino port
PORT=$(ls /dev/ttyACM* 2>/dev/null | head -n 1)
if [ -z "$PORT" ]; then
  PORT=$(ls /dev/ttyUSB* 2>/dev/null | head -n 1)
fi
if [ -z "$PORT" ]; then
  echo "No Arduino port found!"
  exit 1
fi

echo "=== Wetlab Nano Protocol ==="
echo "Arduino Nano: Connected to USB at $PORT"
echo ""

echo "Uploading sketch to Arduino at $PORT..."
arduino-cli compile --fqbn arduino:avr:nano wetlab_nano
if [ $? -ne 0 ]; then
  echo "Compilation failed!"
  exit 1
fi
arduino-cli upload -p "$PORT" --fqbn arduino:avr:nano wetlab_nano
if [ $? -ne 0 ]; then
  echo "Upload failed!"
  exit 1
fi

echo "Waiting for Arduino to reboot..."
sleep 3

echo ""
echo "Starting data collection..."
python /home/raspberrypi/Ulurover-Minilab/Arduino-Serial-Connection/dual-mq135/read-wetlab-csv.py --seconds 10
