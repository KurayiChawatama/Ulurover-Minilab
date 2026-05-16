#!/bin/bash
# Single Arduino MQ-135 Data Collection Runner
# Uploads Arduino sketch with multiple MQ-135 sensors and runs Python script to collect data

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

echo "=== Single Arduino MQ-135 Protocol ==="
echo "Arduino: Connected to USB at $PORT"
echo ""

echo "Uploading sketch to Arduino at $PORT..."
arduino-cli compile --fqbn arduino:avr:uno arduino_b_slave
if [ $? -ne 0 ]; then
  echo "Compilation failed!"
  exit 1
fi
arduino-cli upload -p "$PORT" --fqbn arduino:avr:uno arduino_b_slave
if [ $? -ne 0 ]; then
  echo "Upload failed!"
  exit 1
fi

echo "Waiting for Arduino to reboot..."
sleep 3

echo ""
echo "Starting data collection..."
python /home/raspberrypi/Ulurover-Minilab/Arduino-Serial-Connection/dual-mq135/read-dual-mq135-csv.py --seconds 10
