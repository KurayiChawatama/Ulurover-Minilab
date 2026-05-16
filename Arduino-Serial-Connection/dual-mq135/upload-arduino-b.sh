#!/bin/bash
# Upload Arduino Sketch with MQ-135 Sensors
# Uploads the single Arduino sketch with multiple MQ-135 sensors connected

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

echo "=== Single Arduino Upload ==="
echo "Port: $PORT"
echo ""

echo "Compiling arduino_b_slave.ino..."
arduino-cli compile --fqbn arduino:avr:uno arduino_b_slave
if [ $? -ne 0 ]; then
  echo "Compilation failed!"
  exit 1
fi

echo "Uploading to Arduino..."
arduino-cli upload -p "$PORT" --fqbn arduino:avr:uno arduino_b_slave
if [ $? -ne 0 ]; then
  echo "Upload failed!"
  exit 1
fi

echo ""
echo "SUCCESS! Arduino is now programmed with MQ-135 sensor support."
echo ""
echo "Next steps:"
echo "1. Arduino is ready to collect MQ-135 sensor data"
echo "2. Connect your MQ-135 sensors to:"
echo "   - Sensor 1: A0"
echo "   - Sensor 2: A1"
echo "   - Sensor 3: A2"
echo "3. Run ./run-dual-mq135.sh to start data collection"
