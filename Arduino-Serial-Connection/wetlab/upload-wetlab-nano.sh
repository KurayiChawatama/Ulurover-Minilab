#!/bin/bash
# Upload Wetlab Nano Sketch
# Uploads the updated wetlab controller sketch for the new schematic

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

echo "=== Wetlab Nano Upload ==="
echo "Port: $PORT"
echo ""

echo "Compiling wetlab_nano.ino..."
arduino-cli compile --fqbn arduino:avr:nano wetlab_nano
if [ $? -ne 0 ]; then
  echo "Compilation failed!"
  exit 1
fi

echo "Uploading to Arduino..."
arduino-cli upload -p "$PORT" --fqbn arduino:avr:nano wetlab_nano
if [ $? -ne 0 ]; then
  echo "Upload failed!"
  exit 1
fi

echo ""
echo "SUCCESS! Arduino is now programmed with wetlab controller support."
echo ""
echo "Next steps:"
echo "1. Arduino is ready to collect wetlab gas sensor data"
echo "2. Connect your MQ-135 sensors to:"
echo "   - Sensor 1: A5"
echo "   - Sensor 2: A4"
echo "   - Sensor 3: A3"
echo "3. Run ./run-wetlab.sh to start data collection"
