# Arduino Dual Blink Control via Raspberry Pi

## Overview
This project allows you to control the blink rate of two Arduino Nanos connected to a Raspberry Pi via USB, using a Python script and serial communication.

## Quick Start
1. **Connect both Arduino Nanos** to the Raspberry Pi via USB.
2. **Upload the Arduino sketch** to both boards (see below for commands).
3. **Run the Python script** to control the blink rate:
   ```bash
   bash run-arduino-blink.sh
   ```
   Enter a blink rate in milliseconds (e.g., `1000` for 1 second) and press Enter. Type `exit` to quit.

## Uploading the Arduino Sketch
1. **Compile the sketch:**
   ```bash
   arduino-cli compile --fqbn arduino:avr:nano Arduino-Serial-Connection.ino
   ```
2. **Upload to each Nano:**
   ```bash
   arduino-cli upload -p /dev/ttyACM0 --fqbn arduino:avr:nano Arduino-Serial-Connection.ino
   arduino-cli upload -p /dev/ttyACM1 --fqbn arduino:avr:nano Arduino-Serial-Connection.ino
   ```

## Key Arduino CLI Commands
- `arduino-cli board list` — List connected Arduino boards and their ports.
- `arduino-cli compile --fqbn arduino:avr:nano <sketch>.ino` — Compile a sketch for Arduino Nano.
- `arduino-cli upload -p <port> --fqbn arduino:avr:nano <sketch>.ino` — Upload a sketch to a board on the given port.
- `arduino-cli core install arduino:avr` — Install the AVR core (required for most Arduino boards like Nano, Uno, Mega).

## What is `arduino:avr`?
- **`arduino:avr`** is the official Arduino core for AVR-based boards (like Nano, Uno, Mega). It provides the necessary tools and libraries to compile and upload code to these boards.
- The `--fqbn arduino:avr:nano` flag tells Arduino CLI to use the Nano board definition from the AVR core.

## Requirements
- Raspberry Pi with Python and Conda
- Two Arduino Nanos
- Arduino CLI installed and in your `$PATH`
- `pyserial` Python package installed in the `arduino-serial` conda environment

---
For more details, see the comments in the Python and Arduino files.
