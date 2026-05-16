# Single Arduino MQ-135 CO2 Sensor Protocol

This protocol reads CO2 PPM values from multiple MQ-135 sensors connected directly to a single Arduino.

## Hardware Setup

- **Arduino (Main)**: Connected to Raspberry Pi via USB
  - Multiple MQ-135 sensors on pins A0, A1, A2 (up to 3 by default, expandable)
  - Direct analog connections - no I2C required

### Sensor Wiring
- MQ-135 Sensor 1: A0
- MQ-135 Sensor 2: A1
- MQ-135 Sensor 3: A2
- Power: 5V
- Ground: GND

## Files

1. **arduino_b_slave/** - Sketch folder with Arduino sketch for single Arduino operation
   - Contains `arduino_b_slave.ino` (updated to remove I2C code)
2. **read-dual-mq135-csv.py** - Python script to read and log data
3. **check-sensor-voltages.py** - Diagnostic tool to monitor sensor differences 
4. **run-dual-mq135.sh** - Main script to upload and run
5. **upload-arduino-b.sh** - Helper script to upload to Arduino
6. **dashboard/** - Web dashboard for data collection and visualization

**Note:** Arduino sketch must be in its own folder with matching names for arduino-cli to compile it.

## Data Format

Output CSV format: `Seconds,CO2_PPM_1,CO2_PPM_2,CO2_PPM_3`

Example:
```
Seconds,CO2_PPM_1,CO2_PPM_2,CO2_PPM_3
0,425.3,422.1,428.5
2,426.1,423.2,429.1
4,425.8,422.9,428.3
```
  - Output: `Seconds,CO2_PPM_A,CO2_PPM_B_Sensor (B1)` or `(B2)` etc.

## Customization

### Baseline Offset (NEW)

**When sensors are not properly warmed up**, they often read extremely low values (1-10 PPM instead of expected 400-420 PPM). The code now includes a **BASELINE_OFFSET_PPM** feature to compensate:

```cpp
#define BASELINE_OFFSET_PPM 400.0  // Default: adds 400 PPM to all readings
```

**To adjust the offset** in [arduino_b_slave/arduino_b_slave.ino](arduino_b_slave/arduino_b_slave.ino):
- Set to `0` for no offset (when sensors are properly calibrated)
- Set to `400` to baseline uncalibrated sensors to atmospheric level
- Adjust based on known reference (e.g., outdoor air should read ~420 PPM)

**This is a temporary workaround.** For accurate measurements:
1. Warm up sensors for 24-48 hours
2. Calibrate in known CO2 environment (fresh outdoor air)
3. Reduce or eliminate offset once sensors stabilize

### Why Sensors Show Different Values

If your 3 sensors consistently show different readings (e.g., B1=404 PPM, B2=408 PPM, B3=402 PPM):

**Normal variations (±5-10 PPM):**
- Individual sensor tolerances
- Slight manufacturing differences
- Acceptable for most applications

**Large variations (>20 PPM):**
1. **Voltage differences** (most common cause)
   - Check breadboard power distribution
   - Measure voltage at each sensor (should be 5.0V ±0.1V)
   - Voltage drops cause different readings for same CO2 level
   
2. **Different environmental exposure**
   - Sensors placed at different heights/locations
   - Air flow differences
   - Temperature gradients
   
3. **Individual calibration needed**
   - Each sensor may need unique R0 value
   - Use diagnostic script to measure: `./check-sensor-voltages.py`

### Adding More Sensors to Arduino B

Edit [arduino_b_slave/arduino_b_slave.ino](arduino_b_slave/arduino_b_slave.ino):

1. Change `NUM_SENSORS` definition (line 21):
   ```cpp
   #define NUM_SENSORS 5  // Change from 3 to your number
   ```

2. Add sensor objects (after line 25):
   ```cpp
   MQUnifiedsensor MQ135_4(Board, Voltage_Resolution, ADC_Bit_Resolution, A3, Type);
   MQUnifiedsensor MQ135_5(Board, Voltage_Resolution, ADC_Bit_Resolution, A4, Type);
   ```

3. Update array size (line 29):
   ```cpp
   float co2_readings[NUM_SENSORS] = {0.0, 0.0, 0.0, 0.0, 0.0};
   ```

4. Initialize sensors in `setup()` (around line 39):
   ```cpp
   initializeSensor(MQ135_4, 3);
   initializeSensor(MQ135_5, 4);
   ```

5. Update readings in `loop()` (around line 55):
   ```cpp
   MQ135_4.update();
   co2_readings[3] = MQ135_4.readSensor();
   
   MQ135_5.update();
   co2_readings[4] = MQ135_5.readSensor();
   ```

Then update [arduino_a_master/arduino_a_master.ino](arduino_a_master/arduino_a_master.ino):

1. Change `NUM_SLAVE_SENSORS` (line 21):
   ```cpp
   #define NUM_SLAVE_SENSORS 5
   ```

## Installation Steps

### 1. Upload Arduino B First

**IMPORTANT**: You must upload the slave sketch to Arduino B BEFORE connecting it via I2C.

1. Connect Arduino B to the Raspberry Pi via USB
2. Run: `./upload-arduino-b.sh`
3. Disconnect Arduino B from USB
4. Connect Arduino B to Arduino A via I2C (SDA, SCL, GND)

### 2. Run the Full Protocol

1. Connect Arduino A to the Raspberry Pi via USB
2. Ensure Arduino B is connected to Arduino A via I2C
3. Run: `./run-dual-mq135.sh`

## Output

The script generates a CSV file with the format:
```
Seconds,CO2_PPM_A,CO2_PPM_B1,CO2_PPM_B2,CO2_PPM_B3
0,450.23,452.17,448.92,451.33
2,451.34,453.22,449.81,452.45
4,449.87,451.98,448.55,450.92
...
```

- **CO2_PPM_A**: Reading from Arduino A's MQ-135
- **CO2_PPM_B1**: Reading from Arduino B's sensor 1 (A0)
- **CO2_PPM_B2**: Reading from Arduino B's sensor 2 (A1)
- **CO2_PPM_B3**: Reading from Arduino B's sensor 3 (A2)
- (Add more columns if you have more sensors)

## Python Script Options

```bash
python read-dual-mq135-csv.py [OPTIONS]

Options:
  --port PORT       Serial port (default: auto-detect)
  --baud RATE       Baud rate (default: 9600)
  --seconds SEC     Duration to run (0 = indefinite)
  --output FILE     CSV output filename
```

## Troubleshooting

### General Issues
- If no I2C communication: Check wiring (SDA, SCL, GND)
- If Arduino B not responding: Verify I2C address is 0x08
- If readings only from A: Arduino B may not be powered or not running slave sketch
- Check I2C scanner: Upload i2c_scanner example to Arduino A to verify Arduino B is detected

### Low or Inaccurate CO2 Readings

**Expected values:** Normal atmospheric CO2 is 400-420 PPM. Indoor levels typically range from 400-1000 PPM.

**If you see very low readings (< 100 PPM):**

1. **Sensors need warmup time (MOST COMMON)**
   - MQ-135 sensors require **24-48 hours** of continuous power to stabilize
   - First readings after power-on are unreliable
   - Solution: Leave both Arduinos powered on for 24-48 hours

2. **Improved calibration (v2.0)**
   - The code now uses 100 calibration samples with outlier filtering
   - Calibration happens during setup (takes ~15 seconds)
   - Upload takes longer due to improved calibration during startup

3. **Verify sensor connections**
   - Check that MQ-135 sensors are properly connected to A0, A1, A2
   - Ensure 5V and GND are connected to sensors

4. **Calibration procedure for best results:**
   - Power on the Arduinos and let sensors warm up for 24-48 hours
   - Place sensors in fresh outdoor air (known ~420 PPM CO2)
   - Re-upload the Arduino B sketch to recalibrate
   - The improved calibration will average 100 samples while filtering outliers

5. **Check specific sensor values:**
   - If one sensor reads correctly but others don't, check individual wiring
   - Try swapping sensors to isolate hardware issues

6. **Advanced: Manual R0 calibration**
   - If automatic calibration fails, you can manually set R0 values
   - Calculate R0 in known clean air and hardcode in the sketch

## Dependencies

- MQUnifiedsensor library (installed in Arduino libraries)
- Python serial library
- arduino-cli
- arduino-serial conda environment
