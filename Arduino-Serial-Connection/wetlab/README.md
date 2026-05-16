# Wetlab Nano MQ-135 Protocol

This protocol reads CO2 PPM values from multiple MQ-135 sensors connected directly to the wetlab Nano.

## Hardware Setup

- **Wetlab Nano**: Connected to Raspberry Pi via USB
   - Multiple MQ-135 sensors on pins A5, A4, A3
   - Direct analog connections - no I2C required

### Sensor Wiring
- MQ-135 Sensor 1: A5
- MQ-135 Sensor 2: A4
- MQ-135 Sensor 3: A3
- Power: 5V
- Ground: GND

## Files

1. **wetlab_nano/** - Sketch folder with the current wetlab controller sketch
   - Contains `wetlab_nano.ino`
2. **read-wetlab-csv.py** - Python script to read and log data
3. **check-wetlab-sensors.py** - Diagnostic tool to monitor sensor differences 
4. **run-wetlab.sh** - Main script to upload and run
5. **upload-wetlab-nano.sh** - Helper script to upload to Arduino
6. **dashboard/** - Web dashboard for wetlab and weather-station data collection and visualization

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

**To adjust the offset** in [wetlab_nano/wetlab_nano.ino](wetlab_nano/wetlab_nano.ino):
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
   - Use diagnostic script to measure: `./check-wetlab-sensors.py`

### Adding More Sensors to the Wetlab Nano

Edit [wetlab_nano/wetlab_nano.ino](wetlab_nano/wetlab_nano.ino):

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

Then update `NUM_GAS_SENSORS` in [wetlab_nano/wetlab_nano.ino](wetlab_nano/wetlab_nano.ino).

## Installation Steps

### 1. Upload the Wetlab Nano First

1. Connect the wetlab Nano to the Raspberry Pi via USB
2. Run: `./upload-wetlab-nano.sh`

### 2. Run the Full Protocol

1. Connect the wetlab Nano to the Raspberry Pi via USB
2. Ensure the weather-station Arduino is connected separately and powered if you want environmental data
3. Run: `./run-wetlab.sh`

## Output

The script generates a CSV file with the format:
```
Seconds,CO2_PPM_A,CO2_PPM_B1,CO2_PPM_B2,CO2_PPM_B3
0,450.23,452.17,448.92,451.33
2,451.34,453.22,449.81,452.45
4,449.87,451.98,448.55,450.92
...
```

- **CO2_PPM_1**: Reading from wetlab sensor 1 (A5)
- **CO2_PPM_2**: Reading from wetlab sensor 2 (A4)
- **CO2_PPM_3**: Reading from wetlab sensor 3 (A3)
- (Add more columns if you have more sensors)

## Python Script Options

```bash
python read-wetlab-csv.py [OPTIONS]

Options:
  --port PORT       Serial port (default: auto-detect)
  --baud RATE       Baud rate (default: 9600)
  --seconds SEC     Duration to run (0 = indefinite)
  --output FILE     CSV output filename
```

## Troubleshooting

### General Issues
- If no I2C communication: Check wiring (SDA, SCL, GND)
- If the wetlab Nano is not responding: Verify the USB connection and that the sketch was uploaded successfully
- If readings are missing: Confirm the sensors are wired to A5, A4, and A3
- Check the serial monitor for the boot banner and CSV header

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
   - Check that MQ-135 sensors are properly connected to A5, A4, A3
   - Ensure 5V and GND are connected to sensors

4. **Calibration procedure for best results:**
   - Power on the Arduinos and let sensors warm up for 24-48 hours
   - Place sensors in fresh outdoor air (known ~420 PPM CO2)
   - Re-upload the wetlab Nano sketch to recalibrate
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
