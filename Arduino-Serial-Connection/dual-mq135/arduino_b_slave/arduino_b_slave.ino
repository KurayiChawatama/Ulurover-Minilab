/**
 * Single Arduino - Multiple MQ-135 Sensors (Standalone)
 * Reads from 3 MQ-135 CO2 sensors and outputs CSV via Serial
 * Outputs CSV: Seconds,CO2_PPM_1,CO2_PPM_2,CO2_PPM_3
 *
 * CURRENT CONFIGURATION: 3 sensors for reaction rate monitoring
 *
 * Sensors: MQ-135 on A0, A1, A2 (3 sensors active)
 * Board: Arduino UNO
 * Connection: USB to Raspberry Pi
 * Baud Rate: 9600
 * 
 * BASELINE OFFSET FEATURE:
 *   - Adds configurable offset to all readings (default: 400 PPM)
 *   - Compensates for uncalibrated/cold sensors
 *   - Adjust BASELINE_OFFSET_PPM to tune (0 = no offset)
 * 
 * CALIBRATION IMPROVEMENTS:
 *   - 100 calibration samples (up from 10) for better accuracy
 *   - 2-second warmup period before calibration
 *   - Statistical outlier filtering (2 sigma threshold)
 *   - 100ms delay between samples for sensor stability
 * 
 * DIAGNOSTIC MODE:
 *   - Monitors raw analog values and voltages
 *   - Helps diagnose voltage differences between sensors
 *   - Enable/disable with ENABLE_DIAGNOSTICS flag
 * 
 * TROUBLESHOOTING VOLTAGE DIFFERENCES:
 *   - If sensors show different readings, check:
 *     * Power supply quality (should be stable 5V)
 *     * Breadboard connections (voltage drops on long traces)
 *     * Individual sensor wiring
 *   - Different voltages = different readings even for same CO2 level
 * 
 * IMPORTANT: MQ-135 sensors require 24-48 hours of continuous power
 *            for optimal accuracy. First readings may be inaccurate.
 *            Normal atmospheric CO2 is ~400-420 PPM.
 *
 * Author: Kurayi Chawatama
 * Modified: Removed I2C slave code for standalone operation
 */

#include <MQUnifiedsensor.h>

#define Board "Arduino UNO"
#define Type "MQ-135"
#define Voltage_Resolution 5
#define ADC_Bit_Resolution 10
#define RatioMQ135CleanAir 3.6
#define NUM_SENSORS 3  // Change this to match your number of sensors

// BASELINE OFFSET: Add this to compensate for uncalibrated sensors
// Set to 0 for no offset, or ~400 to baseline to atmospheric CO2
#define BASELINE_OFFSET_PPM 400.0  // Adjust as needed

// DIAGNOSTIC MODE: Enable to also read raw analog values and voltages
#define ENABLE_DIAGNOSTICS true

// Create sensor objects for each MQ-135
MQUnifiedsensor MQ135_1(Board, Voltage_Resolution, ADC_Bit_Resolution, A0, Type);
// Uncomment additional sensors as needed:
MQUnifiedsensor MQ135_2(Board, Voltage_Resolution, ADC_Bit_Resolution, A1, Type);
MQUnifiedsensor MQ135_3(Board, Voltage_Resolution, ADC_Bit_Resolution, A2, Type);
// MQUnifiedsensor MQ135_4(Board, Voltage_Resolution, ADC_Bit_Resolution, A3, Type);

float co2_readings[NUM_SENSORS];
float raw_voltages[NUM_SENSORS];  // Store raw voltages for diagnostics
int raw_analog[NUM_SENSORS];      // Store raw analog values

void setup() {
  Serial.begin(9600);  // Initialize serial communication
  
  // Print CSV header
  Serial.print("Seconds");
  for (int i = 1; i <= NUM_SENSORS; i++) {
    Serial.print(",CO2_PPM_");
    Serial.print(i);
  }
  Serial.println();
  
  // Initialize and calibrate all sensors
  initializeSensor(MQ135_1, 0);
  // Uncomment additional sensors as needed:
  initializeSensor(MQ135_2, 1);
  initializeSensor(MQ135_3, 2);
  // initializeSensor(MQ135_4, 3);
}

void initializeSensor(MQUnifiedsensor &sensor, int index) {
  sensor.setRegressionMethod(1);
  sensor.setA(110.47);
  sensor.setB(-2.862);
  sensor.init();
  
  // Warmup period: discard first readings to stabilize sensor
  for (int i = 0; i < 20; i++) {
    sensor.update();
    delay(100);
  }
  
  // Improved calibration with more samples and outlier filtering
  float calcR0 = 0;
  int validSamples = 0;
  const int totalSamples = 100;
  const int warmupSamples = 10;
  
  // Additional warmup samples (discarded)
  for (int i = 0; i < warmupSamples; i++) {
    sensor.update();
    sensor.calibrate(RatioMQ135CleanAir);
    delay(200);
  }
  
  // Collect calibration samples
  float samples[totalSamples];
  for (int i = 0; i < totalSamples; i++) {
    sensor.update();
    samples[i] = sensor.calibrate(RatioMQ135CleanAir);
    delay(100);
  }
  
  // Calculate mean and standard deviation for outlier filtering
  float sum = 0;
  for (int i = 0; i < totalSamples; i++) {
    sum += samples[i];
  }
  float mean = sum / totalSamples;
  
  float variance = 0;
  for (int i = 0; i < totalSamples; i++) {
    float diff = samples[i] - mean;
    variance += diff * diff;
  }
  float stdDev = sqrt(variance / totalSamples);
  
  // Average samples within 2 standard deviations (reject outliers)
  for (int i = 0; i < totalSamples; i++) {
    if (abs(samples[i] - mean) <= 2 * stdDev) {
      calcR0 += samples[i];
      validSamples++;
    }
  }
  
  // Set R0 with filtered average
  if (validSamples > 0) {
    sensor.setR0(calcR0 / validSamples);
  } else {
    sensor.setR0(mean);  // Fallback to mean if all rejected
  }
}

void loop() {
  static unsigned long seconds = 0;
  
  // Update all sensor readings with baseline offset
  MQ135_1.update();
  co2_readings[0] = MQ135_1.readSensor() + BASELINE_OFFSET_PPM;
  if (ENABLE_DIAGNOSTICS) {
    raw_analog[0] = analogRead(A0);
    raw_voltages[0] = (raw_analog[0] / 1024.0) * Voltage_Resolution;
  }
  
  // Uncomment additional sensors as needed:
  MQ135_2.update();
  co2_readings[1] = MQ135_2.readSensor() + BASELINE_OFFSET_PPM;
  if (ENABLE_DIAGNOSTICS) {
    raw_analog[1] = analogRead(A1);
    raw_voltages[1] = (raw_analog[1] / 1024.0) * Voltage_Resolution;
  }
  
  MQ135_3.update();
  co2_readings[2] = MQ135_3.readSensor() + BASELINE_OFFSET_PPM;
  if (ENABLE_DIAGNOSTICS) {
    raw_analog[2] = analogRead(A2);
    raw_voltages[2] = (raw_analog[2] / 1024.0) * Voltage_Resolution;
  }
  
  // MQ135_4.update();
  // co2_readings[3] = MQ135_4.readSensor() + BASELINE_OFFSET_PPM;
  // if (ENABLE_DIAGNOSTICS) {
  //   raw_analog[3] = analogRead(A3);
  //   raw_voltages[3] = (raw_analog[3] / 1024.0) * Voltage_Resolution;
  // }
  
  // Output CSV row with timestamp and all sensor readings
  Serial.print(seconds);
  for (int i = 0; i < NUM_SENSORS; i++) {
    Serial.print(",");
    Serial.print(co2_readings[i]);
  }
  Serial.println();
  
  seconds += 2;
  delay(2000);  // Update every 2 seconds
}
