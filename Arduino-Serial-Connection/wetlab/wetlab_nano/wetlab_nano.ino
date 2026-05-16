/**
 * Wetlab Nano Controller with Servo Control
 *
 * Hardware mapped from the updated minilab schematic:
 * - MQ-135 gas sensors on A5, A4, A3
 * - Peristaltic pumps on D2, D4, D5, D6, D7, D8, D9, D10
 * - Servo PWM on D3 (rotating drum)
 * - Status LED on D11
 * - Weight sensor interfaces reserved on D16/D15 and D13/D12
 *
 * The sketch streams gas sensor readings as CSV and accepts serial commands
 * for servo control (SERVO_START, SERVO_STOP, SERVO_ANGLE:0-180).
 */

#include <MQUnifiedsensor.h>

#define Board "Arduino Nano"
#define Type "MQ-135"
#define Voltage_Resolution 5
#define ADC_Bit_Resolution 10
#define RatioMQ135CleanAir 3.6

#define NUM_GAS_SENSORS 3
#define BASELINE_OFFSET_PPM 400.0

// Servo control with PWM (no Servo library needed)
#define SERVO_MIN_PULSE 1000  // 1ms = 0 degrees
#define SERVO_MAX_PULSE 2000  // 2ms = 180 degrees

const uint8_t gasSensorPins[NUM_GAS_SENSORS] = {A5, A4, A3};
const uint8_t pumpPins[8] = {2, 4, 5, 6, 7, 8, 9, 10};
const uint8_t servoPin = 3;
const uint8_t ledPin = 11;
const uint8_t weightSensor1DtPin = 16;
const uint8_t weightSensor1SckPin = 15;
const uint8_t weightSensor2DtPin = 13;
const uint8_t weightSensor2SckPin = 12;

MQUnifiedsensor gasSensors[NUM_GAS_SENSORS] = {
  MQUnifiedsensor(Board, Voltage_Resolution, ADC_Bit_Resolution, gasSensorPins[0], Type),
  MQUnifiedsensor(Board, Voltage_Resolution, ADC_Bit_Resolution, gasSensorPins[1], Type),
  MQUnifiedsensor(Board, Voltage_Resolution, ADC_Bit_Resolution, gasSensorPins[2], Type)
};

float co2Readings[NUM_GAS_SENSORS];

// Servo control state (adapted for continuous 360° servos)
bool servoRunning = false;
int servoTargetAngle = 0; // retained for compatibility (maps to speed)
int servoPulseWidth = 1500; // neutral (stop) pulse width in microseconds
unsigned long lastServoPulseTime = 0;

void calibrateSensor(MQUnifiedsensor &sensor) {
  sensor.setRegressionMethod(1);
  sensor.setA(110.47);
  sensor.setB(-2.862);
  sensor.init();

  float r0 = 0;
  const int sampleCount = 10;
  for (int i = 0; i < sampleCount; i++) {
    sensor.update();
    r0 += sensor.calibrate(RatioMQ135CleanAir);
    delay(100);
  }
  sensor.setR0(r0 / sampleCount);
}

void configureOutputs() {
  for (uint8_t i = 0; i < 8; i++) {
    pinMode(pumpPins[i], OUTPUT);
    digitalWrite(pumpPins[i], LOW);
  }

  pinMode(ledPin, OUTPUT);
  digitalWrite(ledPin, LOW);

  pinMode(weightSensor1DtPin, INPUT);
  pinMode(weightSensor1SckPin, OUTPUT);
  pinMode(weightSensor2DtPin, INPUT);
  pinMode(weightSensor2SckPin, OUTPUT);

  digitalWrite(weightSensor1SckPin, LOW);
  digitalWrite(weightSensor2SckPin, LOW);

  // Setup servo PWM on D3
  pinMode(servoPin, OUTPUT);
  digitalWrite(servoPin, LOW);
  // Initialize pulse-width control for continuous servo
  servoPulseWidth = 1500; // neutral/stop
  lastServoPulseTime = millis();
}

void handleSerialCommands() {
  if (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    
    if (cmd.startsWith("SERVO_START")) {
      // For continuous servos, start means run at default forward speed
      servoRunning = true;
      servoTargetAngle = 180; // map to maximum forward speed
      servoPulseWidth = SERVO_MAX_PULSE;
      Serial.println("SERVO_STARTED");
    }
    else if (cmd.startsWith("SERVO_STOP")) {
      // Stop the servo by sending the neutral pulse width
      servoRunning = false;
      servoPulseWidth = 1500; // neutral/stop for continuous servos
      Serial.println("SERVO_STOPPED");
    }
    else if (cmd.startsWith("SERVO_ANGLE:")) {
      // Keep the old command name for compatibility but interpret the
      // provided "angle" as a speed mapping for continuous servos.
      int angle = cmd.substring(12).toInt();
      if (angle >= 0 && angle <= 180) {
        servoTargetAngle = angle;
        // Map 0..180 -> 1000..2000us pulse width
        servoPulseWidth = map(angle, 0, 180, SERVO_MIN_PULSE, SERVO_MAX_PULSE);
        // If angle is near middle, consider it stopped
        if (abs(servoPulseWidth - 1500) <= 20) {
          servoRunning = false;
        } else {
          servoRunning = true;
        }
        Serial.println("SERVO_ANGLE_SET:" + String(angle));
      }
    }
  }
}

// Send a single 50Hz servo pulse using the configured pulse width
void writePWMServoPulse(int pulseWidth) {
  digitalWrite(servoPin, HIGH);
  delayMicroseconds(pulseWidth);
  digitalWrite(servoPin, LOW);
}

// The original code sent a block of 20 pulses then slept for ~2s which
// produces discrete, stepped movement. For continuous servos we must
// produce a steady 50Hz stream of pulses. We'll drive the servo pulse
// each 20ms and perform sensor sampling/CSV output on a separate 2s timer.

void printHardwareMap() {
  Serial.println("SYSTEM STARTING");
  Serial.println("WETLAB: Nano controller ready");
  Serial.println("PUMPS: D2,D4,D5,D6,D7,D8,D9,D10");
  Serial.println("SERVO: D3 (rotating drum - external power)");
  Serial.println("LED: D11");
  Serial.println("WEIGHT1: D16/D15");
  Serial.println("WEIGHT2: D13/D12");
  Serial.println("Date/Time not used on this board");
  Serial.println("SERIAL COMMANDS: SERVO_START, SERVO_STOP, SERVO_ANGLE:0-180");
  Serial.println("Seconds,CO2_PPM_1,CO2_PPM_2,CO2_PPM_3");
}

void setup() {
  Serial.begin(9600);
  delay(1000);

  configureOutputs();

  for (uint8_t i = 0; i < NUM_GAS_SENSORS; i++) {
    calibrateSensor(gasSensors[i]);
  }

  printHardwareMap();
}

void loop() {
  static unsigned long seconds = 0;
  static unsigned long lastSensorTime = 0;

  // Handle incoming serial commands for servo and other controls
  handleSerialCommands();

  unsigned long now = millis();

  // Send servo pulse every ~20ms (50Hz)
  if (now - lastServoPulseTime >= 20) {
    writePWMServoPulse(servoPulseWidth);
    lastServoPulseTime = now;
  }

  // Read sensors and output CSV every ~2000ms
  if (now - lastSensorTime >= 2000) {
    // Read gas sensors
    for (uint8_t i = 0; i < NUM_GAS_SENSORS; i++) {
      gasSensors[i].update();
      co2Readings[i] = gasSensors[i].readSensor() + BASELINE_OFFSET_PPM;
    }

    // Output CSV data
    Serial.print(seconds);
    for (uint8_t i = 0; i < NUM_GAS_SENSORS; i++) {
      Serial.print(',');
      Serial.print(co2Readings[i]);
    }
    Serial.println();

    seconds += 2;
    lastSensorTime = now;
  }
}
