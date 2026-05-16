/**
 * Wetlab Nano Controller
 *
 * Hardware mapped from the updated minilab schematic:
 * - MQ-135 gas sensors on A5, A4, A3
 * - Peristaltic pumps on D2, D4, D5, D6, D7, D8, D9, D10
 * - Servo PWM on D3
 * - Status LED on D11
 * - Weight sensor interfaces reserved on D16/D15 and D13/D12
 *
 * The sketch currently streams gas sensor readings as CSV so the existing
 * logging/dashboard pipeline can continue to work while the actuator layer is
 * brought online.
 */

#include <MQUnifiedsensor.h>

#define Board "Arduino Nano"
#define Type "MQ-135"
#define Voltage_Resolution 5
#define ADC_Bit_Resolution 10
#define RatioMQ135CleanAir 3.6

#define NUM_GAS_SENSORS 3
#define BASELINE_OFFSET_PPM 400.0

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

  pinMode(servoPin, OUTPUT);
  digitalWrite(servoPin, LOW);
}

void printHardwareMap() {
  Serial.println("SYSTEM STARTING");
  Serial.println("WETLAB: Nano controller ready");
  Serial.println("PUMPS: D2,D4,D5,D6,D7,D8,D9,D10");
  Serial.println("SERVO: D3");
  Serial.println("LED: D11");
  Serial.println("WEIGHT1: D16/D15");
  Serial.println("WEIGHT2: D13/D12");
  Serial.println("Date/Time not used on this board");
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

  for (uint8_t i = 0; i < NUM_GAS_SENSORS; i++) {
    gasSensors[i].update();
    co2Readings[i] = gasSensors[i].readSensor() + BASELINE_OFFSET_PPM;
  }

  Serial.print(seconds);
  for (uint8_t i = 0; i < NUM_GAS_SENSORS; i++) {
    Serial.print(',');
    Serial.print(co2Readings[i]);
  }
  Serial.println();

  seconds += 2;
  delay(2000);
}