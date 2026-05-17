#include <MQUnifiedsensor.h>
#include <Wire.h>
#include <RTClib.h>
#include <SPI.h>
#include <SD.h>
#include <Adafruit_BME280.h>
#include <Adafruit_VEML6070.h>

//  BOARD
#define Board "Arduino Nano"
#define Voltage_Resolution 5
#define ADC_Bit_Resolution 10

//  MQ
#define PinMQ135 A1
#define PinMQ4   A2
#define PinMQ8   A3

#define RatioMQ135CleanAir 3.6
#define RatioMQ4CleanAir   4.4
#define RatioMQ8CleanAir   70

MQUnifiedsensor MQ135(Board, Voltage_Resolution, ADC_Bit_Resolution, PinMQ135, "MQ-135");
MQUnifiedsensor MQ4  (Board, Voltage_Resolution, ADC_Bit_Resolution, PinMQ4,   "MQ-4");
MQUnifiedsensor MQ8  (Board, Voltage_Resolution, ADC_Bit_Resolution, PinMQ8,   "MQ-8");

//  RTC
RTC_DS3231 rtc;
bool rtcOK = false;

//  ENV
Adafruit_BME280 bme;
bool bmeOK = false;

Adafruit_VEML6070 uv;
bool uvOK = false;

//  SD
#define SD_CS 2
bool sdOK = false;
File dataFile;

//  TIMING
unsigned long lastLog = 0;
const unsigned long interval = 5000;

void setup() {
  Serial.begin(9600);
  Wire.begin();
  delay(1000);

  Serial.println("SYSTEM STARTING");

  // RTC
  if (rtc.begin()) {
    rtcOK = true;
    Serial.println("RTC: OK");
  } else {
    Serial.println("RTC: NOT FOUND");
  }

  // SD
  if (SD.begin(SD_CS)) {
    sdOK = true;
    Serial.println("SD: OK");

    dataFile = SD.open("data.csv", FILE_WRITE);
    if (dataFile && dataFile.size() == 0) {
      dataFile.println("Date,Time,CO2,CH4,H2,Temp,Pressure,Humidity,UV");
    }
    if (dataFile) dataFile.close();
  } else {
    Serial.println("SD: NOT FOUND");
  }

  // BME280
  if (bme.begin(0x76)) {
    bmeOK = true;
    Serial.println("BME280: OK");
  } else {
    Serial.println("BME280: NOT FOUND");
  }

  // VEML6070
  uv.begin(VEML6070_1_T);
  uvOK = true;   // the library is not returning an error
  Serial.println("VEML6070: OK");

  // MQ135
  MQ135.setRegressionMethod(1);
  MQ135.setA(110.47);
  MQ135.setB(-2.862);
  MQ135.init();

  float r0 = 0;
  for (int i = 0; i < 5; i++) {
    MQ135.update();
    r0 += MQ135.calibrate(RatioMQ135CleanAir);
  }
  MQ135.setR0(r0 / 5);

  // MQ4
  MQ4.setRegressionMethod(1);
  MQ4.setA(1012.7);
  MQ4.setB(-2.786);
  MQ4.init(); 

  r0 = 0;
  for (int i = 0; i < 5; i++) {
    MQ4.update();
    r0 += MQ4.calibrate(RatioMQ4CleanAir);
  }
  MQ4.setR0(r0 / 5);

  // MQ8
  MQ8.setRegressionMethod(1);
  MQ8.setA(976.97);
  MQ8.setB(-0.688);
  MQ8.init();

  r0 = 0;
  for (int i = 0; i < 5; i++) {
    MQ8.update();
    r0 += MQ8.calibrate(RatioMQ8CleanAir);
  }
  MQ8.setR0(r0 / 5);

  Serial.println("SETUP FINISHED");
  Serial.println("Date,Time,CO2,CH4,H2,Temp,Pressure,Humidity,UV");
}

void loop() {
  if (millis() - lastLog < interval) return;
  lastLog = millis();

  // TIME
  String date = "NA", time = "NA";
  if (rtcOK) {
    DateTime now = rtc.now();
    date = String(now.year()) + "-" + now.month() + "-" + now.day();
    time = String(now.hour()) + ":" + now.minute() + ":" + now.second();
  }

  // MQ
  MQ135.update();
  MQ4.update();
  MQ8.update();

  float co2 = MQ135.readSensor();
  float ch4 = MQ4.readSensor();
  float h2  = MQ8.readSensor();

  // ENV
  float temp = bmeOK ? bme.readTemperature() : NAN;
  float pres = bmeOK ? bme.readPressure() / 100.0 : NAN;
  float hum  = bmeOK ? bme.readHumidity() : NAN;
  int uvVal  = uvOK  ? uv.readUV() : -1;

  // SERIAL
  Serial.print(date); Serial.print(",");
  Serial.print(time); Serial.print(",");
  Serial.print(co2);  Serial.print(",");
  Serial.print(ch4);  Serial.print(",");
  Serial.print(h2);   Serial.print(",");
  Serial.print(temp); Serial.print(",");
  Serial.print(pres); Serial.print(",");
  Serial.print(hum);  Serial.print(",");
  Serial.println(uvVal);

  // SD
  if (sdOK) {
    dataFile = SD.open("data.csv", FILE_WRITE);
    if (dataFile) {
      dataFile.print(date); dataFile.print(",");
      dataFile.print(time); dataFile.print(",");
      dataFile.print(co2);  dataFile.print(",");
      dataFile.print(ch4);  dataFile.print(",");
      dataFile.print(h2);   dataFile.print(",");
      dataFile.print(temp); dataFile.print(",");
      dataFile.print(pres); dataFile.print(",");
      dataFile.print(hum);  dataFile.print(",");
      dataFile.println(uvVal);
      dataFile.close();
    }
  }
}