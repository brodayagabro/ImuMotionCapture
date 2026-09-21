/*
  MPU-6500 x5 through TCA9548A for ESP32-C3/C6, with NVS Wi-Fi portal.
  Separate snapshot of MPU6500_with_TCA9548A_udp; no wifi_secrets.h required.

  The public Serial/UDP interface intentionally matches the MPU6050 DMP
  firmware:

    FRAME <sequence> <millis> <quaternion_count>
    Q <tca_channel> <w> <x> <y> <z>

  Commands:
    START, STOP, CALIB, CALIB_GYRO, STATUS, GET_CONFIG, HELLO, PING,
    SET_RATE <1..100>

  MPU-6500 registers are read directly. Orientation is calculated by the
  separate MahonyFilter implementation; MPU6050 MotionApps is not loaded.
*/

#include <Arduino.h>
#include <AsyncUDP.h>
#include <Preferences.h>
#include <WiFi.h>
#include <Wire.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#include "MahonyFilter.h"
#include "WiFiPortal.h"


static const uint32_t BAUD_RATE = 115200UL;
static const uint16_t SERIAL_WAIT_MS = 2000U;
static const uint32_t I2C_INIT_CLOCK_HZ = 100000UL;
static const uint32_t I2C_RUN_CLOCK_HZ = 400000UL;
static const uint32_t I2C_TIMEOUT_US = 3000UL;

static const uint16_t DEFAULT_STREAM_RATE_HZ = 10U;
static const uint16_t MIN_STREAM_RATE_HZ = 1U;
static const uint16_t MAX_STREAM_RATE_HZ = 100U;
static const uint16_t FILTER_RATE_HZ = 100U;
static const uint32_t FILTER_INTERVAL_US = 1000000UL / FILTER_RATE_HZ;
static const uint16_t QUAT_PRINT_SCALE = 10000U;

static const uint8_t LED_PIN = 13U;
static const uint8_t CMD_BUFFER_SIZE = 64U;
static const uint16_t UDP_PACKET_BUFFER_SIZE = 1024U;
static const uint8_t MAX_SENSORS = 5U;
static const uint8_t NO_TCA_CHANNEL = 0xFFU;
static const uint16_t CALIBRATION_SAMPLES = 300U;
static const uint8_t CALIBRATION_SAMPLE_DELAY_MS = 2U;

static const uint8_t TCA_ADDR_FIRST = 0x70U;
static const uint8_t TCA_ADDR_LAST = 0x77U;
static const uint8_t MPU_ADDR = 0x68U;
static const uint8_t MPU6500_WHO_AM_I = 0x70U;

static const uint8_t REG_SMPLRT_DIV = 0x19U;
static const uint8_t REG_CONFIG = 0x1AU;
static const uint8_t REG_GYRO_CONFIG = 0x1BU;
static const uint8_t REG_ACCEL_CONFIG = 0x1CU;
static const uint8_t REG_ACCEL_CONFIG_2 = 0x1DU;
static const uint8_t REG_FIFO_EN = 0x23U;
static const uint8_t REG_INT_ENABLE = 0x38U;
static const uint8_t REG_ACCEL_XOUT_H = 0x3BU;
static const uint8_t REG_USER_CTRL = 0x6AU;
static const uint8_t REG_PWR_MGMT_1 = 0x6BU;
static const uint8_t REG_PWR_MGMT_2 = 0x6CU;
static const uint8_t REG_WHO_AM_I = 0x75U;

static const float ACCEL_COUNTS_PER_G = 16384.0f;
static const float GYRO_COUNTS_PER_DEGREE_PER_SECOND = 131.0f;
static const float DEGREES_TO_RADIANS = 0.01745329251994329577f;
static const float FILTER_PROPORTIONAL_GAIN = 2.0f;
static const float FILTER_INTEGRAL_GAIN = 0.05f;
static const float FILTER_ACCEL_TOLERANCE_G = 0.25f;

static const uint16_t UDP_LISTEN_PORT = 4210U;

static const bool AUTO_DETECT_CHANNELS = true;
static const bool VERBOSE_I2C_SCAN = false;
static const uint8_t SENSOR_CHANNELS[MAX_SENSORS] = {0U, 1U, 2U, 6U, 7U};

static const char CALIBRATION_NVS_NAMESPACE[] = "mpu6500_cal";
static const uint32_t CALIBRATION_MAGIC = 0x4D363530UL;


struct RawSample {
  int16_t ax;
  int16_t ay;
  int16_t az;
  int16_t temperature;
  int16_t gx;
  int16_t gy;
  int16_t gz;
};


struct CalibrationData {
  uint32_t magic;
  float accelBiasX;
  float accelBiasY;
  float accelBiasZ;
  float gyroBiasX;
  float gyroBiasY;
  float gyroBiasZ;
};


struct SensorState {
  uint8_t id;
  uint8_t channel;
  uint8_t whoAmI;
  bool ready;
  bool hasQuaternion;
  bool calibrationLoaded;
  OrientationQuaternion quaternion;
  MahonyFilter filter;
  CalibrationData calibration;
  uint32_t readErrorCount;
  uint32_t filterUpdateCount;
};


AsyncUDP udp;
Preferences calibrationPreferences;
SensorState sensors[MAX_SENSORS];
uint8_t sensorCount = 0U;
uint8_t tcaAddress = 0U;
uint8_t activeTcaChannel = NO_TCA_CHANNEL;
bool calibrationStorageReady = false;

bool streamEnabled = false;
bool blinkState = false;
uint16_t streamRateHz = DEFAULT_STREAM_RATE_HZ;
uint32_t frameIntervalMs = 1000UL / DEFAULT_STREAM_RATE_HZ;
uint32_t previousFrameMs = 0UL;
uint32_t frameCounter = 0UL;
uint16_t lateFrameCount = 0U;
uint16_t lastFrameDurationUs = 0U;
uint16_t maxFrameDurationUs = 0U;
uint32_t previousFilterUs = 0UL;

bool udpServerReady = false;
bool udpClientRegistered = false;
IPAddress udpClientIp;
uint16_t udpClientPort = 0U;
WiFiPortal wifiPortal;

struct PendingUdpCommand {
  char text[CMD_BUFFER_SIZE];
  IPAddress remoteIp;
  uint16_t remotePort;
};

PendingUdpCommand pendingUdpCommand;
volatile bool udpCommandPending = false;
portMUX_TYPE udpCommandMux = portMUX_INITIALIZER_UNLOCKED;

char cmdBuffer[CMD_BUFFER_SIZE];
uint8_t cmdIdx = 0U;


void sendUdpReply(const IPAddress& remoteIp, uint16_t remotePort, const char* text) {
  if (!udpServerReady || remotePort == 0U || text == nullptr) {
    return;
  }
  udp.writeTo((const uint8_t*)text, strlen(text), remoteIp, remotePort);
}


void trimCommand(char* text) {
  if (text == nullptr) {
    return;
  }
  char* begin = text;
  while (*begin == ' ' || *begin == '\t' || *begin == '\r' || *begin == '\n') {
    begin++;
  }
  if (begin != text) {
    memmove(text, begin, strlen(begin) + 1U);
  }
  size_t length = strlen(text);
  while (length > 0U) {
    char value = text[length - 1U];
    if (value != ' ' && value != '\t' && value != '\r' && value != '\n') {
      break;
    }
    text[--length] = '\0';
  }
}


void queueUdpCommand(AsyncUDPPacket packet) {
  size_t length = packet.length();
  if (length == 0U) {
    return;
  }
  if (length >= CMD_BUFFER_SIZE) {
    packet.print("ERR CMD_TOO_LONG\n");
    return;
  }

  bool busy = false;
  portENTER_CRITICAL(&udpCommandMux);
  if (udpCommandPending) {
    busy = true;
  } else {
    memcpy(pendingUdpCommand.text, packet.data(), length);
    pendingUdpCommand.text[length] = '\0';
    pendingUdpCommand.remoteIp = packet.remoteIP();
    pendingUdpCommand.remotePort = packet.remotePort();
    udpCommandPending = true;
  }
  portEXIT_CRITICAL(&udpCommandMux);

  if (busy) {
    packet.print("ERR BUSY\n");
  }
}


bool startUdpServer() {
  udp.close();
  udpServerReady = false;
  if (!udp.listen(UDP_LISTEN_PORT)) {
    Serial.print(F("ERR:UDP_LISTEN "));
    Serial.println(UDP_LISTEN_PORT);
    return false;
  }
  udp.onPacket([](AsyncUDPPacket packet) {
    queueUdpCommand(packet);
  });
  udpServerReady = true;
  Serial.print(F("NET:UDP server "));
  Serial.print(WiFi.localIP());
  Serial.write(':');
  Serial.println(UDP_LISTEN_PORT);
  return true;
}


void printNetworkAddress() {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println(F("NET:WiFi disconnected"));
    return;
  }
  Serial.print(F("NET:SSID "));
  Serial.println(WiFi.SSID());
  Serial.print(F("NET:IP "));
  Serial.println(WiFi.localIP());
  Serial.print(F("NET:UDP_PORT "));
  Serial.println(UDP_LISTEN_PORT);
}


void connectWiFi() {
  wifiPortal.begin();
}


void serviceWiFi() {
  wifiPortal.service();
  if (WiFi.status() == WL_CONNECTED) {
    if (!udpServerReady) {
      printNetworkAddress();
      startUdpServer();
    }
    return;
  }
  if (udpServerReady) {
    udp.close();
    udpServerReady = false;
    udpClientRegistered = false;
    // Do not execute a datagram queued on the previous network.
    portENTER_CRITICAL(&udpCommandMux);
    udpCommandPending = false;
    portEXIT_CRITICAL(&udpCommandMux);
    Serial.println(F("NET:UDP stopped; WiFi disconnected"));
  }
}


bool pingI2C(uint8_t address) {
  Wire.beginTransmission(address);
  return Wire.endTransmission() == 0U;
}


bool selectTCA(uint8_t channel) {
  if (channel > 7U || tcaAddress == 0U) {
    return false;
  }
  if (activeTcaChannel == channel) {
    return true;
  }
  Wire.beginTransmission(tcaAddress);
  Wire.write((uint8_t)(1U << channel));
  if (Wire.endTransmission() != 0U) {
    activeTcaChannel = NO_TCA_CHANNEL;
    return false;
  }
  activeTcaChannel = channel;
  return true;
}


void disableTCAChannels() {
  if (tcaAddress == 0U) {
    return;
  }
  Wire.beginTransmission(tcaAddress);
  Wire.write((uint8_t)0U);
  Wire.endTransmission();
  activeTcaChannel = NO_TCA_CHANNEL;
}


bool writeRegister(uint8_t address, uint8_t reg, uint8_t value) {
  Wire.beginTransmission(address);
  Wire.write(reg);
  Wire.write(value);
  return Wire.endTransmission() == 0U;
}


bool readBytes(uint8_t address, uint8_t reg, uint8_t* destination, uint8_t length) {
  Wire.beginTransmission(address);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0U) {
    return false;
  }
  uint8_t received = Wire.requestFrom(address, length);
  if (received != length) {
    while (Wire.available()) {
      Wire.read();
    }
    return false;
  }
  for (uint8_t index = 0U; index < length; index++) {
    if (!Wire.available()) {
      return false;
    }
    destination[index] = (uint8_t)Wire.read();
  }
  return true;
}


bool readRegister(uint8_t address, uint8_t reg, uint8_t* value) {
  return readBytes(address, reg, value, 1U);
}


int16_t signedWord(const uint8_t* bytes) {
  return (int16_t)(((uint16_t)bytes[0] << 8) | bytes[1]);
}


bool readRawSample(uint8_t sensorIndex, RawSample* sample) {
  SensorState& sensor = sensors[sensorIndex];
  if (sample == nullptr || !selectTCA(sensor.channel)) {
    sensor.readErrorCount++;
    return false;
  }
  uint8_t bytes[14];
  if (!readBytes(MPU_ADDR, REG_ACCEL_XOUT_H, bytes, sizeof(bytes))) {
    sensor.readErrorCount++;
    return false;
  }
  sample->ax = signedWord(bytes + 0);
  sample->ay = signedWord(bytes + 2);
  sample->az = signedWord(bytes + 4);
  sample->temperature = signedWord(bytes + 6);
  sample->gx = signedWord(bytes + 8);
  sample->gy = signedWord(bytes + 10);
  sample->gz = signedWord(bytes + 12);
  return true;
}


void printHexByte(uint8_t value) {
  Serial.print(F("0x"));
  if (value < 16U) {
    Serial.write('0');
  }
  Serial.print(value, HEX);
}


void scanMainI2CBus() {
  Serial.print(F("SCAN:MAIN"));
  uint8_t count = 0U;
  for (uint8_t address = 1U; address < 127U; address++) {
    if (pingI2C(address)) {
      Serial.write(' ');
      printHexByte(address);
      count++;
    }
  }
  if (count == 0U) {
    Serial.print(F(" none"));
  }
  Serial.println();
}


bool detectTCAAddress() {
  if (VERBOSE_I2C_SCAN) {
    scanMainI2CBus();
  }
  for (uint8_t address = TCA_ADDR_FIRST; address <= TCA_ADDR_LAST; address++) {
    if (pingI2C(address)) {
      tcaAddress = address;
      Serial.print(F("INIT:TCA9548A found at "));
      printHexByte(tcaAddress);
      Serial.println();
      return true;
    }
  }
  Serial.println(F("ERR:TCA_NOT_FOUND"));
  scanMainI2CBus();
  return false;
}


void addSensor(uint8_t channel) {
  if (sensorCount >= MAX_SENSORS) {
    return;
  }
  SensorState& sensor = sensors[sensorCount++];
  sensor.id = channel;
  sensor.channel = channel;
  sensor.whoAmI = 0U;
  sensor.ready = false;
  sensor.hasQuaternion = false;
  sensor.calibrationLoaded = false;
  sensor.quaternion = {1.0f, 0.0f, 0.0f, 0.0f};
  sensor.filter = MahonyFilter(
    FILTER_PROPORTIONAL_GAIN,
    FILTER_INTEGRAL_GAIN,
    FILTER_ACCEL_TOLERANCE_G
  );
  sensor.calibration = {
    CALIBRATION_MAGIC,
    0.0f,
    0.0f,
    0.0f,
    0.0f,
    0.0f,
    0.0f
  };
  sensor.readErrorCount = 0UL;
  sensor.filterUpdateCount = 0UL;
}


bool discoverSensors() {
  sensorCount = 0U;
  if (AUTO_DETECT_CHANNELS) {
    for (uint8_t channel = 0U; channel < 8U && sensorCount < MAX_SENSORS; channel++) {
      if (selectTCA(channel) && pingI2C(MPU_ADDR)) {
        addSensor(channel);
      }
    }
  } else {
    for (uint8_t index = 0U; index < MAX_SENSORS; index++) {
      uint8_t channel = SENSOR_CHANNELS[index];
      if (selectTCA(channel) && pingI2C(MPU_ADDR)) {
        addSensor(channel);
      }
    }
  }
  Serial.print(F("INIT:sensors detected "));
  Serial.println(sensorCount);
  return sensorCount > 0U;
}


void calibrationStorageKey(char* key, size_t size, uint8_t channel) {
  snprintf(key, size, "ch%u", channel);
}


bool loadStoredCalibration(uint8_t sensorIndex) {
  if (!calibrationStorageReady) {
    return false;
  }
  char key[8];
  calibrationStorageKey(key, sizeof(key), sensors[sensorIndex].channel);
  if (calibrationPreferences.getBytesLength(key) != sizeof(CalibrationData)) {
    return false;
  }
  CalibrationData stored;
  if (calibrationPreferences.getBytes(key, &stored, sizeof(stored)) != sizeof(stored) ||
      stored.magic != CALIBRATION_MAGIC) {
    return false;
  }
  sensors[sensorIndex].calibration = stored;
  sensors[sensorIndex].calibrationLoaded = true;
  return true;
}


void saveStoredCalibration(uint8_t sensorIndex) {
  if (!calibrationStorageReady) {
    return;
  }
  SensorState& sensor = sensors[sensorIndex];
  sensor.calibration.magic = CALIBRATION_MAGIC;
  char key[8];
  calibrationStorageKey(key, sizeof(key), sensor.channel);
  calibrationPreferences.putBytes(key, &sensor.calibration, sizeof(sensor.calibration));
  sensor.calibrationLoaded = true;
}


bool configureSensor(uint8_t sensorIndex) {
  SensorState& sensor = sensors[sensorIndex];
  if (!selectTCA(sensor.channel) ||
      !readRegister(MPU_ADDR, REG_WHO_AM_I, &sensor.whoAmI)) {
    Serial.print(F("ERR:S"));
    Serial.print(sensor.id);
    Serial.println(F(" WHOAMI_READ"));
    return false;
  }
  if (sensor.whoAmI != MPU6500_WHO_AM_I) {
    Serial.print(F("ERR:S"));
    Serial.print(sensor.id);
    Serial.print(F(" expected MPU6500 WHOAMI 0x70, got "));
    printHexByte(sensor.whoAmI);
    Serial.println();
    return false;
  }

  bool ok = writeRegister(MPU_ADDR, REG_PWR_MGMT_1, 0x80U);
  delay(100);
  ok = writeRegister(MPU_ADDR, REG_PWR_MGMT_1, 0x01U) && ok;
  delay(10);
  ok = writeRegister(MPU_ADDR, REG_PWR_MGMT_2, 0x00U) && ok;
  ok = writeRegister(MPU_ADDR, REG_USER_CTRL, 0x00U) && ok;
  ok = writeRegister(MPU_ADDR, REG_FIFO_EN, 0x00U) && ok;
  ok = writeRegister(MPU_ADDR, REG_INT_ENABLE, 0x00U) && ok;
  ok = writeRegister(MPU_ADDR, REG_SMPLRT_DIV, 0x09U) && ok;
  ok = writeRegister(MPU_ADDR, REG_CONFIG, 0x03U) && ok;
  ok = writeRegister(MPU_ADDR, REG_GYRO_CONFIG, 0x00U) && ok;
  ok = writeRegister(MPU_ADDR, REG_ACCEL_CONFIG, 0x00U) && ok;
  ok = writeRegister(MPU_ADDR, REG_ACCEL_CONFIG_2, 0x03U) && ok;

  uint8_t userControl = 0xFFU;
  uint8_t accelConfig = 0xFFU;
  uint8_t gyroConfig = 0xFFU;
  ok = readRegister(MPU_ADDR, REG_USER_CTRL, &userControl) && ok;
  ok = readRegister(MPU_ADDR, REG_ACCEL_CONFIG, &accelConfig) && ok;
  ok = readRegister(MPU_ADDR, REG_GYRO_CONFIG, &gyroConfig) && ok;
  ok = ((userControl & 0xC0U) == 0U) && ok;
  ok = ((accelConfig & 0x18U) == 0U) && ok;
  ok = ((gyroConfig & 0x18U) == 0U) && ok;
  if (!ok) {
    Serial.print(F("ERR:S"));
    Serial.print(sensor.id);
    Serial.println(F(" CONFIG"));
    return false;
  }

  loadStoredCalibration(sensorIndex);
  sensor.ready = true;
  RawSample raw;
  if (readRawSample(sensorIndex, &raw)) {
    const CalibrationData& calibration = sensor.calibration;
    sensor.filter.resetFromAcceleration(
      ((float)raw.ax - calibration.accelBiasX) / ACCEL_COUNTS_PER_G,
      ((float)raw.ay - calibration.accelBiasY) / ACCEL_COUNTS_PER_G,
      ((float)raw.az - calibration.accelBiasZ) / ACCEL_COUNTS_PER_G
    );
    sensor.quaternion = sensor.filter.quaternion();
    sensor.hasQuaternion = true;
  }

  Serial.print(F("OK:S"));
  Serial.print(sensor.id);
  Serial.print(F(" MPU6500 filter=Mahony calibration="));
  Serial.println(sensor.calibrationLoaded ? F("loaded") : F("defaults"));
  return true;
}


bool updateSensorFilter(uint8_t sensorIndex, float deltaTimeSeconds) {
  SensorState& sensor = sensors[sensorIndex];
  if (!sensor.ready) {
    return false;
  }
  RawSample raw;
  if (!readRawSample(sensorIndex, &raw)) {
    return false;
  }
  const CalibrationData& calibration = sensor.calibration;
  float axG = ((float)raw.ax - calibration.accelBiasX) / ACCEL_COUNTS_PER_G;
  float ayG = ((float)raw.ay - calibration.accelBiasY) / ACCEL_COUNTS_PER_G;
  float azG = ((float)raw.az - calibration.accelBiasZ) / ACCEL_COUNTS_PER_G;
  float gx = (
    ((float)raw.gx - calibration.gyroBiasX) /
    GYRO_COUNTS_PER_DEGREE_PER_SECOND
  ) * DEGREES_TO_RADIANS;
  float gy = (
    ((float)raw.gy - calibration.gyroBiasY) /
    GYRO_COUNTS_PER_DEGREE_PER_SECOND
  ) * DEGREES_TO_RADIANS;
  float gz = (
    ((float)raw.gz - calibration.gyroBiasZ) /
    GYRO_COUNTS_PER_DEGREE_PER_SECOND
  ) * DEGREES_TO_RADIANS;

  if (!sensor.hasQuaternion) {
    sensor.hasQuaternion = sensor.filter.resetFromAcceleration(axG, ayG, azG);
  } else {
    sensor.hasQuaternion = sensor.filter.update(gx, gy, gz, axG, ayG, azG, deltaTimeSeconds);
  }
  if (sensor.hasQuaternion) {
    sensor.quaternion = sensor.filter.quaternion();
    sensor.filterUpdateCount++;
  }
  return sensor.hasQuaternion;
}


void serviceFilters() {
  uint32_t nowUs = micros();
  uint32_t elapsedUs = (uint32_t)(nowUs - previousFilterUs);
  if (elapsedUs < FILTER_INTERVAL_US) {
    return;
  }
  float deltaTimeSeconds = (float)elapsedUs * 1.0e-6f;
  if (elapsedUs > FILTER_INTERVAL_US * 5UL) {
    deltaTimeSeconds = (float)FILTER_INTERVAL_US * 1.0e-6f;
  }
  previousFilterUs = nowUs;
  for (uint8_t index = 0U; index < sensorCount; index++) {
    updateSensorFilter(index, deltaTimeSeconds);
  }
}


bool collectCalibrationMeans(
  uint8_t sensorIndex,
  float* meanAx,
  float* meanAy,
  float* meanAz,
  float* meanGx,
  float* meanGy,
  float* meanGz
) {
  int64_t sumAx = 0;
  int64_t sumAy = 0;
  int64_t sumAz = 0;
  int64_t sumGx = 0;
  int64_t sumGy = 0;
  int64_t sumGz = 0;
  uint16_t accepted = 0U;
  for (uint16_t sampleIndex = 0U; sampleIndex < CALIBRATION_SAMPLES; sampleIndex++) {
    RawSample raw;
    if (readRawSample(sensorIndex, &raw)) {
      sumAx += raw.ax;
      sumAy += raw.ay;
      sumAz += raw.az;
      sumGx += raw.gx;
      sumGy += raw.gy;
      sumGz += raw.gz;
      accepted++;
    }
    delay(CALIBRATION_SAMPLE_DELAY_MS);
  }
  if (accepted < CALIBRATION_SAMPLES * 9U / 10U) {
    return false;
  }
  float denominator = (float)accepted;
  *meanAx = (float)sumAx / denominator;
  *meanAy = (float)sumAy / denominator;
  *meanAz = (float)sumAz / denominator;
  *meanGx = (float)sumGx / denominator;
  *meanGy = (float)sumGy / denominator;
  *meanGz = (float)sumGz / denominator;
  return true;
}


void printCalibration(uint8_t sensorIndex) {
  const SensorState& sensor = sensors[sensorIndex];
  const CalibrationData& value = sensor.calibration;
  Serial.print(F("CAL:S"));
  Serial.print(sensor.id);
  Serial.print(F(" accel_bias "));
  Serial.print(value.accelBiasX, 2);
  Serial.write(' ');
  Serial.print(value.accelBiasY, 2);
  Serial.write(' ');
  Serial.print(value.accelBiasZ, 2);
  Serial.print(F(" gyro_bias "));
  Serial.print(value.gyroBiasX, 2);
  Serial.write(' ');
  Serial.print(value.gyroBiasY, 2);
  Serial.write(' ');
  Serial.println(value.gyroBiasZ, 2);
}


void calibrateAllSensors(bool calibrateAccelerometers) {
  bool wasStreaming = streamEnabled;
  streamEnabled = false;
  Serial.println(
    calibrateAccelerometers
      ? F("CAL:ALL BEGIN keep flat Z-up and still")
      : F("CAL:GYRO BEGIN keep all sensors still")
  );

  for (uint8_t index = 0U; index < sensorCount; index++) {
    SensorState& sensor = sensors[index];
    if (!sensor.ready) {
      continue;
    }
    float meanAx;
    float meanAy;
    float meanAz;
    float meanGx;
    float meanGy;
    float meanGz;
    if (!collectCalibrationMeans(
          index,
          &meanAx,
          &meanAy,
          &meanAz,
          &meanGx,
          &meanGy,
          &meanGz
        )) {
      Serial.print(F("ERR:CAL S"));
      Serial.println(sensor.id);
      continue;
    }

    sensor.calibration.gyroBiasX = meanGx;
    sensor.calibration.gyroBiasY = meanGy;
    sensor.calibration.gyroBiasZ = meanGz;
    if (calibrateAccelerometers) {
      sensor.calibration.accelBiasX = meanAx;
      sensor.calibration.accelBiasY = meanAy;
      sensor.calibration.accelBiasZ = meanAz - ACCEL_COUNTS_PER_G;
    }
    saveStoredCalibration(index);
    printCalibration(index);

    sensor.filter.resetFromAcceleration(
      (meanAx - sensor.calibration.accelBiasX) / ACCEL_COUNTS_PER_G,
      (meanAy - sensor.calibration.accelBiasY) / ACCEL_COUNTS_PER_G,
      (meanAz - sensor.calibration.accelBiasZ) / ACCEL_COUNTS_PER_G
    );
    sensor.quaternion = sensor.filter.quaternion();
    sensor.hasQuaternion = true;
  }
  Serial.println(calibrateAccelerometers ? F("CAL:ALL DONE") : F("CAL:GYRO DONE"));
  previousFilterUs = micros();
  previousFrameMs = millis();
  streamEnabled = wasStreaming;
}


void resetTimingStats() {
  frameCounter = 0UL;
  lateFrameCount = 0U;
  lastFrameDurationUs = 0U;
  maxFrameDurationUs = 0U;
}


void printStatus() {
  Serial.print(F("STAT:sensors "));
  Serial.print(sensorCount);
  Serial.print(F(" streaming "));
  Serial.println(streamEnabled ? F("yes") : F("no"));
  Serial.print(F("STAT:rate_hz "));
  Serial.print(streamRateHz);
  Serial.print(F(" filter_hz "));
  Serial.print(FILTER_RATE_HZ);
  Serial.print(F(" frames "));
  Serial.println(frameCounter);
  for (uint8_t index = 0U; index < sensorCount; index++) {
    const SensorState& sensor = sensors[index];
    Serial.print(F("STAT:S"));
    Serial.print(sensor.id);
    Serial.print(F(" ready "));
    Serial.print(sensor.ready ? 1U : 0U);
    Serial.print(F(" read_errors "));
    Serial.print(sensor.readErrorCount);
    Serial.print(F(" updates "));
    Serial.println(sensor.filterUpdateCount);
  }
}


void sendStatusUdp(const IPAddress& remoteIp, uint16_t remotePort) {
  char response[UDP_PACKET_BUFFER_SIZE];
  int written = snprintf(
    response,
    sizeof(response),
    "STATUS sensors=%u streaming=%u rate_hz=%u frame_ms=%lu frames=%lu "
    "late=%u last_us=%u max_us=%u packet_size=0 wifi=%u udp_port=%u "
    "filter=mahony filter_hz=%u\n",
    sensorCount,
    streamEnabled ? 1U : 0U,
    streamRateHz,
    (unsigned long)frameIntervalMs,
    (unsigned long)frameCounter,
    lateFrameCount,
    lastFrameDurationUs,
    maxFrameDurationUs,
    WiFi.status() == WL_CONNECTED ? 1U : 0U,
    UDP_LISTEN_PORT,
    FILTER_RATE_HZ
  );
  if (written < 0 || (size_t)written >= sizeof(response)) {
    return;
  }
  size_t used = (size_t)written;
  for (uint8_t index = 0U; index < sensorCount; index++) {
    const SensorState& sensor = sensors[index];
    size_t remaining = sizeof(response) - used;
    written = snprintf(
      response + used,
      remaining,
      "SENSOR id=%u channel=%u dmp=0 fifo_resets=0 filter=1 ready=%u "
      "whoami=0x%02X read_errors=%lu calibrated=%u\n",
      sensor.id,
      sensor.channel,
      sensor.ready ? 1U : 0U,
      sensor.whoAmI,
      (unsigned long)sensor.readErrorCount,
      sensor.calibrationLoaded ? 1U : 0U
    );
    if (written < 0 || (size_t)written >= remaining) {
      return;
    }
    used += (size_t)written;
  }
  sendUdpReply(remoteIp, remotePort, response);
}


void handleCommand(char* command, const IPAddress* remoteIp, uint16_t remotePort) {
  trimCommand(command);
  if (command[0] == '\0') {
    return;
  }
  bool fromUdp = remoteIp != nullptr && remotePort != 0U;
  if (fromUdp) {
    udpClientIp = *remoteIp;
    udpClientPort = remotePort;
    udpClientRegistered = true;
  }

  if (strcasecmp(command, "HELLO") == 0) {
    char response[96];
    snprintf(
      response,
      sizeof(response),
      "ACK HELLO sensors=%u rate_hz=%u udp_port=%u\n",
      sensorCount,
      streamRateHz,
      UDP_LISTEN_PORT
    );
    Serial.println(F("ACK:HELLO"));
    if (fromUdp) {
      sendUdpReply(*remoteIp, remotePort, response);
    }
  } else if (strcasecmp(command, "PING") == 0) {
    Serial.println(F("ACK:PING"));
    if (fromUdp) {
      sendUdpReply(*remoteIp, remotePort, "PONG\n");
    }
  } else if (strcasecmp(command, "START") == 0) {
    resetTimingStats();
    previousFrameMs = millis();
    streamEnabled = true;
    Serial.println(F("ACK:START"));
    if (fromUdp) {
      sendUdpReply(*remoteIp, remotePort, "ACK START\n");
    }
  } else if (strcasecmp(command, "STOP") == 0) {
    streamEnabled = false;
    digitalWrite(LED_PIN, LOW);
    Serial.println(F("ACK:STOP"));
    if (fromUdp) {
      sendUdpReply(*remoteIp, remotePort, "ACK STOP\n");
    }
  } else if (strcasecmp(command, "CALIB_GYRO") == 0) {
    Serial.println(F("ACK:CALIB_GYRO BEGIN"));
    if (fromUdp) {
      sendUdpReply(*remoteIp, remotePort, "ACK CALIB_GYRO BEGIN keep_still=1\n");
    }
    calibrateAllSensors(false);
    Serial.println(F("ACK:CALIB_GYRO DONE"));
    if (fromUdp) {
      sendUdpReply(*remoteIp, remotePort, "ACK CALIB_GYRO DONE\n");
    }
  } else if (strcasecmp(command, "CALIB") == 0) {
    Serial.println(F("ACK:CALIB BEGIN flat_z_up=1"));
    if (fromUdp) {
      sendUdpReply(*remoteIp, remotePort, "ACK CALIB BEGIN flat_z_up=1 keep_still=1\n");
    }
    calibrateAllSensors(true);
    Serial.println(F("ACK:CALIB DONE"));
    if (fromUdp) {
      sendUdpReply(*remoteIp, remotePort, "ACK CALIB DONE\n");
    }
  } else if (strcasecmp(command, "STATUS") == 0 ||
             strcasecmp(command, "GET_CONFIG") == 0) {
    printStatus();
    if (fromUdp) {
      sendStatusUdp(*remoteIp, remotePort);
    }
  } else if (strncasecmp(command, "SET_RATE", 8U) == 0) {
    const char* valueText = command + 8U;
    while (*valueText == ' ' || *valueText == '\t') {
      valueText++;
    }
    char* end = nullptr;
    long requestedRate = strtol(valueText, &end, 10);
    bool validNumber = end != valueText && *end == '\0';
    if (!validNumber ||
        requestedRate < (long)MIN_STREAM_RATE_HZ ||
        requestedRate > (long)MAX_STREAM_RATE_HZ) {
      char response[80];
      snprintf(
        response,
        sizeof(response),
        "ERR RATE expected=%u..%u\n",
        MIN_STREAM_RATE_HZ,
        MAX_STREAM_RATE_HZ
      );
      if (fromUdp) {
        sendUdpReply(*remoteIp, remotePort, response);
      }
      return;
    }
    streamRateHz = (uint16_t)requestedRate;
    frameIntervalMs = (1000UL + streamRateHz / 2U) / streamRateHz;
    if (frameIntervalMs == 0UL) {
      frameIntervalMs = 1UL;
    }
    previousFrameMs = millis();
    char response[80];
    snprintf(
      response,
      sizeof(response),
      "ACK SET_RATE rate_hz=%u frame_ms=%lu\n",
      streamRateHz,
      (unsigned long)frameIntervalMs
    );
    Serial.print(F("ACK:SET_RATE "));
    Serial.println(streamRateHz);
    if (fromUdp) {
      sendUdpReply(*remoteIp, remotePort, response);
    }
  } else {
    char response[CMD_BUFFER_SIZE + 24U];
    snprintf(response, sizeof(response), "ERR UNKNOWN %s\n", command);
    Serial.print(F("ERR:UNKNOWN "));
    Serial.println(command);
    if (fromUdp) {
      sendUdpReply(*remoteIp, remotePort, response);
    }
  }
}


void processCommands() {
  while (Serial.available() > 0) {
    char value = Serial.read();
    if (value == '\n' || value == '\r') {
      cmdBuffer[cmdIdx] = '\0';
      handleCommand(cmdBuffer, nullptr, 0U);
      cmdIdx = 0U;
      cmdBuffer[0] = '\0';
    } else if (cmdIdx < CMD_BUFFER_SIZE - 1U) {
      cmdBuffer[cmdIdx++] = value;
    } else {
      cmdIdx = 0U;
      cmdBuffer[0] = '\0';
      Serial.println(F("ERR:CMD_TOO_LONG"));
    }
  }
}


void processUdpCommands() {
  PendingUdpCommand command;
  bool hasCommand = false;
  portENTER_CRITICAL(&udpCommandMux);
  if (udpCommandPending) {
    memcpy(&command, &pendingUdpCommand, sizeof(command));
    udpCommandPending = false;
    hasCommand = true;
  }
  portEXIT_CRITICAL(&udpCommandMux);
  if (hasCommand) {
    handleCommand(command.text, &command.remoteIp, command.remotePort);
  }
}


void printFourDigits(uint16_t value) {
  if (value < 1000U) {
    Serial.write('0');
  }
  if (value < 100U) {
    Serial.write('0');
  }
  if (value < 10U) {
    Serial.write('0');
  }
  Serial.print(value);
}


void printQuatComponent(float value) {
  if (value < 0.0f) {
    Serial.write('-');
    value = -value;
  }
  if (value > 1.0f) {
    value = 1.0f;
  }
  uint16_t scaled = (uint16_t)(value * (float)QUAT_PRINT_SCALE + 0.5f);
  if (scaled > QUAT_PRINT_SCALE) {
    scaled = QUAT_PRINT_SCALE;
  }
  Serial.print((uint8_t)(scaled / QUAT_PRINT_SCALE));
  Serial.write('.');
  printFourDigits((uint16_t)(scaled % QUAT_PRINT_SCALE));
}


void sendSensorQuaternion(const SensorState& sensor) {
  Serial.print(F("quat "));
  Serial.print(sensor.id);
  Serial.write(' ');
  printQuatComponent(sensor.quaternion.w);
  Serial.write(' ');
  printQuatComponent(sensor.quaternion.x);
  Serial.write(' ');
  printQuatComponent(sensor.quaternion.y);
  Serial.write(' ');
  printQuatComponent(sensor.quaternion.z);
  Serial.write('\n');
}


void sendAllQuaternions() {
  uint8_t quaternionCount = 0U;
  for (uint8_t index = 0U; index < sensorCount; index++) {
    if (sensors[index].ready && sensors[index].hasQuaternion) {
      quaternionCount++;
    }
  }
  if (!udpServerReady || !udpClientRegistered || udpClientPort == 0U) {
    return;
  }

  char datagram[UDP_PACKET_BUFFER_SIZE];
  int written = snprintf(
    datagram,
    sizeof(datagram),
    "FRAME %lu %lu %u\n",
    (unsigned long)frameCounter,
    (unsigned long)millis(),
    quaternionCount
  );
  if (written < 0 || (size_t)written >= sizeof(datagram)) {
    return;
  }
  size_t used = (size_t)written;
  for (uint8_t index = 0U; index < sensorCount; index++) {
    const SensorState& sensor = sensors[index];
    if (!sensor.ready || !sensor.hasQuaternion) {
      continue;
    }
    size_t remaining = sizeof(datagram) - used;
    written = snprintf(
      datagram + used,
      remaining,
      "Q %u %.6f %.6f %.6f %.6f\n",
      sensor.id,
      (double)sensor.quaternion.w,
      (double)sensor.quaternion.x,
      (double)sensor.quaternion.y,
      (double)sensor.quaternion.z
    );
    if (written < 0 || (size_t)written >= remaining) {
      Serial.println(F("ERR:UDP_FRAME_TOO_LARGE"));
      return;
    }
    used += (size_t)written;
  }
  udp.writeTo((const uint8_t*)datagram, used, udpClientIp, udpClientPort);
}


void runStreamingFrame() {
  uint32_t startedUs = micros();
  sendAllQuaternions();
  frameCounter++;
  uint32_t elapsedUs = micros() - startedUs;
  lastFrameDurationUs = elapsedUs > 65535UL ? 65535U : (uint16_t)elapsedUs;
  if (lastFrameDurationUs > maxFrameDurationUs) {
    maxFrameDurationUs = lastFrameDurationUs;
  }
  blinkState = !blinkState;
  digitalWrite(LED_PIN, blinkState);
}


void serviceFrameScheduler() {
  if (!streamEnabled) {
    return;
  }
  uint32_t currentMillis = millis();
  uint32_t elapsedMs = (uint32_t)(currentMillis - previousFrameMs);
  if (elapsedMs < frameIntervalMs) {
    return;
  }
  if (elapsedMs > frameIntervalMs && lateFrameCount < 65535U) {
    lateFrameCount++;
  }
  previousFrameMs += frameIntervalMs;
  if ((uint32_t)(currentMillis - previousFrameMs) >= frameIntervalMs) {
    previousFrameMs = currentMillis;
  }
  runStreamingFrame();
}


void setupSensors() {
  Serial.println(F("INIT:MPU6500 xN via TCA, filter=Mahony"));
  if (!detectTCAAddress()) {
    return;
  }
  disableTCAChannels();
  if (!discoverSensors()) {
    return;
  }
  uint8_t readyCount = 0U;
  for (uint8_t index = 0U; index < sensorCount; index++) {
    if (configureSensor(index)) {
      readyCount++;
    }
  }
  Wire.setClock(I2C_RUN_CLOCK_HZ);
  previousFilterUs = micros();
  Serial.print(F("RDY:sensors_ready "));
  Serial.println(readyCount);
}


void setup() {
  pinMode(LED_PIN, OUTPUT);
  Serial.begin(BAUD_RATE);
  uint32_t startedAt = millis();
  while (!Serial && (uint32_t)(millis() - startedAt) < SERIAL_WAIT_MS) {
  }

  calibrationStorageReady = calibrationPreferences.begin(
    CALIBRATION_NVS_NAMESPACE,
    false
  );
  Serial.println(
    calibrationStorageReady
      ? F("INIT:stored MPU6500 calibration ready")
      : F("WARN:stored MPU6500 calibration unavailable")
  );

#if defined(SDA) && defined(SCL)
  pinMode(SDA, INPUT_PULLUP);
  pinMode(SCL, INPUT_PULLUP);
#endif
  Wire.begin();
  Wire.setClock(I2C_INIT_CLOCK_HZ);
#if defined(WIRE_HAS_TIMEOUT)
  Wire.setWireTimeout(I2C_TIMEOUT_US, true);
#endif

  setupSensors();
  connectWiFi();
  Serial.println(F("RDY:send START to stream sensor quaternions"));
}


void loop() {
  processCommands();
  processUdpCommands();
  serviceWiFi();
  serviceFilters();
  serviceFrameScheduler();
}
