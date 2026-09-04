/*
  Raw MPU-6500 gravity diagnostic for ESP32 + TCA9548A.

  No DMP firmware and no calibration offsets are used. Sensor registers are
  read directly and every I2C transaction length is checked.

  UDP packet:
    RAW_FRAME <sequence> <millis> <sensor_count>
    S <tca_channel> <ax> <ay> <az> <temperature> <gx> <gy> <gz>

  Commands: HELLO, START, STOP, STATUS, PING
*/

#include <Arduino.h>
#include <AsyncUDP.h>
#include <WiFi.h>
#include <Wire.h>
#include <string.h>
#include <strings.h>

#include "wifi_secrets.h"


static const uint32_t BAUD_RATE = 115200UL;
static const uint32_t SERIAL_WAIT_MS = 2000UL;
static const uint32_t I2C_INIT_CLOCK_HZ = 100000UL;
static const uint32_t I2C_RUN_CLOCK_HZ = 400000UL;
static const uint32_t I2C_TIMEOUT_US = 3000UL;
static const uint16_t UDP_PORT = 4210U;
static const uint16_t STREAM_RATE_HZ = 20U;
static const uint32_t FRAME_INTERVAL_MS = 1000UL / STREAM_RATE_HZ;
static const uint8_t MAX_SENSORS = 5U;
static const uint8_t MPU_ADDRESS = 0x68U;
static const uint8_t TCA_ADDRESS_FIRST = 0x70U;
static const uint8_t TCA_ADDRESS_LAST = 0x77U;
static const uint8_t NO_TCA_CHANNEL = 0xFFU;
static const size_t COMMAND_BUFFER_SIZE = 48U;
static const size_t DATAGRAM_BUFFER_SIZE = 768U;

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
static const uint8_t MPU6500_WHO_AM_I = 0x70U;


struct SensorSample {
  uint8_t id;
  uint8_t channel;
  uint8_t whoAmI;
  bool ready;
  bool valid;
  int16_t ax;
  int16_t ay;
  int16_t az;
  int16_t temperature;
  int16_t gx;
  int16_t gy;
  int16_t gz;
  uint32_t readErrors;
};


AsyncUDP udp;
SensorSample sensors[MAX_SENSORS];
uint8_t sensorCount = 0U;
uint8_t tcaAddress = 0U;
uint8_t activeTcaChannel = NO_TCA_CHANNEL;

bool udpReady = false;
bool clientRegistered = false;
bool streaming = false;
IPAddress clientIp;
uint16_t clientPort = 0U;
uint32_t frameSequence = 0UL;
uint32_t previousFrameMs = 0UL;

struct PendingCommand {
  char text[COMMAND_BUFFER_SIZE];
  IPAddress remoteIp;
  uint16_t remotePort;
};

PendingCommand pendingCommand;
volatile bool commandPending = false;
portMUX_TYPE commandMux = portMUX_INITIALIZER_UNLOCKED;


bool pingI2C(uint8_t address) {
  Wire.beginTransmission(address);
  return Wire.endTransmission() == 0U;
}


bool selectTCA(uint8_t channel) {
  if (tcaAddress == 0U || channel > 7U) {
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


bool discoverHardware() {
  for (uint8_t address = TCA_ADDRESS_FIRST; address <= TCA_ADDRESS_LAST; address++) {
    if (pingI2C(address)) {
      tcaAddress = address;
      break;
    }
  }
  if (tcaAddress == 0U) {
    Serial.println(F("ERR:NO_TCA9548A"));
    return false;
  }

  for (uint8_t channel = 0U; channel < 8U && sensorCount < MAX_SENSORS; channel++) {
    if (!selectTCA(channel) || !pingI2C(MPU_ADDRESS)) {
      continue;
    }
    SensorSample& sample = sensors[sensorCount++];
    sample.id = channel;
    sample.channel = channel;
    sample.whoAmI = 0U;
    sample.ready = false;
    sample.valid = false;
    sample.ax = sample.ay = sample.az = 0;
    sample.temperature = 0;
    sample.gx = sample.gy = sample.gz = 0;
    sample.readErrors = 0UL;
  }

  Serial.print(F("INIT:TCA 0x"));
  Serial.println(tcaAddress, HEX);
  Serial.print(F("INIT:sensors "));
  Serial.println(sensorCount);
  return sensorCount > 0U;
}


bool initializeSensor(uint8_t index) {
  SensorSample& sample = sensors[index];
  if (!selectTCA(sample.channel) ||
      !readRegister(MPU_ADDRESS, REG_WHO_AM_I, &sample.whoAmI)) {
    Serial.print(F("ERR:WHOAMI channel="));
    Serial.println(sample.channel);
    return false;
  }
  if (sample.whoAmI != MPU6500_WHO_AM_I) {
    Serial.print(F("ERR:NOT_MPU6500 channel="));
    Serial.print(sample.channel);
    Serial.print(F(" whoami=0x"));
    Serial.println(sample.whoAmI, HEX);
    return false;
  }

  bool ok = writeRegister(MPU_ADDRESS, REG_PWR_MGMT_1, 0x80U);
  delay(100);
  ok = writeRegister(MPU_ADDRESS, REG_PWR_MGMT_1, 0x01U) && ok;
  delay(10);
  ok = writeRegister(MPU_ADDRESS, REG_PWR_MGMT_2, 0x00U) && ok;
  ok = writeRegister(MPU_ADDRESS, REG_USER_CTRL, 0x00U) && ok;
  ok = writeRegister(MPU_ADDRESS, REG_FIFO_EN, 0x00U) && ok;
  ok = writeRegister(MPU_ADDRESS, REG_INT_ENABLE, 0x00U) && ok;
  ok = writeRegister(MPU_ADDRESS, REG_SMPLRT_DIV, 0x04U) && ok;
  ok = writeRegister(MPU_ADDRESS, REG_CONFIG, 0x03U) && ok;
  ok = writeRegister(MPU_ADDRESS, REG_GYRO_CONFIG, 0x00U) && ok;
  ok = writeRegister(MPU_ADDRESS, REG_ACCEL_CONFIG, 0x00U) && ok;
  ok = writeRegister(MPU_ADDRESS, REG_ACCEL_CONFIG_2, 0x03U) && ok;

  uint8_t userControl = 0xFFU;
  uint8_t accelConfig = 0xFFU;
  uint8_t gyroConfig = 0xFFU;
  ok = readRegister(MPU_ADDRESS, REG_USER_CTRL, &userControl) && ok;
  ok = readRegister(MPU_ADDRESS, REG_ACCEL_CONFIG, &accelConfig) && ok;
  ok = readRegister(MPU_ADDRESS, REG_GYRO_CONFIG, &gyroConfig) && ok;
  ok = (userControl & 0xC0U) == 0U && ok;
  ok = (accelConfig & 0x18U) == 0U && ok;
  ok = (gyroConfig & 0x18U) == 0U && ok;

  sample.ready = ok;
  Serial.print(ok ? F("OK:RAW channel=") : F("ERR:RAW_CONFIG channel="));
  Serial.print(sample.channel);
  Serial.print(F(" whoami=0x"));
  Serial.println(sample.whoAmI, HEX);
  return ok;
}


bool readSensor(uint8_t index) {
  SensorSample& sample = sensors[index];
  if (!sample.ready || !selectTCA(sample.channel)) {
    sample.valid = false;
    sample.readErrors++;
    return false;
  }
  uint8_t bytes[14];
  if (!readBytes(MPU_ADDRESS, REG_ACCEL_XOUT_H, bytes, sizeof(bytes))) {
    sample.valid = false;
    sample.readErrors++;
    return false;
  }
  sample.ax = signedWord(bytes + 0);
  sample.ay = signedWord(bytes + 2);
  sample.az = signedWord(bytes + 4);
  sample.temperature = signedWord(bytes + 6);
  sample.gx = signedWord(bytes + 8);
  sample.gy = signedWord(bytes + 10);
  sample.gz = signedWord(bytes + 12);
  sample.valid = true;
  return true;
}


void sendReply(const IPAddress& remoteIp, uint16_t remotePort, const char* text) {
  if (udpReady && remotePort != 0U) {
    udp.writeTo((const uint8_t*)text, strlen(text), remoteIp, remotePort);
  }
}


uint8_t readySensorCount() {
  uint8_t count = 0U;
  for (uint8_t index = 0U; index < sensorCount; index++) {
    if (sensors[index].ready) {
      count++;
    }
  }
  return count;
}


void sendStatus(const IPAddress& remoteIp, uint16_t remotePort) {
  char response[640];
  int written = snprintf(
    response,
    sizeof(response),
    "STATUS raw_benchmark=1 sensors=%u ready=%u rate_hz=%u streaming=%u\n",
    sensorCount,
    readySensorCount(),
    STREAM_RATE_HZ,
    streaming ? 1U : 0U
  );
  if (written < 0 || (size_t)written >= sizeof(response)) {
    return;
  }
  size_t used = (size_t)written;
  for (uint8_t index = 0U; index < sensorCount; index++) {
    size_t remaining = sizeof(response) - used;
    written = snprintf(
      response + used,
      remaining,
      "SENSOR id=%u ready=%u whoami=0x%02X read_errors=%lu\n",
      sensors[index].id,
      sensors[index].ready ? 1U : 0U,
      sensors[index].whoAmI,
      (unsigned long)sensors[index].readErrors
    );
    if (written < 0 || (size_t)written >= remaining) {
      return;
    }
    used += (size_t)written;
  }
  sendReply(remoteIp, remotePort, response);
}


void trimCommand(char* command) {
  size_t length = strlen(command);
  while (length > 0U) {
    char value = command[length - 1U];
    if (value != '\r' && value != '\n' && value != ' ' && value != '\t') {
      break;
    }
    command[--length] = '\0';
  }
}


void handleCommand(PendingCommand& command) {
  trimCommand(command.text);
  clientIp = command.remoteIp;
  clientPort = command.remotePort;
  clientRegistered = true;

  if (strcasecmp(command.text, "HELLO") == 0) {
    char response[112];
    snprintf(
      response,
      sizeof(response),
      "ACK HELLO raw_benchmark=1 sensors=%u ready=%u rate_hz=%u\n",
      sensorCount,
      readySensorCount(),
      STREAM_RATE_HZ
    );
    sendReply(command.remoteIp, command.remotePort, response);
  } else if (strcasecmp(command.text, "START") == 0) {
    frameSequence = 0UL;
    previousFrameMs = millis();
    streaming = true;
    sendReply(command.remoteIp, command.remotePort, "ACK START\n");
  } else if (strcasecmp(command.text, "STOP") == 0) {
    streaming = false;
    sendReply(command.remoteIp, command.remotePort, "ACK STOP\n");
  } else if (strcasecmp(command.text, "STATUS") == 0) {
    sendStatus(command.remoteIp, command.remotePort);
  } else if (strcasecmp(command.text, "PING") == 0) {
    sendReply(command.remoteIp, command.remotePort, "PONG\n");
  } else {
    sendReply(command.remoteIp, command.remotePort, "ERR UNKNOWN_COMMAND\n");
  }
}


void processPendingCommand() {
  PendingCommand command;
  bool available = false;
  portENTER_CRITICAL(&commandMux);
  if (commandPending) {
    memcpy(&command, &pendingCommand, sizeof(command));
    commandPending = false;
    available = true;
  }
  portEXIT_CRITICAL(&commandMux);
  if (available) {
    handleCommand(command);
  }
}


bool startUdp() {
  if (!udp.listen(UDP_PORT)) {
    Serial.println(F("ERR:UDP_LISTEN"));
    return false;
  }
  udp.onPacket([](AsyncUDPPacket packet) {
    if (packet.length() == 0U || packet.length() >= COMMAND_BUFFER_SIZE) {
      packet.print("ERR BAD_COMMAND\n");
      return;
    }
    portENTER_CRITICAL(&commandMux);
    if (!commandPending) {
      memcpy(pendingCommand.text, packet.data(), packet.length());
      pendingCommand.text[packet.length()] = '\0';
      pendingCommand.remoteIp = packet.remoteIP();
      pendingCommand.remotePort = packet.remotePort();
      commandPending = true;
    }
    portEXIT_CRITICAL(&commandMux);
  });
  udpReady = true;
  return true;
}


void sendFrame() {
  if (!udpReady || !clientRegistered || clientPort == 0U) {
    return;
  }
  uint8_t validCount = 0U;
  for (uint8_t index = 0U; index < sensorCount; index++) {
    if (readSensor(index)) {
      validCount++;
    }
  }

  char datagram[DATAGRAM_BUFFER_SIZE];
  int written = snprintf(
    datagram,
    sizeof(datagram),
    "RAW_FRAME %lu %lu %u\n",
    (unsigned long)frameSequence,
    (unsigned long)millis(),
    validCount
  );
  if (written < 0 || (size_t)written >= sizeof(datagram)) {
    return;
  }
  size_t used = (size_t)written;
  for (uint8_t index = 0U; index < sensorCount; index++) {
    const SensorSample& sample = sensors[index];
    if (!sample.valid) {
      continue;
    }
    size_t remaining = sizeof(datagram) - used;
    written = snprintf(
      datagram + used,
      remaining,
      "S %u %d %d %d %d %d %d %d\n",
      sample.id,
      sample.ax,
      sample.ay,
      sample.az,
      sample.temperature,
      sample.gx,
      sample.gy,
      sample.gz
    );
    if (written < 0 || (size_t)written >= remaining) {
      return;
    }
    used += (size_t)written;
  }
  udp.writeTo((const uint8_t*)datagram, used, clientIp, clientPort);
}


void serviceFrame() {
  if (!streaming) {
    return;
  }
  uint32_t now = millis();
  if ((uint32_t)(now - previousFrameMs) < FRAME_INTERVAL_MS) {
    return;
  }
  previousFrameMs = now;
  sendFrame();
  frameSequence++;
}


void setup() {
  Serial.begin(BAUD_RATE);
  uint32_t serialStarted = millis();
  while (!Serial && (uint32_t)(millis() - serialStarted) < SERIAL_WAIT_MS) {
  }

  Wire.begin();
  Wire.setClock(I2C_INIT_CLOCK_HZ);
#if defined(WIRE_HAS_TIMEOUT)
  Wire.setWireTimeout(I2C_TIMEOUT_US, true);
#endif

  if (discoverHardware()) {
    for (uint8_t index = 0U; index < sensorCount; index++) {
      initializeSensor(index);
    }
  }
  Wire.setClock(I2C_RUN_CLOCK_HZ);

  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print(F("NET:connecting"));
  while (WiFi.status() != WL_CONNECTED) {
    delay(250);
    Serial.write('.');
  }
  Serial.println();
  Serial.print(F("NET:IP "));
  Serial.println(WiFi.localIP());
  startUdp();
  Serial.println(F("RDY:raw MPU6500; send HELLO then START"));
}


void loop() {
  processPendingCommand();
  serviceFrame();
}
