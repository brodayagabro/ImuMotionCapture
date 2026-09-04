/*
  MPU6050 gravity/DMP diagnostic for ESP32 + TCA9548A.

  This sketch deliberately applies zero accelerometer and gyro offsets. It
  streams direct register readings together with the DMP quaternion so the
  host can distinguish an accelerometer problem from a DMP problem.

  UDP packet:
    GRAVITY_FRAME <sequence> <millis> <sensor_count>
    S <tca_channel> <ax> <ay> <az> <gx> <gy> <gz> <qw> <qx> <qy> <qz>

  Commands: HELLO, START, STOP, STATUS, PING
*/

#include <Arduino.h>
#include <AsyncUDP.h>
#include <WiFi.h>
#include <Wire.h>
#include <string.h>
#include <strings.h>

#include "I2Cdev.h"
#include "MPU6050_6Axis_MotionApps20.h"
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
static const uint8_t MPU_WHO_AM_I_REGISTER = 0x75U;
static const uint8_t TCA_ADDRESS_FIRST = 0x70U;
static const uint8_t TCA_ADDRESS_LAST = 0x77U;
static const uint8_t NO_TCA_CHANNEL = 0xFFU;
static const uint16_t FIFO_OVERFLOW_BYTES = 1024U;
static const uint8_t MAX_FIFO_PACKETS = 20U;
static const size_t COMMAND_BUFFER_SIZE = 48U;
static const size_t DATAGRAM_BUFFER_SIZE = 1024U;


struct SensorSample {
  uint8_t id;
  uint8_t channel;
  bool ready;
  bool valid;
  uint8_t dmpStatus;
  uint8_t whoAmI;
  Quaternion quaternion;
  int16_t ax;
  int16_t ay;
  int16_t az;
  int16_t gx;
  int16_t gy;
  int16_t gz;
  uint16_t fifoResets;
};


MPU6050 mpu(MPU_ADDRESS);
AsyncUDP udp;
SensorSample sensors[MAX_SENSORS];
uint8_t sensorCount = 0U;
uint8_t tcaAddress = 0U;
uint8_t activeTcaChannel = NO_TCA_CHANNEL;
uint16_t dmpPacketSize = 0U;
uint8_t fifoBuffer[64];

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
  return Wire.endTransmission() == 0;
}


bool readRegister(uint8_t address, uint8_t reg, uint8_t* value) {
  Wire.beginTransmission(address);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0U) {
    return false;
  }
  if (Wire.requestFrom(address, (uint8_t)1U) != 1U) {
    return false;
  }
  *value = Wire.read();
  return true;
}


bool isSupportedWhoAmI(uint8_t value) {
  return value == 0x68U || value == 0x70U || value == 0x71U || value == 0x73U;
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

  sensorCount = 0U;
  for (uint8_t channel = 0U; channel < 8U && sensorCount < MAX_SENSORS; channel++) {
    if (!selectTCA(channel) || !pingI2C(MPU_ADDRESS)) {
      continue;
    }
    SensorSample& sample = sensors[sensorCount++];
    sample.id = channel;
    sample.channel = channel;
    sample.ready = false;
    sample.valid = false;
    sample.dmpStatus = 0xFFU;
    sample.whoAmI = 0U;
    sample.quaternion = Quaternion(1.0f, 0.0f, 0.0f, 0.0f);
    sample.ax = sample.ay = sample.az = 0;
    sample.gx = sample.gy = sample.gz = 0;
    sample.fifoResets = 0U;
  }

  Serial.print(F("INIT:TCA 0x"));
  Serial.println(tcaAddress, HEX);
  Serial.print(F("INIT:sensors "));
  Serial.println(sensorCount);
  return sensorCount > 0U;
}


bool initializeSensor(uint8_t index) {
  SensorSample& sample = sensors[index];
  sample.ready = false;
  sample.valid = false;
  sample.dmpStatus = 0xFEU;
  if (!selectTCA(sample.channel)) {
    return false;
  }

  mpu.initialize();
  selectTCA(sample.channel);
  readRegister(MPU_ADDRESS, MPU_WHO_AM_I_REGISTER, &sample.whoAmI);
  bool libraryConnection = mpu.testConnection();
  if (!libraryConnection && !isSupportedWhoAmI(sample.whoAmI)) {
    sample.dmpStatus = 0xFDU;
    Serial.print(F("ERR:MPU_CONNECTION channel="));
    Serial.println(sample.channel);
    return false;
  }
  if (!libraryConnection) {
    Serial.print(F("WARN:MPU_COMPATIBLE_WHOAMI channel="));
    Serial.print(sample.channel);
    Serial.print(F(" whoami=0x"));
    Serial.println(sample.whoAmI, HEX);
  }

  selectTCA(sample.channel);
  uint8_t status = mpu.dmpInitialize();
  sample.dmpStatus = status;
  if (status != 0U) {
    Serial.print(F("ERR:DMP_INIT channel="));
    Serial.print(sample.channel);
    Serial.print(F(" status="));
    Serial.println(status);
    return false;
  }

  // The benchmark must expose the native errors instead of hiding them in
  // offsets computed in one arbitrary orientation.
  mpu.setXAccelOffset(0);
  mpu.setYAccelOffset(0);
  mpu.setZAccelOffset(0);
  mpu.setXGyroOffset(0);
  mpu.setYGyroOffset(0);
  mpu.setZGyroOffset(0);
  mpu.setFullScaleAccelRange(MPU6050_ACCEL_FS_2);

  uint16_t packetSize = mpu.dmpGetFIFOPacketSize();
  if (packetSize == 0U || packetSize > sizeof(fifoBuffer)) {
    Serial.print(F("ERR:DMP_PACKET_SIZE channel="));
    Serial.println(sample.channel);
    return false;
  }
  if (dmpPacketSize == 0U) {
    dmpPacketSize = packetSize;
  } else if (packetSize != dmpPacketSize) {
    Serial.println(F("ERR:DMP_PACKET_SIZE_MISMATCH"));
    return false;
  }

  mpu.setDMPEnabled(true);
  mpu.resetFIFO();
  mpu.getIntStatus();
  sample.ready = true;
  sample.valid = false;
  Serial.print(F("OK:MPU channel="));
  Serial.println(sample.channel);
  return true;
}


void resetAllFifos() {
  for (uint8_t index = 0U; index < sensorCount; index++) {
    SensorSample& sample = sensors[index];
    if (!sample.ready || !selectTCA(sample.channel)) {
      continue;
    }
    mpu.resetFIFO();
    mpu.getIntStatus();
    sample.valid = false;
  }
}


bool readSensor(uint8_t index) {
  SensorSample& sample = sensors[index];
  if (!sample.ready || dmpPacketSize == 0U || !selectTCA(sample.channel)) {
    return false;
  }

  uint8_t interruptStatus = mpu.getIntStatus();
  uint16_t fifoCount = mpu.getFIFOCount();
  if ((interruptStatus & 0x10U) != 0U || fifoCount >= FIFO_OVERFLOW_BYTES) {
    mpu.resetFIFO();
    sample.valid = false;
    sample.fifoResets++;
    return false;
  }
  if (fifoCount < dmpPacketSize) {
    return false;
  }

  uint8_t packetCount = (uint8_t)(fifoCount / dmpPacketSize);
  if (packetCount > MAX_FIFO_PACKETS) {
    mpu.resetFIFO();
    sample.valid = false;
    sample.fifoResets++;
    return false;
  }
  while (packetCount > 0U) {
    mpu.getFIFOBytes(fifoBuffer, dmpPacketSize);
    packetCount--;
  }

  mpu.dmpGetQuaternion(&sample.quaternion, fifoBuffer);
  mpu.getMotion6(
    &sample.ax,
    &sample.ay,
    &sample.az,
    &sample.gx,
    &sample.gy,
    &sample.gz
  );
  sample.valid = true;
  return true;
}


void sendReply(const IPAddress& remoteIp, uint16_t remotePort, const char* text) {
  if (udpReady && remotePort != 0U) {
    udp.writeTo((const uint8_t*)text, strlen(text), remoteIp, remotePort);
  }
}


void sendStatus(const IPAddress& remoteIp, uint16_t remotePort) {
  char response[512];
  int written = snprintf(
    response,
    sizeof(response),
    "STATUS gravity_benchmark=1 sensors=%u rate_hz=%u streaming=%u packet_size=%u\n",
    sensorCount,
    STREAM_RATE_HZ,
    streaming ? 1U : 0U,
    dmpPacketSize
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
      "SENSOR id=%u ready=%u dmp_status=%u whoami=0x%02X fifo_resets=%u\n",
      sensors[index].id,
      sensors[index].ready ? 1U : 0U,
      sensors[index].dmpStatus,
      sensors[index].whoAmI,
      sensors[index].fifoResets
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
    uint8_t readyCount = 0U;
    for (uint8_t index = 0U; index < sensorCount; index++) {
      if (sensors[index].ready) {
        readyCount++;
      }
    }
    char response[96];
    snprintf(
      response,
      sizeof(response),
      "ACK HELLO gravity_benchmark=1 sensors=%u ready=%u rate_hz=%u\n",
      sensorCount,
      readyCount,
      STREAM_RATE_HZ
    );
    sendReply(command.remoteIp, command.remotePort, response);
  } else if (strcasecmp(command.text, "START") == 0) {
    resetAllFifos();
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
  Serial.print(F("NET:UDP port="));
  Serial.println(UDP_PORT);
  return true;
}


void sendFrame() {
  if (!udpReady || !clientRegistered || clientPort == 0U) {
    return;
  }

  uint8_t validCount = 0U;
  for (uint8_t index = 0U; index < sensorCount; index++) {
    if (sensors[index].ready && sensors[index].valid) {
      validCount++;
    }
  }

  char datagram[DATAGRAM_BUFFER_SIZE];
  int written = snprintf(
    datagram,
    sizeof(datagram),
    "GRAVITY_FRAME %lu %lu %u\n",
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
    if (!sample.ready || !sample.valid) {
      continue;
    }
    size_t remaining = sizeof(datagram) - used;
    written = snprintf(
      datagram + used,
      remaining,
      "S %u %d %d %d %d %d %d %.7f %.7f %.7f %.7f\n",
      sample.id,
      sample.ax,
      sample.ay,
      sample.az,
      sample.gx,
      sample.gy,
      sample.gz,
      (double)sample.quaternion.w,
      (double)sample.quaternion.x,
      (double)sample.quaternion.y,
      (double)sample.quaternion.z
    );
    if (written < 0 || (size_t)written >= remaining) {
      Serial.println(F("ERR:DATAGRAM_TOO_LARGE"));
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
  for (uint8_t index = 0U; index < sensorCount; index++) {
    readSensor(index);
  }
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
  Serial.println(F("RDY:send HELLO then START"));
}


void loop() {
  processPendingCommand();
  serviceFrame();
}
