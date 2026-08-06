/*
  Multiple MPU6050-compatible DMP sensors through TCA9548A.
  ESP32-C3 Wi-Fi/UDP server with Serial fallback.

  Serial stream protocol:
    quat <sensor_id> <w> <x> <y> <z>

  UDP stream protocol (one datagram per frame):
    FRAME <sequence> <millis> <quaternion_count>
    Q <sensor_id> <w> <x> <y> <z>

  Commands:
    START             - reset FIFOs and start streaming
    STOP              - stop streaming
    CALIB             - recalibrate all sensors while fully still
    STATUS            - return sensors and timing counters
    HELLO             - register a UDP client without starting
    PING              - check UDP connectivity
    SET_RATE <1..100> - set acquisition/stream rate in Hz

  A UDP command registers its sender as the only active stream client. The
  callback only queues commands; sensor and configuration work stays in loop().
*/

#include "I2Cdev.h"
#include "MPU6050_6Axis_MotionApps20.h"
#include "Wire.h"
#include <string.h>
#include <strings.h>
#include <stdlib.h>
#include <Arduino.h>
#include "WiFi.h"
#include "AsyncUDP.h"


static const uint32_t BAUD_RATE = 115200UL;
static const uint16_t SERIAL_WAIT_MS = 2000U;
static const uint32_t I2C_INIT_CLOCK_HZ = 100000UL;
static const uint32_t I2C_RUN_CLOCK_HZ = 400000UL;
static const uint32_t I2C_TIMEOUT_US = 3000UL;

static const uint16_t DEFAULT_STREAM_RATE_HZ = 10U;
static const uint16_t MIN_STREAM_RATE_HZ = 1U;
static const uint16_t MAX_STREAM_RATE_HZ = 100U;
static const uint16_t QUAT_PRINT_SCALE = 10000U; // 4 decimal places.

static const uint8_t LED_PIN = 13U;
static const uint8_t CMD_BUFFER_SIZE = 64U;
static const uint16_t UDP_PACKET_BUFFER_SIZE = 512U;
static const uint8_t CALIBRATION_LOOPS = 6U;
static const uint8_t MAX_SENSORS = 5U;
static const uint8_t MAX_FIFO_PACKETS_PER_FRAME = 16U;
static const uint8_t NO_TCA_CHANNEL = 0xFFU;

static const uint16_t FIFO_OVERFLOW_BYTES = 1024U;

static const uint8_t TCA_ADDR_FIRST = 0x70U;
static const uint8_t TCA_ADDR_LAST = 0x77U;
static const uint8_t MPU_ADDR = 0x68U;
static const uint8_t MPU_ADDR_ALT = 0x69U;
static const uint8_t MPU_WHO_AM_I_REG = 0x75U;

static const uint16_t UDP_LISTEN_PORT = 4210U;
static const uint32_t WIFI_CONNECT_TIMEOUT_MS = 20000UL;
static const uint32_t WIFI_RECONNECT_INTERVAL_MS = 5000UL;

// Set to false to use SENSOR_CHANNELS instead of scanning channels 0..7.
static const bool AUTO_DETECT_CHANNELS = true;
static const bool VERBOSE_I2C_SCAN = false;
static const uint8_t SENSOR_CHANNELS[MAX_SENSORS] = {0U, 1U, 2U, 5U, 6U};

// Change these two values before flashing if another WLAN is used.
static const char *WIFI_SSID = "NeuroMorphMIPT";
static const char *WIFI_PASSWORD = "31870016";

MPU6050 mpu(MPU_ADDR);
AsyncUDP udp;

struct SensorState {
  uint8_t id;
  uint8_t channel;
  bool dmpReady;
  bool hasQuat;
  Quaternion q;
  uint16_t fifoResetCount;
};

SensorState sensors[MAX_SENSORS];
uint8_t sensorCount = 0U;
uint8_t tcaAddress = 0U;
uint8_t activeTcaChannel = NO_TCA_CHANNEL;
uint8_t dmpPacketSize = 0U;
uint8_t fifoBuffer[64];

bool streamEnabled = false;
bool blinkState = false;

uint16_t streamRateHz = DEFAULT_STREAM_RATE_HZ;
uint32_t frameIntervalMs = 1000UL / DEFAULT_STREAM_RATE_HZ;

bool udpServerReady = false;
bool udpClientRegistered = false;
IPAddress udpClientIp;
uint16_t udpClientPort = 0U;
uint32_t lastWiFiReconnectMs = 0UL;

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

uint32_t previousFrameMs = 0UL;
uint32_t frameCounter = 0UL;
uint16_t lateFrameCount = 0U;
uint16_t lastFrameDurationUs = 0U;
uint16_t maxFrameDurationUs = 0U;

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
    char c = text[length - 1U];
    if (c != ' ' && c != '\t' && c != '\r' && c != '\n') {
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
  Serial.println(WIFI_SSID);
  Serial.print(F("NET:IP "));
  Serial.println(WiFi.localIP());
  Serial.print(F("NET:UDP_PORT "));
  Serial.println(UDP_LISTEN_PORT);
}

void connectWiFi() {
  Serial.print(F("NET:connecting to "));
  Serial.println(WIFI_SSID);

  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  uint32_t startedAt = millis();
  while (WiFi.status() != WL_CONNECTED &&
         (uint32_t)(millis() - startedAt) < WIFI_CONNECT_TIMEOUT_MS) {
    delay(250);
    Serial.write('.');
  }
  Serial.println();

  if (WiFi.status() == WL_CONNECTED) {
    printNetworkAddress();
    startUdpServer();
  } else {
    Serial.println(F("ERR:WIFI_CONNECT (will retry in loop)"));
  }
  lastWiFiReconnectMs = millis();
}

void serviceWiFi() {
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
    Serial.println(F("NET:UDP stopped; WiFi disconnected"));
  }

  uint32_t now = millis();
  if ((uint32_t)(now - lastWiFiReconnectMs) >= WIFI_RECONNECT_INTERVAL_MS) {
    lastWiFiReconnectMs = now;
    Serial.println(F("NET:reconnecting WiFi"));
    WiFi.reconnect();
  }
}

bool pingI2C(uint8_t address) {
  Wire.beginTransmission(address);
  return Wire.endTransmission() == 0;
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
  if (Wire.endTransmission() != 0) {
    activeTcaChannel = NO_TCA_CHANNEL;
    return false;
  }

  // The TCA9548A channel is valid when the I2C transaction is complete.
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

bool readRegister(uint8_t address, uint8_t reg, uint8_t* value) {
  Wire.beginTransmission(address);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) {
    return false;
  }

  if (Wire.requestFrom(address, (uint8_t)1U) != 1U) {
    return false;
  }

  *value = Wire.read();
  return true;
}

void printHexByte(uint8_t value) {
  Serial.print(F("0x"));
  if (value < 16U) {
    Serial.print(F("0"));
  }
  Serial.print(value, HEX);
}

uint8_t readWhoAmI() {
  uint8_t raw = 0U;
  if (readRegister(MPU_ADDR, MPU_WHO_AM_I_REG, &raw)) {
    return raw;
  }
  return 0U;
}

uint8_t printWhoAmI(uint8_t sensorIndex) {
  SensorState& s = sensors[sensorIndex];
  selectTCA(s.channel);
  uint8_t whoAmI = readWhoAmI();

  Serial.print(F("WHOAMI S"));
  Serial.print(s.id);
  Serial.print(F(" CH"));
  Serial.print(s.channel);
  Serial.print(F(" raw "));
  printHexByte(whoAmI);
  Serial.print(F(" library_id "));
  printHexByte(mpu.getDeviceID());
  Serial.println();

  return whoAmI;
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

void scanTCAChannels() {
  Serial.println(F("SCAN:TCA channels"));

  for (uint8_t channel = 0U; channel < 8U; channel++) {
    Serial.print(F("SCAN:CH"));
    Serial.print(channel);
    Serial.write(' ');

    if (!selectTCA(channel)) {
      Serial.println(F("select_failed"));
      continue;
    }

    uint8_t count = 0U;
    for (uint8_t address = 1U; address < 127U; address++) {
      if (address == tcaAddress) {
        continue;
      }
      if (pingI2C(address)) {
        printHexByte(address);
        Serial.write(' ');
        count++;
      }
    }

    if (count == 0U) {
      Serial.print(F("none"));
    }
    Serial.println();
  }
}

bool isSupportedWhoAmI(uint8_t whoAmI) {
  return whoAmI == 0x68U || whoAmI == 0x70U || whoAmI == 0x71U || whoAmI == 0x73U;
}

void addSensor(uint8_t channel) {
  if (sensorCount >= MAX_SENSORS) {
    return;
  }

  SensorState& s = sensors[sensorCount];
  s.id = sensorCount + 1U;
  s.channel = channel;
  s.dmpReady = false;
  s.hasQuat = false;
  s.q.w = 1.0f;
  s.q.x = 0.0f;
  s.q.y = 0.0f;
  s.q.z = 0.0f;
  s.fifoResetCount = 0U;
  sensorCount++;
}

bool discoverSensors() {
  sensorCount = 0U;

  if (VERBOSE_I2C_SCAN) {
    scanTCAChannels();
  }

  if (AUTO_DETECT_CHANNELS) {
    for (uint8_t channel = 0U; channel < 8U && sensorCount < MAX_SENSORS; channel++) {
      if (!selectTCA(channel)) {
        continue;
      }

      if (pingI2C(MPU_ADDR)) {
        addSensor(channel);
      } else if (pingI2C(MPU_ADDR_ALT)) {
        Serial.print(F("WARN:MPU_AT_0x69_ON_CH"));
        Serial.println(channel);
      }
    }
  } else {
    for (uint8_t i = 0U; i < MAX_SENSORS; i++) {
      uint8_t channel = SENSOR_CHANNELS[i];
      if (!selectTCA(channel)) {
        continue;
      }

      if (pingI2C(MPU_ADDR)) {
        addSensor(channel);
      } else {
        Serial.print(F("WARN:NO_MPU_ON_CONFIG_CH"));
        Serial.println(channel);
      }
    }
  }

  Serial.print(F("INIT:sensors detected "));
  Serial.println(sensorCount);

  for (uint8_t i = 0U; i < sensorCount; i++) {
    Serial.print(F("INIT:S"));
    Serial.print(sensors[i].id);
    Serial.print(F(" channel "));
    Serial.println(sensors[i].channel);
  }

  if (sensorCount == 0U) {
    Serial.println(F("ERR:NO_SENSORS"));
    scanTCAChannels();
    return false;
  }
  return true;
}

void printActiveOffsets(uint8_t sensorIndex) {
  SensorState& s = sensors[sensorIndex];
  selectTCA(s.channel);

  Serial.print(F("OFFSETS S"));
  Serial.print(s.id);
  Serial.print(F(" accel "));
  Serial.print(mpu.getXAccelOffset());
  Serial.write(' ');
  Serial.print(mpu.getYAccelOffset());
  Serial.write(' ');
  Serial.print(mpu.getZAccelOffset());
  Serial.print(F(" gyro "));
  Serial.print(mpu.getXGyroOffset());
  Serial.write(' ');
  Serial.print(mpu.getYGyroOffset());
  Serial.write(' ');
  Serial.println(mpu.getZGyroOffset());
}

bool initSensorDMP(uint8_t sensorIndex) {
  SensorState& s = sensors[sensorIndex];
  selectTCA(s.channel);

  Serial.print(F("INIT:S"));
  Serial.print(s.id);
  Serial.print(F(" DMP on CH"));
  Serial.println(s.channel);

  uint8_t whoAmI = printWhoAmI(sensorIndex);

  mpu.initialize();
  selectTCA(s.channel);
  whoAmI = printWhoAmI(sensorIndex);

  if (!mpu.testConnection()) {
    if (isSupportedWhoAmI(whoAmI)) {
      Serial.print(F("WARN:S"));
      Serial.print(s.id);
      Serial.println(F(" WHOAMI_NOT_MPU6050 trying DMP anyway"));
    } else {
      Serial.print(F("ERR:S"));
      Serial.print(s.id);
      Serial.println(F(" MPU_CONNECTION"));
      return false;
    }
  }

  selectTCA(s.channel);
  uint8_t devStatus = mpu.dmpInitialize();

  mpu.setXGyroOffset(0);
  mpu.setYGyroOffset(0);
  mpu.setZGyroOffset(0);
  mpu.setXAccelOffset(0);
  mpu.setYAccelOffset(0);
  mpu.setZAccelOffset(0);

  if (devStatus != 0U) {
    Serial.print(F("ERR:S"));
    Serial.print(s.id);
    Serial.print(F(" DMP_INIT "));
    Serial.println(devStatus);
    return false;
  }

  Serial.print(F("CAL:S"));
  Serial.print(s.id);
  Serial.println(F(" BEGIN keep all sensors still"));

  selectTCA(s.channel);
  mpu.setDMPEnabled(false);
  mpu.resetFIFO();
  mpu.CalibrateAccel(CALIBRATION_LOOPS);
  mpu.CalibrateGyro(CALIBRATION_LOOPS);
  printActiveOffsets(sensorIndex);
  mpu.setDMPEnabled(true);
  mpu.resetFIFO();

  uint16_t packetSize = mpu.dmpGetFIFOPacketSize();
  if (packetSize == 0U || packetSize > sizeof(fifoBuffer)) {
    Serial.print(F("ERR:S"));
    Serial.print(s.id);
    Serial.print(F(" BAD_PACKET_SIZE "));
    Serial.println(packetSize);
    return false;
  }

  if (dmpPacketSize == 0U) {
    dmpPacketSize = (uint8_t)packetSize;
  } else if (packetSize != dmpPacketSize) {
    Serial.print(F("WARN:S"));
    Serial.print(s.id);
    Serial.print(F(" packet_size_mismatch "));
    Serial.println(packetSize);
  }

  s.dmpReady = true;
  s.hasQuat = false;

  Serial.print(F("OK:S"));
  Serial.print(s.id);
  Serial.print(F(" packet_size "));
  Serial.println(dmpPacketSize);
  return true;
}

void resetAllFifos() {
  for (uint8_t i = 0U; i < sensorCount; i++) {
    if (!sensors[i].dmpReady) {
      continue;
    }

    selectTCA(sensors[i].channel);
    mpu.resetFIFO();
    mpu.getIntStatus();
    sensors[i].hasQuat = false;
  }
}

void resetTimingStats() {
  frameCounter = 0UL;
  lateFrameCount = 0U;
  lastFrameDurationUs = 0U;
  maxFrameDurationUs = 0U;
}

void calibrateAllSensors() {
  bool wasStreaming = streamEnabled;
  streamEnabled = false;

  Serial.println(F("CAL:ALL BEGIN keep all sensors still"));
  for (uint8_t i = 0U; i < sensorCount; i++) {
    if (!sensors[i].dmpReady) {
      continue;
    }

    selectTCA(sensors[i].channel);
    mpu.setDMPEnabled(false);
    mpu.resetFIFO();
    mpu.CalibrateAccel(CALIBRATION_LOOPS);
    mpu.CalibrateGyro(CALIBRATION_LOOPS);
    printActiveOffsets(i);
    mpu.setDMPEnabled(true);
    mpu.resetFIFO();
    sensors[i].hasQuat = false;
  }
  Serial.println(F("CAL:ALL DONE"));

  if (wasStreaming) {
    resetAllFifos();
    previousFrameMs = millis();
  }
  streamEnabled = wasStreaming;
}

void printStatus() {
  Serial.print(F("STAT:sensors "));
  Serial.print(sensorCount);
  Serial.print(F(" streaming "));
  Serial.println(streamEnabled ? F("yes") : F("no"));

  Serial.print(F("STAT:rate_hz "));
  Serial.print(streamRateHz);
  Serial.print(F(" frame_ms "));
  Serial.print(frameIntervalMs);
  Serial.print(F(" frames "));
  Serial.print(frameCounter);
  Serial.print(F(" late "));
  Serial.print(lateFrameCount);
  Serial.print(F(" last_us "));
  Serial.print(lastFrameDurationUs);
  Serial.print(F(" max_us "));
  Serial.print(maxFrameDurationUs);
  Serial.print(F(" packet_size "));
  Serial.println(dmpPacketSize);

  Serial.print(F("STAT:wifi "));
  Serial.print(WiFi.status() == WL_CONNECTED ? F("connected ip ") : F("disconnected ip "));
  Serial.print(WiFi.localIP());
  Serial.print(F(" udp_port "));
  Serial.println(UDP_LISTEN_PORT);

  for (uint8_t i = 0U; i < sensorCount; i++) {
    Serial.print(F("STAT:S"));
    Serial.print(sensors[i].id);
    Serial.print(F(" ch "));
    Serial.print(sensors[i].channel);
    Serial.print(F(" dmp "));
    Serial.print(sensors[i].dmpReady ? F("ready") : F("not_ready"));
    Serial.print(F(" fifo_resets "));
    Serial.println(sensors[i].fifoResetCount);
  }
}

void sendStatusUdp(const IPAddress& remoteIp, uint16_t remotePort) {
  char response[UDP_PACKET_BUFFER_SIZE];
  int written = snprintf(
    response,
    sizeof(response),
    "STATUS sensors=%u streaming=%u rate_hz=%u frame_ms=%lu frames=%lu "
    "late=%u last_us=%u max_us=%u packet_size=%u wifi=%u udp_port=%u\n",
    sensorCount,
    streamEnabled ? 1U : 0U,
    streamRateHz,
    (unsigned long)frameIntervalMs,
    (unsigned long)frameCounter,
    lateFrameCount,
    lastFrameDurationUs,
    maxFrameDurationUs,
    dmpPacketSize,
    WiFi.status() == WL_CONNECTED ? 1U : 0U,
    UDP_LISTEN_PORT
  );

  if (written < 0) {
    return;
  }
  size_t used = (size_t)written;
  if (used >= sizeof(response)) {
    used = sizeof(response) - 1U;
  }

  for (uint8_t i = 0U; i < sensorCount && used < sizeof(response) - 1U; i++) {
    size_t remaining = sizeof(response) - used;
    written = snprintf(
      response + used,
      remaining,
      "SENSOR id=%u channel=%u dmp=%u fifo_resets=%u\n",
      sensors[i].id,
      sensors[i].channel,
      sensors[i].dmpReady ? 1U : 0U,
      sensors[i].fifoResetCount
    );
    if (written < 0) {
      break;
    }
    if ((size_t)written >= remaining) {
      used = sizeof(response) - 1U;
      break;
    }
    used += (size_t)written;
  }

  response[used] = '\0';
  sendUdpReply(remoteIp, remotePort, response);
}

void handleCommand(char* cmd, const IPAddress* remoteIp, uint16_t remotePort) {
  trimCommand(cmd);
  if (cmd[0] == '\0') {
    return;
  }

  bool fromUdp = remoteIp != nullptr && remotePort != 0U;
  if (fromUdp) {
    udpClientIp = *remoteIp;
    udpClientPort = remotePort;
    udpClientRegistered = true;
  }

  if (strcasecmp(cmd, "HELLO") == 0) {
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
  } else if (strcasecmp(cmd, "PING") == 0) {
    Serial.println(F("ACK:PING"));
    if (fromUdp) {
      sendUdpReply(*remoteIp, remotePort, "PONG\n");
    }
  } else if (strcasecmp(cmd, "START") == 0) {
    resetAllFifos();
    resetTimingStats();
    previousFrameMs = millis();
    streamEnabled = true;
    Serial.println(F("ACK:START"));
    if (fromUdp) {
      sendUdpReply(*remoteIp, remotePort, "ACK START\n");
    }
  } else if (strcasecmp(cmd, "STOP") == 0) {
    streamEnabled = false;
    digitalWrite(LED_PIN, LOW);
    Serial.println(F("ACK:STOP"));
    if (fromUdp) {
      sendUdpReply(*remoteIp, remotePort, "ACK STOP\n");
    }
  } else if (strcasecmp(cmd, "CALIB") == 0) {
    Serial.println(F("ACK:CALIB BEGIN"));
    if (fromUdp) {
      sendUdpReply(*remoteIp, remotePort, "ACK CALIB BEGIN keep_sensors_still=1\n");
    }
    calibrateAllSensors();
    Serial.println(F("ACK:CALIB DONE"));
    if (fromUdp) {
      sendUdpReply(*remoteIp, remotePort, "ACK CALIB DONE\n");
    }
  } else if (strcasecmp(cmd, "STATUS") == 0 ||
             strcasecmp(cmd, "GET_CONFIG") == 0) {
    printStatus();
    if (fromUdp) {
      sendStatusUdp(*remoteIp, remotePort);
    }
  } else if (strncasecmp(cmd, "SET_RATE", 8U) == 0) {
    const char* valueText = cmd + 8U;
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
      Serial.print(F("ERR:RATE expected "));
      Serial.print(MIN_STREAM_RATE_HZ);
      Serial.write('-');
      Serial.println(MAX_STREAM_RATE_HZ);
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
    Serial.print(streamRateHz);
    Serial.print(F(" Hz frame_ms "));
    Serial.println(frameIntervalMs);
    if (fromUdp) {
      sendUdpReply(*remoteIp, remotePort, response);
    }
  } else {
    char response[CMD_BUFFER_SIZE + 24U];
    snprintf(response, sizeof(response), "ERR UNKNOWN %s\n", cmd);
    Serial.print(F("ERR:UNKNOWN "));
    Serial.println(cmd);
    if (fromUdp) {
      sendUdpReply(*remoteIp, remotePort, response);
    }
  }
}

void processCommands() {
  while (Serial.available() > 0) {
    char c = Serial.read();

    if (c == '\n' || c == '\r') {
      cmdBuffer[cmdIdx] = '\0';
      handleCommand(cmdBuffer, nullptr, 0U);
      cmdIdx = 0U;
      cmdBuffer[0] = '\0';
    } else if (cmdIdx < CMD_BUFFER_SIZE - 1U) {
      cmdBuffer[cmdIdx++] = c;
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
    memcpy(command.text, pendingUdpCommand.text, sizeof(command.text));
    command.remoteIp = pendingUdpCommand.remoteIp;
    command.remotePort = pendingUdpCommand.remotePort;
    udpCommandPending = false;
    hasCommand = true;
  }
  portEXIT_CRITICAL(&udpCommandMux);

  if (hasCommand) {
    handleCommand(command.text, &command.remoteIp, command.remotePort);
  }
}

bool readLatestSensorPacket(uint8_t sensorIndex) {
  SensorState& s = sensors[sensorIndex];
  if (!s.dmpReady || dmpPacketSize == 0U) {
    return false;
  }

  if (!selectTCA(s.channel)) {
    return false;
  }

  uint8_t intStatus = mpu.getIntStatus();
  uint16_t fifoCount = mpu.getFIFOCount();

  if ((intStatus & 0x10U) != 0U || fifoCount >= FIFO_OVERFLOW_BYTES) {
    mpu.resetFIFO();
    mpu.getIntStatus();
    s.hasQuat = false;
    s.fifoResetCount++;
    return false;
  }

  if (fifoCount < dmpPacketSize) {
    return false;
  }

  uint8_t packetCount = (uint8_t)(fifoCount / dmpPacketSize);
  if (packetCount > MAX_FIFO_PACKETS_PER_FRAME) {
    // Too much backlog means old orientation and a long blocking drain.
    // Reset to recover low latency and preserve the 50 ms frame budget.
    mpu.resetFIFO();
    mpu.getIntStatus();
    s.hasQuat = false;
    s.fifoResetCount++;
    return false;
  }

  while (packetCount > 0U) {
    mpu.getFIFOBytes(fifoBuffer, dmpPacketSize);
    packetCount--;
  }

  mpu.dmpGetQuaternion(&s.q, fifoBuffer);
  s.hasQuat = true;
  return true;
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

void sendSensorQuaternion(const SensorState& s) {
  Serial.print(F("quat "));
  Serial.print(s.id);
  Serial.write(' ');
  printQuatComponent(s.q.w);
  Serial.write(' ');
  printQuatComponent(s.q.x);
  Serial.write(' ');
  printQuatComponent(s.q.y);
  Serial.write(' ');
  printQuatComponent(s.q.z);
  Serial.write('\n');
}

void sendAllQuaternions() {
  uint8_t quaternionCount = 0U;
  for (uint8_t i = 0U; i < sensorCount; i++) {
    if (sensors[i].dmpReady && sensors[i].hasQuat) {
      sendSensorQuaternion(sensors[i]);
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
  if (written < 0) {
    return;
  }

  size_t used = (size_t)written;
  if (used >= sizeof(datagram)) {
    return;
  }

  for (uint8_t i = 0U; i < sensorCount; i++) {
    const SensorState& s = sensors[i];
    if (!s.dmpReady || !s.hasQuat) {
      continue;
    }

    size_t remaining = sizeof(datagram) - used;
    written = snprintf(
      datagram + used,
      remaining,
      "Q %u %.6f %.6f %.6f %.6f\n",
      s.id,
      (double)s.q.w,
      (double)s.q.x,
      (double)s.q.y,
      (double)s.q.z
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

  for (uint8_t i = 0U; i < sensorCount; i++) {
    readLatestSensorPacket(i);
  }

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
    // We missed at least one full frame. Do not emit burst frames; resync.
    previousFrameMs = currentMillis;
  }

  runStreamingFrame();
}

void setupDMP() {
  Serial.println(F("INIT:MPU xN via TCA"));
  Serial.println(F("INIT:checking TCA9548A at 0x70..0x77"));

  if (!detectTCAAddress()) {
    return;
  }

  disableTCAChannels();

  if (!discoverSensors()) {
    return;
  }

  uint8_t readyCount = 0U;
  for (uint8_t i = 0U; i < sensorCount; i++) {
    if (initSensorDMP(i)) {
      readyCount++;
    }
  }

  resetAllFifos();

  Wire.setClock(I2C_RUN_CLOCK_HZ);
  Serial.print(F("INIT:I2C run clock "));
  Serial.println(I2C_RUN_CLOCK_HZ);

  Serial.print(F("RDY:sensors_ready "));
  Serial.println(readyCount);
  Serial.println(F("RDY:send START to stream sensor quaternions"));
}

void setup() {
  pinMode(LED_PIN, OUTPUT);
  Serial.begin(BAUD_RATE);

  uint32_t startedAt = millis();
  while (!Serial && (uint32_t)(millis() - startedAt) < SERIAL_WAIT_MS) {
  }

  #if defined(SDA) && defined(SCL)
    pinMode(SDA, INPUT_PULLUP);
    pinMode(SCL, INPUT_PULLUP);
  #endif

  Wire.begin();
  Wire.setClock(I2C_INIT_CLOCK_HZ);

  #if defined(WIRE_HAS_TIMEOUT)
    Wire.setWireTimeout(I2C_TIMEOUT_US, true);
    Serial.println(F("INIT:I2C timeout enabled"));
  #endif

  setupDMP();
  connectWiFi();
}

void loop() {
  processCommands();
  processUdpCommands();
  serviceWiFi();
  serviceFrameScheduler();
}
