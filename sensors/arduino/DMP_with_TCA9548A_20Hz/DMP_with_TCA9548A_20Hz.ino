/*
  Multiple MPU6050-compatible DMP sensors through TCA9548A.
  Optimized for Arduino Uno / ATmega328P 20 Hz streaming.

  Runtime protocol:
    quat <sensor_id> <w> <x> <y> <z>

  Commands:
    START  - reset FIFOs and start 20 Hz streaming
    STOP   - stop streaming
    CALIB  - recalibrate all detected sensors while fully still
    STATUS - print detected sensors and timing counters

  The streaming loop is frame-based: one acquisition/output frame every
  50 ms. The scheduler advances by adding FRAME_INTERVAL_MS to the previous
  frame timestamp, so ordinary jitter does not accumulate as long-term drift.
*/

#include "I2Cdev.h"
#include "MPU6050_6Axis_MotionApps20.h"
#include "Wire.h"
#include <string.h>

static const uint32_t BAUD_RATE = 115200UL;
static const uint16_t SERIAL_WAIT_MS = 2000U;
static const uint32_t I2C_INIT_CLOCK_HZ = 100000UL;
static const uint32_t I2C_RUN_CLOCK_HZ = 400000UL;
static const uint32_t I2C_TIMEOUT_US = 3000UL;

static const uint32_t FRAME_INTERVAL_MS = 100UL;  // 20 Hz exactly.
static const uint16_t QUAT_PRINT_SCALE = 10000U; // 4 decimal places.

static const uint8_t LED_PIN = 13U;
static const uint8_t CMD_BUFFER_SIZE = 16U;
static const uint8_t CALIBRATION_LOOPS = 6U;
static const uint8_t MAX_SENSORS = 4U;
static const uint8_t MAX_FIFO_PACKETS_PER_FRAME = 16U;
static const uint8_t NO_TCA_CHANNEL = 0xFFU;

static const uint16_t FIFO_OVERFLOW_BYTES = 1024U;

static const uint8_t TCA_ADDR_FIRST = 0x70U;
static const uint8_t TCA_ADDR_LAST = 0x77U;
static const uint8_t MPU_ADDR = 0x68U;
static const uint8_t MPU_ADDR_ALT = 0x69U;
static const uint8_t MPU_WHO_AM_I_REG = 0x75U;

// Set to false to use SENSOR_CHANNELS instead of scanning channels 0..7.
static const bool AUTO_DETECT_CHANNELS = true;
static const bool VERBOSE_I2C_SCAN = false;
static const uint8_t SENSOR_CHANNELS[MAX_SENSORS] = {0U, 1U, 5U, 6U};

MPU6050 mpu(MPU_ADDR);

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

char cmdBuffer[CMD_BUFFER_SIZE];
uint8_t cmdIdx = 0U;

uint32_t previousFrameMs = 0UL;
uint32_t frameCounter = 0UL;
uint16_t lateFrameCount = 0U;
uint16_t lastFrameDurationUs = 0U;
uint16_t maxFrameDurationUs = 0U;

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

  Serial.print(F("STAT:frame_ms "));
  Serial.print(FRAME_INTERVAL_MS);
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

void handleCommand(const char* cmd) {
  if (strcmp(cmd, "START") == 0) {
    resetAllFifos();
    resetTimingStats();
    previousFrameMs = millis();
    streamEnabled = true;
    Serial.println(F("ACK:START"));
  } else if (strcmp(cmd, "STOP") == 0) {
    streamEnabled = false;
    Serial.println(F("ACK:STOP"));
  } else if (strcmp(cmd, "CALIB") == 0) {
    calibrateAllSensors();
    Serial.println(F("ACK:CALIB"));
  } else if (strcmp(cmd, "STATUS") == 0) {
    printStatus();
  } else if (cmd[0] != '\0') {
    Serial.print(F("ERR:UNKNOWN "));
    Serial.println(cmd);
  }
}

void processCommands() {
  while (Serial.available() > 0) {
    char c = Serial.read();

    if (c == '\n' || c == '\r') {
      cmdBuffer[cmdIdx] = '\0';
      handleCommand(cmdBuffer);
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
  for (uint8_t i = 0U; i < sensorCount; i++) {
    if (sensors[i].dmpReady && sensors[i].hasQuat) {
      sendSensorQuaternion(sensors[i]);
    }
  }
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
  if (elapsedMs < FRAME_INTERVAL_MS) {
    return;
  }

  if (elapsedMs > FRAME_INTERVAL_MS && lateFrameCount < 65535U) {
    lateFrameCount++;
  }

  previousFrameMs += FRAME_INTERVAL_MS;

  if ((uint32_t)(currentMillis - previousFrameMs) >= FRAME_INTERVAL_MS) {
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
}

void loop() {
  processCommands();
  serviceFrameScheduler();
}
