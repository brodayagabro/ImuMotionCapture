// MPU6050 x4 + TCA9548A: Стабильная инициализация
// Каналы: 0, 1, 5, 6

#include "I2Cdev.h"
#include "MPU6050_6Axis_MotionApps20.h"
#include <Wire.h>

#define NUM_SENSORS 4
#define TCA_ADDR 0x70
#define MPU_ADDR 0x68

// Физические каналы TCA
const uint8_t sensorChannels[NUM_SENSORS] = {0, 1, 5, 6};

MPU6050 mpu(MPU_ADDR); // Явно указываем адрес

struct SensorConfig {
  uint8_t channel;
  bool dmpReady;
  uint16_t packetSize;
  uint8_t fifoBuffer[64];
  Quaternion q;
} sensors[NUM_SENSORS];

bool dataSendingEnabled = false;
#define CMD_BUFFER_SIZE 16
char cmdBuffer[CMD_BUFFER_SIZE];
uint8_t cmdIdx = 0;

// ================= УПРАВЛЕНИЕ TCA =================
void selectTCA(uint8_t ch) {
  Wire.beginTransmission(TCA_ADDR);
  Wire.write(1 << ch);
  Wire.endTransmission();
  delay(20); // Критично для стабилизации TCA
}

// Сырой I2C-пинг (работает точно так же, как сканер)
bool pingMPU(uint8_t addr) {
  Wire.beginTransmission(addr);
  return (Wire.endTransmission() == 0);
}

// ================= ОБРАБОТКА КОМАНД =================
void processCommands() {
  while (Serial.available() > 0 && cmdIdx < CMD_BUFFER_SIZE - 1) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      cmdBuffer[cmdIdx] = '\0';
      if (strcmp(cmdBuffer, "START") == 0) {
        dataSendingEnabled = true;
        Serial.println("ACK:START");
      } else if (strcmp(cmdBuffer, "STOP") == 0) {
        dataSendingEnabled = false;
        Serial.println("ACK:STOP");
      } else if (strcmp(cmdBuffer, "CALIB") == 0) {
        Serial.println("ACK:CALIB (Running ~15s)");
        for (uint8_t i = 0; i < NUM_SENSORS; i++) {
          if (sensors[i].dmpReady) {
            selectTCA(sensors[i].channel);
            mpu.CalibrateAccel(6);
            mpu.CalibrateGyro(6);
            // Запишите офсеты через mpu.PrintActiveOffsets() и вставьте в initSensorDMP
          }
        }
        Serial.println("CALIB DONE");
      } else {
        Serial.print("ERR:Unknown: "); Serial.println(cmdBuffer);
      }
      cmdIdx = 0;
      memset(cmdBuffer, 0, CMD_BUFFER_SIZE);
    } else {
      cmdBuffer[cmdIdx++] = c;
    }
  }
}

// ================= ИНИЦИАЛИЗАЦИЯ ДАТЧИКА =================
bool initSensorDMP(uint8_t sensorId, SensorConfig& s) {
  selectTCA(s.channel);
  
  // 1. Сырой пинг (обходит баги библиотеки)
  if (!pingMPU(MPU_ADDR)) {
    Serial.print("ERR:Sensor "); Serial.print(sensorId+1); 
    Serial.println(" not responding to I2C ping");
    return false;
  }
  Serial.print("Sensor "); Serial.print(sensorId+1); 
  Serial.println(" detected on I2C");
  // supply your own gyro offsets here, scaled for min sensitivity
  mpu.setXGyroOffset(220);
  mpu.setYGyroOffset(76);
  mpu.setZGyroOffset(-85);
  mpu.setZAccelOffset(1788); // 1688 factory default for my test chip

  // 2. Инициализация DMP
  Serial.print("Loading DMP for sensor "); Serial.println(sensorId+1);
  uint8_t devStatus = mpu.dmpInitialize();

  if (devStatus == 0) {
    mpu.setDMPEnabled(true);
    s.packetSize = mpu.dmpGetFIFOPacketSize();
    s.dmpReady = true;
    Serial.print("OK:Sensor "); Serial.println(sensorId+1);
    return true;
  } else {
    Serial.print("ERR:DMP init sensor "); Serial.print(sensorId+1);
    Serial.print(" code="); Serial.println(devStatus);
    return false;
  }
}

// ================= SETUP =================
void setup() {
  Serial.begin(115200);
  while (!Serial); delay(200);
  
  // Инициализируем I2C на 100 кГц (надёжнее для TCA при старте)
  Wire.begin();
  Wire.setClock(100000UL);
  
  Serial.println("=== MPU6050 x4 + TCA9548A ===");
  Serial.println("Channels: 0, 1, 5, 6 | Init Clock: 100kHz");
  
  for (uint8_t i = 0; i < NUM_SENSORS; i++) {
    sensors[i].channel = sensorChannels[i];
    sensors[i].dmpReady = false;
    initSensorDMP(i, sensors[i]);
    delay(100); // Пауза между датчиками
  }
  
  // Переключаемся на высокую скорость только после успешной инициализации
  Wire.setClock(400000UL);
  Serial.println("Switched to 400kHz I2C");
  Serial.println("=== Ready. Send START to begin ===");
}

// ================= MAIN LOOP =================
void loop() {
  processCommands();
  
  for (uint8_t i = 0; i < NUM_SENSORS; i++) {
    if (!sensors[i].dmpReady) continue;
    
    selectTCA(sensors[i].channel);
    
    uint16_t fifoCount = mpu.getFIFOCount();
    while (fifoCount >= sensors[i].packetSize) {
      if (mpu.dmpGetCurrentFIFOPacket(sensors[i].fifoBuffer)) {
        mpu.dmpGetQuaternion(&sensors[i].q, sensors[i].fifoBuffer);
        
        if (dataSendingEnabled) {
          Serial.print("quat ");
          Serial.print(i + 1); Serial.print(" ");
          Serial.print(sensors[i].q.w, 4); Serial.print(" ");
          Serial.print(sensors[i].q.x, 4); Serial.print(" ");
          Serial.print(sensors[i].q.y, 4); Serial.print(" ");
          Serial.println(sensors[i].q.z, 4);
        }
      } else {
        mpu.resetFIFO(); 
        break;
      }
      fifoCount -= sensors[i].packetSize;
    }
    
    // Сброс при переполнении
    if (mpu.getIntStatus() & 0x10) mpu.resetFIFO();
  }
  
  delay(50);
}