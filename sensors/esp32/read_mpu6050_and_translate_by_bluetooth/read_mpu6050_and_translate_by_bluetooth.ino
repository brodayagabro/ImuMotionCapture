// ==================== ПОДКЛЮЧЕНИЕ БИБЛИОТЕК ====================
#include <Wire.h>              // Библиотека для I2C коммуникации
#include <Adafruit_MPU6050.h>  // Библиотека для работы с датчиком MPU-6050
#include <Adafruit_Sensor.h>   // Общая библиотека для датчиков Adafruit

// === Bluetooth только для ESP32 ===
#ifdef ESP32
  #include "BluetoothSerial.h"
  BluetoothSerial ESP_BT;      // Объект Bluetooth (как в вашем примере)
  #define BT_ENABLED true
#else
  #define BT_ENABLED false
#endif

// ==================== СОЗДАНИЕ ОБЪЕКТА ДАТЧИКА ====================
Adafruit_MPU6050 mpu;

// ==================== ОПТИМИЗИРОВАННЫЕ ПАРАМЕТРЫ ====================
#define UPDATE_RATE 100              // Частота обновления: 500 Гц
#define SAMPLE_INTERVAL 2000         // 2000 мкс = 500 Гц
#define CALIBRATION_SAMPLES 1000     // Измерений для калибровки

// Параметры комплементарного фильтра
float alpha = 0.98f;                 // 98% гироскоп, 2% акселерометр
float dt = 1.0f / UPDATE_RATE;

// ==================== ТАЙМИНГИ ====================
unsigned long lastMicros = 0;        // Для 500 Гц опроса датчика
unsigned long lastSend = 0;          // Для ограничения частоты отправки
#define SEND_INTERVAL_MS 20          // Отправлять данные каждые 20 мс (50 Гц)

// ==================== ПЕРЕМЕННЫЕ ФИЛЬТРА ====================
float roll = 0, pitch = 0;
float gyroRoll = 0, gyroPitch = 0;
float accelRoll = 0, accelPitch = 0;

// ==================== КАЛИБРОВКА ====================
struct CalibrationData {
  float gyroX, gyroY, gyroZ;
  float accelX, accelY, accelZ;
};
CalibrationData calData;

// ==================== ОРИЕНТАЦИЯ ====================
#define SENSOR_VERTICAL true

// ==================== BLUETOOTH НАСТРОЙКИ ====================
#define BT_DEVICE_NAME "ESP32-IMU"   // Имя устройства в поиске

// ==================== ФУНКЦИЯ ОТПРАВКИ ДАННЫХ ====================
inline void sendData(float timestamp, float pitch, float roll, float temp) {
  char buffer[64];
  snprintf(buffer, sizeof(buffer), "t:%lu P:%.1f R:%.1f T:%.1f", 
           (unsigned long)timestamp, pitch, roll, temp);
  
  // Всегда отправляем в USB-Serial
  Serial.println(buffer);
  
  // Отправляем в Bluetooth, если он включён
  #if BT_ENABLED
    ESP_BT.println(buffer);
  #endif
}

// ==================== SETUP ====================
void setup() {
  // USB-Serial для отладки
  Serial.begin(115200);
  while (!Serial && millis() < 2000);
  
  Serial.println(F("=== MPU6050 Fast Reader ==="));
  
  // === ИНИЦИАЛИЗАЦИЯ BLUETOOTH (стиль как в примере) ===
  #if BT_ENABLED
    ESP_BT.begin(BT_DEVICE_NAME);    // Запуск Bluetooth с именем устройства
    Serial.println(F("✓ Bluetooth ready to pair"));
    Serial.print(F("📱 Device name: "));
    Serial.println(BT_DEVICE_NAME);
  #endif
  
  // I2C шина
  Wire.begin();
  Wire.setClock(400000);  // 400 кГц
  
  // Датчик MPU-6050
  if (!mpu.begin()) {
    Serial.println(F("❌ ERROR: MPU6050 not found! Check I2C wiring."));
    while (1) delay(100);
  }
  Serial.println(F("✓ MPU6050 initialized"));
  
  // Настройки датчика
  mpu.setAccelerometerRange(MPU6050_RANGE_4_G);    // ±4g
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);         // ±500°/с
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);      // 21 Гц
  
  // Калибровка
  fastCalibrateSensors();
  
  // Тайминги
  lastMicros = micros();
  lastSend = millis();
  
  Serial.println(F("📤 Streaming started"));
  Serial.println(F("Format: t:ms P:pitch_deg R:roll_deg T:temp_C"));
  Serial.println(F("-------------------------------------------"));
}

// ==================== LOOP ====================
void loop() {
  unsigned long currentMicros = micros();
  
  // === ЧТЕНИЕ ДАТЧИКА 500 Гц ===
  if (currentMicros - lastMicros >= SAMPLE_INTERVAL) {
    lastMicros = currentMicros;
    
    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp);
    processSensorData(a, g);
    
    // === ОТПРАВКА ДАННЫХ с ограниченной частотой ===
    unsigned long now = millis();
    if (now - lastSend >= SEND_INTERVAL_MS) {
      lastSend = now;
      sendData(now, pitch, roll, temp.temperature);
    }
  }
  
  // === ОБРАБОТКА КОМАНД ЧЕРЕЗ BLUETOOTH (стиль как в примере) ===
  #if BT_ENABLED
    if (ESP_BT.available()) {              // Проверяем, есть ли данные по Bluetooth
      int incoming = ESP_BT.read();        // Считываем байт
      
      // Обработка команд (как в вашем примере с LED)
      if (incoming == 'C' || incoming == 'c') {   // Команда перекалибровки
        Serial.println(F("🔄 Recalibrating..."));
        ESP_BT.println("Recalibrating...");       // Ответ клиенту
        fastCalibrateSensors();
        ESP_BT.println("Calibration done");
        Serial.println(F("✓ Done"));
      }
      // Можно добавить другие команды по аналогии:
      // if (incoming == '1') { ... }
      // if (incoming == '0') { ... }
    }
  #endif
  
  // === ДУБЛИРУЕМ ОБРАБОТКУ КОМАНД ЧЕРЕЗ USB (для отладки) ===
  if (Serial.available()) {
    int incoming = Serial.read();
    if (incoming == 'C' || incoming == 'c') {
      Serial.println(F("🔄 Recalibrating..."));
      fastCalibrateSensors();
      Serial.println(F("✓ Done"));
    }
  }
  
  // === НЕБОЛЬШАЯ ЗАДЕРЖКА ДЛЯ СТАБИЛЬНОСТИ (как в примере) ===
  delay(20);
}

// ==================== ОБРАБОТКА ДАННЫХ ====================
void processSensorData(sensors_event_t &a, sensors_event_t &g) {
  float accX = a.acceleration.x - calData.accelX;
  float accY = a.acceleration.y - calData.accelY;
  float accZ = a.acceleration.z - calData.accelZ;
  
  float gyroX = g.gyro.x - calData.gyroX;
  float gyroY = g.gyro.y - calData.gyroY;
  float gyroZ = g.gyro.z - calData.gyroZ;
  
  if (SENSOR_VERTICAL) {
    float sqrtYZ = sqrt(accY * accY + accZ * accZ);
    float sqrtXZ = sqrt(accX * accX + accZ * accZ);
    accelRoll  = atan2(-accX, sqrtYZ) * 57.2958f;
    accelPitch = atan2(accY, sqrtXZ) * 57.2958f;
  }
  
  gyroRoll  += gyroY * dt * 57.2958f;
  gyroPitch += gyroX * dt * 57.2958f;
  
  roll  = alpha * (roll  + gyroY * dt * 57.2958f) + (1 - alpha) * accelRoll;
  pitch = alpha * (pitch + gyroX * dt * 57.2958f) + (1 - alpha) * accelPitch;
  
  roll  = constrainAngleFast(roll);
  pitch = constrainAngleFast(pitch);
}

// ==================== ОГРАНИЧЕНИЕ УГЛА ====================
float constrainAngleFast(float angle) {
  angle = fmod(angle + 180.0f, 360.0f);
  if (angle < 0) angle += 360.0f;
  return angle - 180.0f;
}

// ==================== КАЛИБРОВКА ====================
void fastCalibrateSensors() {
  Serial.print(F("Calibrating..."));
  #if BT_ENABLED
    ESP_BT.print("Calibrating...");
  #endif
  
  float sumAcc[3] = {0}, sumGyro[3] = {0};
  
  for (int i = 0; i < CALIBRATION_SAMPLES; i++) {
    sensors_event_t a, g, t;
    mpu.getEvent(&a, &g, &t);
    
    sumAcc[0]  += a.acceleration.x;
    sumAcc[1]  += a.acceleration.y;
    sumAcc[2]  += a.acceleration.z;
    sumGyro[0] += g.gyro.x;
    sumGyro[1] += g.gyro.y;
    sumGyro[2] += g.gyro.z;
    
    delayMicroseconds(500);
    
    if (i % 200 == 0 && i > 0) {
      Serial.print(F("."));
      #if BT_ENABLED
        ESP_BT.print(".");
      #endif
    }
  }
  
  calData.accelX = sumAcc[0] / CALIBRATION_SAMPLES;
  calData.accelY = sumAcc[1] / CALIBRATION_SAMPLES;
  calData.accelZ = sumAcc[2] / CALIBRATION_SAMPLES;
  calData.gyroX  = sumGyro[0] / CALIBRATION_SAMPLES;
  calData.gyroY  = sumGyro[1] / CALIBRATION_SAMPLES;
  calData.gyroZ  = sumGyro[2] / CALIBRATION_SAMPLES;
  
  if (SENSOR_VERTICAL) {
    if (fabs(calData.accelX) > fabs(calData.accelY)) {
      calData.accelX -= 9.81f * (calData.accelX > 0 ? 1.0f : -1.0f);
    } else {
      calData.accelY -= 9.81f * (calData.accelY > 0 ? 1.0f : -1.0f);
    }
  }
  
  Serial.println(F(" DONE"));
  Serial.printf("Acc offsets: %.2f, %.2f, %.2f\n", 
                calData.accelX, calData.accelY, calData.accelZ);
  Serial.printf("Gyro offsets: %.2f, %.2f, %.2f\n", 
                calData.gyroX, calData.gyroY, calData.gyroZ);
  
  #if BT_ENABLED
    ESP_BT.println(" DONE");
  #endif
}