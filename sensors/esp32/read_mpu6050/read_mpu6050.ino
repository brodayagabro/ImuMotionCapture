// ==================== ПОДКЛЮЧЕНИЕ БИБЛИОТЕК ====================
#include <Wire.h>              // Библиотека для I2C коммуникации
#include <Adafruit_MPU6050.h>  // Библиотека для работы с датчиком MPU-6050
#include <Adafruit_Sensor.h>   // Общая библиотека для работы с датчиками Adafruit

// ==================== СОЗДАНИЕ ОБЪЕКТА ДАТЧИКА ====================
Adafruit_MPU6050 mpu;  // Создание объекта для работы с MPU-6050

// ==================== ОПТИМИЗИРОВАННЫЕ ПАРАМЕТРЫ ====================
#define UPDATE_RATE 500        // Желаемая частота обновления данных: 500 Гц
#define SAMPLE_INTERVAL 2000   // Интервал между измерениями в микросекундах: 2000 мкс = 500 Гц

// Параметры комплементарного фильтра:
float alpha = 0.98;            // Коэффициент фильтра: 98% гироскоп, 2% акселерометр
float dt = 1.0 / UPDATE_RATE;  // Фиксированный временной интервал для вычислений

// ==================== ПЕРЕМЕННЫЕ ДЛЯ ВЫСОКОСКОРОСТНОЙ РАБОТЫ ====================
unsigned long lastMicros = 0;     // Время последнего измерения в микросекундах

// ==================== ПЕРЕМЕННЫЕ ДЛЯ ФИЛЬТРАЦИИ ДАННЫХ ====================
float roll = 0, pitch = 0;        // Финальные углы после комплементарного фильтра
float gyroRoll = 0, gyroPitch = 0; // Углы от интегрирования гироскопа
float accelRoll = 0, accelPitch = 0; // Углы от акселерометра

// ==================== СТРУКТУРА ДЛЯ КАЛИБРОВОЧНЫХ ДАННЫХ ====================
struct CalibrationData {
  float gyroX, gyroY, gyroZ;    // Смещения нуля для гироскопа
  float accelX, accelY, accelZ; // Смещения для акселерометра
};
CalibrationData calData;
const int calibrationSamples = 1000; // Количество измерений для калибровки

// ==================== КОНФИГУРАЦИЯ ОРИЕНТАЦИИ ДАТЧИКА ====================
#define SENSOR_VERTICAL true  // true: датчик вертикально, false: горизонтально

// ==================== ФУНКЦИЯ НАСТРОЙКИ ====================
void setup() {
  // Инициализация последовательного порта для вывода данных
  Serial.begin(115200);
  while (!Serial) delay(10); // Ждём готовности порта (для некоторых плат)
  
  Serial.println(F("=== MPU6050 Fast Reader ==="));
  
  // Инициализация I2C шины
  Wire.begin();
  Wire.setClock(400000);  // Высокая скорость I2C: 400 кГц
  
  // Инициализация датчика MPU-6050
  if (!mpu.begin()) {
    Serial.println(F("ERROR: MPU6050 not found! Check wiring."));
    while (1); // Остановка при ошибке
  }
  Serial.println(F("MPU6050 initialized successfully"));
  
  // Настройки датчика для высокой скорости
  mpu.setAccelerometerRange(MPU6050_RANGE_4_G);    // ±4g
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);         // ±500°/с
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);      // Полоса 21 Гц
  
  // Калибровка датчиков
  fastCalibrateSensors();
  
  // Инициализация таймера
  lastMicros = micros();
  
  Serial.println(F("Calibration complete. Starting data output..."));
  Serial.println(F("Format: timestamp_ms, pitch_deg, roll_deg"));
  Serial.println(F("-------------------------------------------"));
}

// ==================== ОСНОВНОЙ ЦИКЛ ПРОГРАММЫ ====================
void loop() {
  unsigned long currentMicros = micros();
  
  // ВЫСОКОЧАСТОТНОЕ ОБНОВЛЕНИЕ ДАННЫХ (500 Гц)
  if (currentMicros - lastMicros >= SAMPLE_INTERVAL) {
    lastMicros = currentMicros;
    
    // Чтение данных с датчика
    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp);
    
    // Обработка и фильтрация
    processSensorData(a, g);
    
    // ВЫВОД ДАННЫХ В SERIAL (каждое измерение)
    // Формат: время_мс;pitch;roll
    Serial.print("t:");
    Serial.print(millis());
    Serial.print(" P:");
    Serial.print(pitch, 1);  // 1 знак после запятой
    Serial.print(" R:");
    Serial.println(roll, 1);
    
    // ОПЦИОНАЛЬНО: вывод реже, чтобы не перегружать порт
    // if (millis() - lastSerialOutput >= 50) { ... }
  }
}

// ==================== ОБРАБОТКА ДАННЫХ ДАТЧИКА ====================
void processSensorData(sensors_event_t &a, sensors_event_t &g) {
  // Применение калибровочных смещений
  float accX = a.acceleration.x - calData.accelX;
  float accY = a.acceleration.y - calData.accelY;
  float accZ = a.acceleration.z - calData.accelZ;
  
  float gyroX = g.gyro.x - calData.gyroX;
  float gyroY = g.gyro.y - calData.gyroY;
  float gyroZ = g.gyro.z - calData.gyroZ;
  
  // Вычисление углов от акселерометра
  if (SENSOR_VERTICAL) {
    float sqrtYZ = sqrt(accY * accY + accZ * accZ);
    float sqrtXZ = sqrt(accX * accX + accZ * accZ);
    accelRoll = atan2(-accX, sqrtYZ) * 57.2958f;
    accelPitch = atan2(accY, sqrtXZ) * 57.2958f;
  }
  
  // Интегрирование гироскопа
  gyroRoll += gyroY * dt * 57.2958f;
  gyroPitch += gyroX * dt * 57.2958f;
  
  // Комплементарный фильтр
  roll = alpha * (roll + gyroY * dt * 57.2958f) + (1 - alpha) * accelRoll;
  pitch = alpha * (pitch + gyroX * dt * 57.2958f) + (1 - alpha) * accelPitch;
  
  // Ограничение углов
  roll = constrainAngleFast(roll);
  pitch = constrainAngleFast(pitch);
}

// ==================== ОГРАНИЧЕНИЕ УГЛА ====================
float constrainAngleFast(float angle) {
  angle = fmod(angle + 180.0f, 360.0f);
  if (angle < 0) angle += 360.0f;
  return angle - 180.0f;
}

// ==================== КАЛИБРОВКА ДАТЧИКОВ ====================
void fastCalibrateSensors() {
  Serial.print(F("Calibrating..."));
  
  float sumAccX = 0, sumAccY = 0, sumAccZ = 0;
  float sumGyroX = 0, sumGyroY = 0, sumGyroZ = 0;
  
  for (int i = 0; i < calibrationSamples; i++) {
    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp);
    
    sumAccX += a.acceleration.x;
    sumAccY += a.acceleration.y;
    sumAccZ += a.acceleration.z;
    sumGyroX += g.gyro.x;
    sumGyroY += g.gyro.y;
    sumGyroZ += g.gyro.z;
    
    delayMicroseconds(500);
    
    // Индикация прогресса каждые 200 измерений
    if (i % 200 == 0 && i > 0) Serial.print(F("."));
  }
  
  // Вычисление средних значений
  calData.accelX = sumAccX / calibrationSamples;
  calData.accelY = sumAccY / calibrationSamples;
  calData.accelZ = sumAccZ / calibrationSamples;
  calData.gyroX = sumGyroX / calibrationSamples;
  calData.gyroY = sumGyroY / calibrationSamples;
  calData.gyroZ = sumGyroZ / calibrationSamples;
  
  // Коррекция для вертикального датчика
  if (SENSOR_VERTICAL) {
    if (fabs(calData.accelX) > fabs(calData.accelY)) {
      calData.accelX -= 9.81f * (calData.accelX > 0 ? 1.0f : -1.0f);
    } else {
      calData.accelY -= 9.81f * (calData.accelY > 0 ? 1.0f : -1.0f);
    }
  }
  
  Serial.println(F(" DONE"));
  Serial.print(F("Acc offsets: "));
  Serial.print(calData.accelX, 2); Serial.print(", ");
  Serial.print(calData.accelY, 2); Serial.print(", ");
  Serial.println(calData.accelZ, 2);
  Serial.print(F("Gyro offsets: "));
  Serial.print(calData.gyroX, 2); Serial.print(", ");
  Serial.print(calData.gyroY, 2); Serial.print(", ");
  Serial.println(calData.gyroZ, 2);
}