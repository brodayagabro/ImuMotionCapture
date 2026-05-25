// I2C Scanner for TCA9548A + MPU6050
#include <Wire.h>

#define TCA_ADDR 0x70  // Может быть 0x70-0x77, проверьте!

void scanBus(const char* label) {
  Serial.print(label);
  Serial.print(" I2C devices found: ");
  byte count = 0;
  
  for (byte addr = 0x01; addr < 0x7F; addr++) {
    Wire.beginTransmission(addr);
    if (Wire.endTransmission() == 0) {
      Serial.print("0x");
      if (addr < 0x10) Serial.print("0");
      Serial.print(addr, HEX);
      Serial.print(" ");
      count++;
    }
  }
  if (count == 0) Serial.print("NONE");
  Serial.println();
}

void selectTCA(uint8_t ch) {
  Wire.beginTransmission(TCA_ADDR);
  Wire.write(1 << ch);
  Wire.endTransmission();
  delayMicroseconds(20);  // Стабилизация
}

void setup() {
  Serial.begin(115200);
  while (!Serial);
  delay(200);
  
  Serial.println("\n=== I2C DIAGNOSTIC ===");
  
  Wire.begin();
  Wire.setClock(100000);  // Начинаем с 100 кГц для надёжности
  
  // 1. Сканируем главную шину (без мультиплексора)
  scanBus("MAIN BUS");
  
  // 2. Проверяем, отвечает ли сам TCA9548A
  Wire.beginTransmission(TCA_ADDR);
  if (Wire.endTransmission() == 0) {
    Serial.println("✓ TCA9548A found at 0x70");
  } else {
    Serial.println("✗ TCA9548A NOT FOUND at 0x70");
    Serial.println("  → Проверьте адрес: A0-A2 пины? Попробуйте 0x71, 0x72...");
    return;
  }
  
  // 3. Сканируем каждый канал мультиплексора
  for (uint8_t ch = 0; ch < 8; ch++) {
    selectTCA(ch);
    scanBus((String("  CH" + String(ch) + "     ")).c_str());
  }
  
  Serial.println("=== END ===\n");
}

void loop() {
  delay(5000);  // Повтор каждые 5 сек
  // Можно раскомментировать для непрерывного сканирования:
  // setup();
}