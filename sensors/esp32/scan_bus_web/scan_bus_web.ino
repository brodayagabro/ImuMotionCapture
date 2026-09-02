// I2C Scanner for TCA9548A + MPU6050 with Web Interface
// ESP32-C3
#include <Wire.h>
#include <WiFi.h>
#include <WebServer.h>

#define TCA_ADDR 0x70

// ===== Настройки Wi-Fi =====
// Вариант 1: ESP32-C3 как точка доступа (проще для отладки)
const char* ap_ssid = "ESP32C3_I2C";
const char* ap_pass = "12345678";

// Вариант 2: подключение к роутеру (раскомментируйте и заполните)
const char* sta_ssid = "NeuroMorphMIPT";
const char* sta_pass = "31870016";

WebServer server(80);
String lastScanResult = "";

// ===== Страница HTML =====
const char PAGE_HTML[] PROGMEM = R"rawliteral(
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>I2C Scanner - ESP32-C3</title>
  <style>
    body { font-family: monospace; background: #1e1e1e; color: #d4d4d4; padding: 20px; }
    h1 { color: #569cd6; }
    button {
      background: #0e639c; color: white; border: none;
      padding: 10px 20px; font-size: 16px; cursor: pointer;
      border-radius: 4px; margin: 10px 0;
    }
    button:hover { background: #1177bb; }
    button:disabled { background: #555; cursor: not-allowed; }
    #output {
      background: #000; color: #4ec9b0; padding: 15px;
      border-radius: 4px; white-space: pre-wrap;
      font-size: 14px; min-height: 200px;
      border: 1px solid #333;
    }
    .status { color: #888; font-size: 12px; }
  </style>
</head>
<body>
  <h1>🔌 I2C Diagnostic (TCA9548A)</h1>
  <button id="scanBtn" onclick="doScan()">Сканировать шину</button>
  <button onclick="autoScan()">Автообновление: <span id="autoState">OFF</span></button>
  <div class="status" id="status">Готов к сканированию</div>
  <div id="output">Нажмите "Сканировать шину" для запуска...</div>

  <script>
    let autoInterval = null;

    function doScan() {
      const btn = document.getElementById('scanBtn');
      const out = document.getElementById('output');
      const st  = document.getElementById('status');
      btn.disabled = true;
      st.textContent = 'Сканирование...';
      out.textContent = '';

      fetch('/scan')
        .then(r => r.text())
        .then(text => {
          out.textContent = text;
          st.textContent = 'Последнее обновление: ' + new Date().toLocaleTimeString();
          btn.disabled = false;
        })
        .catch(err => {
          out.textContent = 'Ошибка: ' + err;
          btn.disabled = false;
        });
    }

    function autoScan() {
      const el = document.getElementById('autoState');
      if (autoInterval) {
        clearInterval(autoInterval);
        autoInterval = null;
        el.textContent = 'OFF';
      } else {
        autoInterval = setInterval(doScan, 3000);
        el.textContent = 'ON (3с)';
        doScan();
      }
    }
  </script>
</body>
</html>
)rawliteral";

// ===== Сканер =====
String scanBusToString(const char* label) {
  String out = String(label) + " I2C devices found: ";
  int count = 0;

  for (byte addr = 0x01; addr < 0x7F; addr++) {
    Wire.beginTransmission(addr);
    if (Wire.endTransmission() == 0) {
      out += "0x";
      if (addr < 0x10) out += "0";
      out += String(addr, HEX);
      out += " ";
      count++;
    }
  }
  if (count == 0) out += "NONE";
  out += "\n";
  return out;
}

void selectTCA(uint8_t ch) {
  Wire.beginTransmission(TCA_ADDR);
  Wire.write(1 << ch);
  Wire.endTransmission();
  delayMicroseconds(20);
}

String performFullScan() {
  String result = "";
  result += "=== I2C DIAGNOSTIC ===\n";

  // Главная шина
  result += scanBusToString("MAIN BUS");

  // Проверка TCA9548A
  Wire.beginTransmission(TCA_ADDR);
  if (Wire.endTransmission() == 0) {
    result += "✓ TCA9548A found at 0x70\n";
  } else {
    result += "✗ TCA9548A NOT FOUND at 0x70\n";
    result += "  → Проверьте адрес: A0-A2 пины? Попробуйте 0x71, 0x72...\n";
    result += "=== END ===\n";
    return result;
  }

  // Каналы мультиплексора
  for (uint8_t ch = 0; ch < 8; ch++) {
    selectTCA(ch);
    String label = "  CH" + String(ch) + "     ";
    result += scanBusToString(label.c_str());
  }

  result += "=== END ===\n";
  return result;
}

// ===== Обработчики HTTP =====
void handleRoot() {
  server.send(200, "text/html", PAGE_HTML);
}

void handleScan() {
  String result = performFullScan();
  lastScanResult = result;
  server.send(200, "text/plain", result);
}

void handleNotFound() {
  server.send(404, "text/plain", "404");
}

// ===== SETUP =====
void setup() {
  Serial.begin(115200);
  delay(200);
  Serial.println("\n=== ESP32-C3 I2C Web Scanner ===");

  // I2C
  // ESP32-C3: по умолчанию SDA=8, SCL=9. Можно переопределить:
  // Wire.begin(8, 9);
  Wire.begin();
  Wire.setClock(100000);

  // Wi-Fi
  #if defined(sta_ssid)
    Serial.printf("Connecting to %s...\n", sta_ssid);
    WiFi.mode(WIFI_STA);
    WiFi.begin(sta_ssid, sta_pass);
    int tries = 0;
    while (WiFi.status() != WL_CONNECTED && tries < 30) {
      delay(500); Serial.print("."); tries++;
    }
    if (WiFi.status() == WL_CONNECTED) {
      Serial.println("\nConnected! IP: " + WiFi.localIP().toString());
    } else {
      Serial.println("\nFailed. Falling back to AP mode.");
      WiFi.mode(WIFI_AP);
      WiFi.softAP(ap_ssid, ap_pass);
      Serial.println("AP IP: " + WiFi.softAPIP().toString());
    }
  #else
    WiFi.mode(WIFI_AP);
    WiFi.softAP(ap_ssid, ap_pass);
    Serial.println("AP mode. SSID: " + String(ap_ssid));
    Serial.println("AP IP: " + WiFi.softAPIP().toString());
  #endif

  // HTTP
  server.on("/", handleRoot);
  server.on("/scan", handleScan);
  server.onNotFound(handleNotFound);
  server.begin();
  Serial.println("HTTP server started");
}

// ===== LOOP =====
void loop() {
  server.handleClient();
}