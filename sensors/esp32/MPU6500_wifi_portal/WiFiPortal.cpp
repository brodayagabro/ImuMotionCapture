#include "WiFiPortal.h"
#include <WiFi.h>
#include <esp_system.h>
#include <string.h>

namespace {
constexpr uint32_t MAGIC = 0x57494631U;
constexpr uint32_t PORTAL_GRACE_MS = 15000U;
const char* AP_SSID = "MOCAP_MIPT";
const char* AP_PASSWORD = "12345678";

bool valid(const String& ssid, const String& password) {
  if (ssid.length() == 0 || ssid.length() > 32) return false;
  for (size_t i = 0; i < ssid.length(); ++i) if (!ssid[i]) return false;
  for (size_t i = 0; i < password.length(); ++i) if (!password[i]) return false;
  if (password.length() == 64) {
    for (size_t i = 0; i < 64; ++i) {
      char c = password[i];
      if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F'))) return false;
    }
    return true;
  }
  return password.length() == 0 || (password.length() >= 8 && password.length() <= 63);
}
}

void WiFiPortal::begin() {
  // Credentials are persisted only in our NVS blob, never by WiFi.begin().
  WiFi.persistent(false);
  WiFi.mode(WIFI_STA);
  WiFi.setAutoReconnect(false);
  WiFi.setMinSecurity(WIFI_AUTH_OPEN);  // An empty form password permits open SSIDs.
  WiFi.setSleep(false);
  storageReady_ = storage_.begin("mocap_wifi", false);
  bool loaded = storageReady_ && storage_.getBytesLength("credentials") == sizeof(credentials_)
    && storage_.getBytes("credentials", &credentials_, sizeof(credentials_)) == sizeof(credentials_)
    && credentials_.magic == MAGIC
    && memchr(credentials_.ssid, 0, sizeof(credentials_.ssid))
    && memchr(credentials_.password, 0, sizeof(credentials_.password))
    && valid(String(credentials_.ssid), String(credentials_.password));
  server_.on("/", HTTP_GET, [this]() { page(); });
  server_.on("/save", HTTP_POST, [this]() { save(); });
  server_.onNotFound([this]() {
    if (allowRequest()) server_.send(404, "text/plain", "Not found. Open /");
  });
  if (loaded) {
    message_ = "Подключение к сохранённой сети.";
    retry_.begin(millis());
    startAttempt();
  } else {
    memset(&credentials_, 0, sizeof(credentials_));
    message_ = storageReady_ ? "Укажите сеть Wi-Fi 2,4 ГГц." : "Ошибка NVS: сохранение недоступно. Перезагрузите устройство.";
    openPortal();
  }
}

void WiFiPortal::startAttempt() {
  WiFi.disconnect(false, false);
  WiFi.begin(credentials_.ssid, credentials_.password);
  Serial.printf("NET:WiFi attempt %u/3\n", retry_.attempt());
}

void WiFiPortal::openPortal() {
  if (portalOpen_) return;
  lastPortalTry_ = millis();
  WiFi.mode(WIFI_AP_STA);
  IPAddress ip(192, 168, 4, 1);
  if (!WiFi.softAPConfig(ip, ip, IPAddress(255, 255, 255, 0)) ||
      !WiFi.softAP(AP_SSID, AP_PASSWORD)) {
    Serial.println("ERR:WIFI_AP_START");
    return;
  }
  char token[33];
  snprintf(token, sizeof(token), "%08lx%08lx%08lx%08lx",
           (unsigned long)esp_random(), (unsigned long)esp_random(),
           (unsigned long)esp_random(), (unsigned long)esp_random());
  token_ = token;
  portalOpen_ = true;
  server_.begin();
  Serial.println("NET:AP MOCAP_MIPT http://192.168.4.1");
}

void WiFiPortal::closePortal() {
  server_.stop();
  WiFi.softAPdisconnect(true);
  WiFi.mode(WIFI_STA);
  portalOpen_ = false;
  closeScheduled_ = false;
  Serial.println("NET:configuration AP stopped");
}

bool WiFiPortal::allowRequest() {
  // Even while AP+STA coexist, never expose configuration on the router LAN.
  if (!portalOpen_ || server_.client().localIP() != WiFi.softAPIP()) {
    server_.send(403, "text/plain", "Connect to MOCAP_MIPT to configure Wi-Fi");
    return false;
  }
  server_.sendHeader("Cache-Control", "no-store");
  server_.sendHeader("X-Frame-Options", "DENY");
  server_.sendHeader("X-Content-Type-Options", "nosniff");
  return true;
}

void WiFiPortal::page() {
  if (!allowRequest()) return;
  String html = F("<!doctype html><html lang='ru'><meta charset='utf-8'>"
    "<meta name='viewport' content='width=device-width,initial-scale=1'>"
    "<title>MOCAP — Wi-Fi</title><style>body{font:18px sans-serif;max-width:32em;"
    "margin:3em auto;padding:1em}input,button{font:inherit;padding:.5em;box-sizing:border-box;"
    "width:100%;margin:.5em 0}button{cursor:pointer}</style><h1>MOCAP_MIPT</h1><p>");
  html += message_;  // Only fixed messages/IP, never submitted credentials.
  html += F("</p>");
  if (connected_) {
    html += F("<p>IP костюма в вашей сети: <strong>");
    html += WiFi.localIP().toString();
    html += F("</strong>, UDP: 4210. Переключитесь обратно в вашу Wi-Fi сеть. "
              "В приложении укажите этот IP и нажмите HELLO/START. "
              "Точка настройки отключится через 15 секунд после подключения.</p>");
  } else if (retry_.active() || connectQueued_) {
    html += F("<meta http-equiv='refresh' content='3'><p>Проверка сети, попытка ");
    html += String(retry_.active() ? retry_.attempt() : 1);
    html += F("/3. До 20 секунд на попытку. Страница обновится автоматически.</p>");
  } else {
    html += F("<form action='/save' method='post'><input type='hidden' name='token' value='");
    html += token_;
    html += F("'><label>Имя сети (SSID)<input name='ssid' maxlength='32' required "
      "autocomplete='off'></label><label>Пароль<input name='password' type='password' "
      "maxlength='64' autocomplete='new-password'></label><p>Для открытой сети оставьте "
      "пароль пустым. Пароль защищённой сети: 8–63 байта или 64 hex-символа.</p>"
      "<button>Сохранить и подключиться</button></form>");
  }
  html += F("</html>");
  server_.send(200, "text/html; charset=utf-8", html);
}

void WiFiPortal::save() {
  if (!allowRequest()) return;
  if (!server_.hasArg("token") || server_.arg("token") != token_) {
    server_.send(403, "text/plain; charset=utf-8", "Обновите страницу настройки.");
    return;
  }
  if (retry_.active() || connectQueued_ || connected_) {
    server_.send(409, "text/plain; charset=utf-8", "Подключение уже выполняется или завершено.");
    return;
  }
  if (!storageReady_) {
    server_.send(503, "text/plain; charset=utf-8", "NVS недоступно. Перезагрузите устройство.");
    return;
  }
  String ssid = server_.arg("ssid");
  String password = server_.arg("password");
  if (!valid(ssid, password)) {
    server_.send(400, "text/plain; charset=utf-8", "Неверная длина SSID или пароля. Вернитесь к форме.");
    return;
  }
  memset(&credentials_, 0, sizeof(credentials_));
  credentials_.magic = MAGIC;
  ssid.toCharArray(credentials_.ssid, sizeof(credentials_.ssid));
  password.toCharArray(credentials_.password, sizeof(credentials_.password));
  pendingSave_ = true;
  connectQueued_ = true;
  queuedAt_ = millis();
  message_ = "Проверяем подключение. Настройки сохранятся только при успехе.";
  server_.sendHeader("Location", "/");
  server_.send(303, "text/plain", "Connecting");
}

void WiFiPortal::service() {
  if (portalOpen_) server_.handleClient();
  uint32_t now = millis();
  if (connectQueued_ && uint32_t(now - queuedAt_) >= 250U) {
    connectQueued_ = false;
    retry_.begin(now);
    startAttempt();
  }
  if (WiFi.status() == WL_CONNECTED) {
    if (!connected_) {
      connected_ = true;
      retry_.stop();
      bool saved = !pendingSave_ || (storageReady_ &&
        storage_.putBytes("credentials", &credentials_, sizeof(credentials_)) == sizeof(credentials_));
      pendingSave_ = false;
      message_ = saved ? "Подключено. Сеть сохранена." : "Подключено, но ошибка записи NVS. После перезапуска сеть может быть потеряна.";
      Serial.println(saved ? "NET:WiFi connected" : "ERR:WIFI_NVS_SAVE");
      Serial.print("NET:IP ");
      Serial.println(WiFi.localIP());
      if (portalOpen_ && saved) { closeScheduled_ = true; closeAt_ = now; }
    }
    if (closeScheduled_ && uint32_t(now - closeAt_) >= PORTAL_GRACE_MS) closePortal();
    return;
  }
  if (connected_) {
    connected_ = false;
    closeScheduled_ = false;
    message_ = "Связь потеряна. Повторное подключение.";
    retry_.begin(now);
    startAttempt();
  }
  WiFiRetry::Result result = retry_.poll(now);
  if (result == WiFiRetry::Retry) startAttempt();
  if (result == WiFiRetry::Exhausted) {
    WiFi.disconnect(false, false);
    pendingSave_ = false;
    message_ = "Три попытки не удались. Проверьте SSID и пароль. Прежняя сохранённая сеть не изменена.";
    Serial.println("NET:WiFi failed after 3 attempts");
    openPortal();
  }
  if (!retry_.active() && !connectQueued_ && !portalOpen_ &&
      uint32_t(now - lastPortalTry_) >= 5000U) openPortal();
}
