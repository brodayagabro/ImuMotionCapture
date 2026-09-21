#pragma once
#include <Arduino.h>
#include <Preferences.h>
#include <WebServer.h>
#include "WiFiRetry.h"

class WiFiPortal {
 public:
  void begin();
  void service();
 private:
  struct Credentials {
    uint32_t magic;
    char ssid[33];
    char password[65];
  } credentials_{};
  Preferences storage_;
  WebServer server_{80};
  WiFiRetry retry_;
  bool storageReady_ = false;
  bool portalOpen_ = false;
  bool connected_ = false;
  bool pendingSave_ = false;
  bool connectQueued_ = false;
  bool closeScheduled_ = false;
  uint32_t queuedAt_ = 0;
  uint32_t closeAt_ = 0;
  uint32_t lastPortalTry_ = 0;
  String token_;
  String message_;
  void startAttempt();
  void openPortal();
  void closePortal();
  void page();
  void save();
  bool allowRequest();
};
