#pragma once
#include <stdint.h>

// Platform-independent retry budget, including millis() wraparound.
class WiFiRetry {
 public:
  enum Result { Waiting, Retry, Exhausted };
  static constexpr uint32_t TIMEOUT_MS = 20000U;
  void begin(uint32_t now) { attempt_ = 1; started_ = now; active_ = true; }
  void stop() { active_ = false; }
  bool active() const { return active_; }
  unsigned attempt() const { return attempt_; }
  Result poll(uint32_t now) {
    if (!active_ || uint32_t(now - started_) < TIMEOUT_MS) return Waiting;
    if (attempt_ == 3) { active_ = false; return Exhausted; }
    ++attempt_;
    started_ = now;
    return Retry;
  }
 private:
  uint32_t started_ = 0;
  unsigned attempt_ = 0;
  bool active_ = false;
};
