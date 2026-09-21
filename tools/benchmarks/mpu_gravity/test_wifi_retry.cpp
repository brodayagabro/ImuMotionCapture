#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include "../../../sensors/esp32/MPU6500_wifi_portal/WiFiRetry.h"

int main() {
  WiFiRetry retry;
  assert(!retry.active());
  assert(retry.poll(100000) == WiFiRetry::Waiting);
  retry.begin(100);
  assert(retry.attempt() == 1 && retry.active());
  assert(retry.poll(20099) == WiFiRetry::Waiting);
  assert(retry.poll(20100) == WiFiRetry::Retry && retry.attempt() == 2);
  assert(retry.poll(40100) == WiFiRetry::Retry && retry.attempt() == 3);
  assert(retry.poll(60100) == WiFiRetry::Exhausted && !retry.active());
  assert(retry.poll(80100) == WiFiRetry::Waiting && retry.attempt() == 3);
  // A form submission or disconnect starts a fresh budget.
  retry.begin(100000);
  assert(retry.attempt() == 1);
  retry.stop();
  assert(retry.poll(200000) == WiFiRetry::Waiting);
  // Device uptime can overflow millis() after approximately 49 days.
  uint32_t start = UINT32_MAX - 10000U;
  retry.begin(start);
  assert(retry.poll(start + 19999U) == WiFiRetry::Waiting);
  assert(retry.poll(start + 20000U) == WiFiRetry::Retry);
  assert(retry.poll(start + 40000U) == WiFiRetry::Retry);
  assert(retry.poll(start + 60000U) == WiFiRetry::Exhausted);
  puts("WiFi retry tests passed");
}
