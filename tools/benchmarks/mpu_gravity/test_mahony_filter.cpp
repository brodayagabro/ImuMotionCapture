#include "../../../sensors/esp32/MPU6500_with_TCA9548A_udp/MahonyFilter.h"

#include <cmath>
#include <cstdlib>
#include <iostream>


static void requireNear(float actual, float expected, float tolerance, const char* name) {
  if (std::fabs(actual - expected) > tolerance) {
    std::cerr << name << ": expected " << expected << ", got " << actual << std::endl;
    std::exit(1);
  }
}


static void requireGravity(
  MahonyFilter& filter,
  float expectedX,
  float expectedY,
  float expectedZ,
  float tolerance
) {
  float x;
  float y;
  float z;
  filter.gravity(&x, &y, &z);
  requireNear(x, expectedX, tolerance, "gravity.x");
  requireNear(y, expectedY, tolerance, "gravity.y");
  requireNear(z, expectedZ, tolerance, "gravity.z");
}


int main() {
  MahonyFilter filter;

  filter.resetFromAcceleration(0.0f, 0.0f, 1.0f);
  requireGravity(filter, 0.0f, 0.0f, 1.0f, 1.0e-5f);

  filter.resetFromAcceleration(0.0f, 0.0f, -1.0f);
  requireGravity(filter, 0.0f, 0.0f, -1.0f, 1.0e-5f);

  filter.resetFromAcceleration(1.0f, 0.0f, 0.0f);
  requireGravity(filter, 1.0f, 0.0f, 0.0f, 1.0e-5f);

  filter.resetFromAcceleration(-1.0f, 0.0f, 0.0f);
  requireGravity(filter, -1.0f, 0.0f, 0.0f, 1.0e-5f);

  filter.resetFromAcceleration(0.0f, 1.0f, 0.0f);
  requireGravity(filter, 0.0f, 1.0f, 0.0f, 1.0e-5f);

  filter.resetFromAcceleration(0.0f, -1.0f, 0.0f);
  requireGravity(filter, 0.0f, -1.0f, 0.0f, 1.0e-5f);

  // With no gyroscope motion the correction must converge to the measured
  // gravity direction instead of slowly returning toward the identity pose.
  filter.reset();
  for (int index = 0; index < 2000; index++) {
    filter.update(0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.005f);
  }
  requireGravity(filter, 1.0f, 0.0f, 0.0f, 0.02f);

  filter.reset();
  const float halfPi = 1.57079632679f;
  for (int index = 0; index < 1000; index++) {
    filter.update(halfPi, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.001f);
  }
  requireGravity(filter, 0.0f, 1.0f, 0.0f, 0.002f);

  filter.reset();
  for (int index = 0; index < 1000; index++) {
    filter.update(0.0f, 0.0f, halfPi, 0.0f, 0.0f, 1.0f, 0.001f);
  }
  requireGravity(filter, 0.0f, 0.0f, 1.0f, 0.002f);

  std::cout << "MahonyFilter tests passed" << std::endl;
  return 0;
}
