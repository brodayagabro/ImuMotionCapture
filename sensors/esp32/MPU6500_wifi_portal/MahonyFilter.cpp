#include "MahonyFilter.h"

#include <math.h>


static const float INTEGRAL_LIMIT_RAD_PER_SECOND = 0.5f;
static const float MIN_VECTOR_NORM = 1.0e-6f;


MahonyFilter::MahonyFilter(
  float proportionalGain,
  float integralGain,
  float accelerationToleranceG
) :
  proportionalGain_(proportionalGain),
  integralGain_(integralGain),
  accelerationToleranceG_(accelerationToleranceG) {
  reset();
}


void MahonyFilter::reset() {
  quaternion_.w = 1.0f;
  quaternion_.x = 0.0f;
  quaternion_.y = 0.0f;
  quaternion_.z = 0.0f;
  integralX_ = 0.0f;
  integralY_ = 0.0f;
  integralZ_ = 0.0f;
}


bool MahonyFilter::resetFromAcceleration(float axG, float ayG, float azG) {
  float norm = sqrtf(axG * axG + ayG * ayG + azG * azG);
  if (!isfinite(norm) || norm < MIN_VECTOR_NORM) {
    reset();
    return false;
  }

  axG /= norm;
  ayG /= norm;
  azG /= norm;
  float roll = atan2f(ayG, azG);
  float pitch = atan2f(-axG, sqrtf(ayG * ayG + azG * azG));
  float halfRoll = 0.5f * roll;
  float halfPitch = 0.5f * pitch;
  float cosineRoll = cosf(halfRoll);
  float sineRoll = sinf(halfRoll);
  float cosinePitch = cosf(halfPitch);
  float sinePitch = sinf(halfPitch);

  quaternion_.w = cosineRoll * cosinePitch;
  quaternion_.x = sineRoll * cosinePitch;
  quaternion_.y = cosineRoll * sinePitch;
  quaternion_.z = -sineRoll * sinePitch;
  integralX_ = 0.0f;
  integralY_ = 0.0f;
  integralZ_ = 0.0f;
  return normalizeQuaternion();
}


bool MahonyFilter::update(
  float gxRadPerSecond,
  float gyRadPerSecond,
  float gzRadPerSecond,
  float axG,
  float ayG,
  float azG,
  float deltaTimeSeconds
) {
  if (!isfinite(deltaTimeSeconds) ||
      deltaTimeSeconds <= 0.0f ||
      deltaTimeSeconds > 0.1f) {
    return false;
  }

  float accelNorm = sqrtf(axG * axG + ayG * ayG + azG * azG);
  if (isfinite(accelNorm) && accelNorm >= MIN_VECTOR_NORM) {
    float trust = 1.0f;
    if (accelerationToleranceG_ > 0.0f) {
      trust = 1.0f - fabsf(accelNorm - 1.0f) / accelerationToleranceG_;
      trust = clamp(trust, 0.0f, 1.0f);
    }

    if (trust > 0.0f) {
      axG /= accelNorm;
      ayG /= accelNorm;
      azG /= accelNorm;

      float halfVx = quaternion_.x * quaternion_.z -
                     quaternion_.w * quaternion_.y;
      float halfVy = quaternion_.w * quaternion_.x +
                     quaternion_.y * quaternion_.z;
      float halfVz = quaternion_.w * quaternion_.w - 0.5f +
                     quaternion_.z * quaternion_.z;
      float halfErrorX = (ayG * halfVz - azG * halfVy) * trust;
      float halfErrorY = (azG * halfVx - axG * halfVz) * trust;
      float halfErrorZ = (axG * halfVy - ayG * halfVx) * trust;

      if (integralGain_ > 0.0f) {
        integralX_ += 2.0f * integralGain_ * halfErrorX * deltaTimeSeconds;
        integralY_ += 2.0f * integralGain_ * halfErrorY * deltaTimeSeconds;
        integralZ_ += 2.0f * integralGain_ * halfErrorZ * deltaTimeSeconds;
        integralX_ = clamp(
          integralX_,
          -INTEGRAL_LIMIT_RAD_PER_SECOND,
          INTEGRAL_LIMIT_RAD_PER_SECOND
        );
        integralY_ = clamp(
          integralY_,
          -INTEGRAL_LIMIT_RAD_PER_SECOND,
          INTEGRAL_LIMIT_RAD_PER_SECOND
        );
        integralZ_ = clamp(
          integralZ_,
          -INTEGRAL_LIMIT_RAD_PER_SECOND,
          INTEGRAL_LIMIT_RAD_PER_SECOND
        );
      }

      gxRadPerSecond += integralX_ + 2.0f * proportionalGain_ * halfErrorX;
      gyRadPerSecond += integralY_ + 2.0f * proportionalGain_ * halfErrorY;
      gzRadPerSecond += integralZ_ + 2.0f * proportionalGain_ * halfErrorZ;
    }
  }

  float halfDeltaTime = 0.5f * deltaTimeSeconds;
  gxRadPerSecond *= halfDeltaTime;
  gyRadPerSecond *= halfDeltaTime;
  gzRadPerSecond *= halfDeltaTime;

  float oldW = quaternion_.w;
  float oldX = quaternion_.x;
  float oldY = quaternion_.y;
  quaternion_.w +=
    -oldX * gxRadPerSecond -
    oldY * gyRadPerSecond -
    quaternion_.z * gzRadPerSecond;
  quaternion_.x +=
    oldW * gxRadPerSecond +
    oldY * gzRadPerSecond -
    quaternion_.z * gyRadPerSecond;
  quaternion_.y +=
    oldW * gyRadPerSecond -
    oldX * gzRadPerSecond +
    quaternion_.z * gxRadPerSecond;
  quaternion_.z +=
    oldW * gzRadPerSecond +
    oldX * gyRadPerSecond -
    oldY * gxRadPerSecond;
  return normalizeQuaternion();
}


OrientationQuaternion MahonyFilter::quaternion() const {
  return quaternion_;
}


void MahonyFilter::gravity(float* x, float* y, float* z) const {
  if (x != nullptr) {
    *x = 2.0f * (
      quaternion_.x * quaternion_.z -
      quaternion_.w * quaternion_.y
    );
  }
  if (y != nullptr) {
    *y = 2.0f * (
      quaternion_.w * quaternion_.x +
      quaternion_.y * quaternion_.z
    );
  }
  if (z != nullptr) {
    *z =
      quaternion_.w * quaternion_.w -
      quaternion_.x * quaternion_.x -
      quaternion_.y * quaternion_.y +
      quaternion_.z * quaternion_.z;
  }
}


bool MahonyFilter::normalizeQuaternion() {
  float norm = sqrtf(
    quaternion_.w * quaternion_.w +
    quaternion_.x * quaternion_.x +
    quaternion_.y * quaternion_.y +
    quaternion_.z * quaternion_.z
  );
  if (!isfinite(norm) || norm < MIN_VECTOR_NORM) {
    reset();
    return false;
  }
  float reciprocal = 1.0f / norm;
  quaternion_.w *= reciprocal;
  quaternion_.x *= reciprocal;
  quaternion_.y *= reciprocal;
  quaternion_.z *= reciprocal;
  return true;
}


float MahonyFilter::clamp(float value, float minimum, float maximum) {
  if (value < minimum) {
    return minimum;
  }
  if (value > maximum) {
    return maximum;
  }
  return value;
}
