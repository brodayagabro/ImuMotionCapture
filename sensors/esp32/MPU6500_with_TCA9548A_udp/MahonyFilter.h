#pragma once


struct OrientationQuaternion {
  float w;
  float x;
  float y;
  float z;
};


class MahonyFilter {
 public:
  MahonyFilter(
    float proportionalGain = 2.0f,
    float integralGain = 0.05f,
    float accelerationToleranceG = 0.25f
  );

  void reset();
  bool resetFromAcceleration(float axG, float ayG, float azG);
  bool update(
    float gxRadPerSecond,
    float gyRadPerSecond,
    float gzRadPerSecond,
    float axG,
    float ayG,
    float azG,
    float deltaTimeSeconds
  );

  OrientationQuaternion quaternion() const;
  void gravity(float* x, float* y, float* z) const;

 private:
  bool normalizeQuaternion();
  static float clamp(float value, float minimum, float maximum);

  float proportionalGain_;
  float integralGain_;
  float accelerationToleranceG_;
  float integralX_;
  float integralY_;
  float integralZ_;
  OrientationQuaternion quaternion_;
};
