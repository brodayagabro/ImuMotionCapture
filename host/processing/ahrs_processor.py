import numpy as np
import imufusion
from host.shared.config import SAMPLE_RATE

class AHRSProcessor:
    def __init__(self):
        self.ahrs = imufusion.Ahrs()
        self.ahrs.settings = imufusion.AhrsSettings(
            gain=0.5,
            magnetic_rejection=10,
            recovery_trigger_period=SAMPLE_RATE * 20
        )

    def update(self, gyro: np.ndarray, accel: np.ndarray, mag: np.ndarray) -> np.ndarray:
        self.ahrs.update(gyro, accel, mag, 1.0 / SAMPLE_RATE)
        return np.array(imufusion.quaternion_to_euler(self.ahrs.quaternion))