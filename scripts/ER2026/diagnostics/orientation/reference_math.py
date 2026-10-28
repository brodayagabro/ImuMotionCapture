"""Optional SciPy oracle/optimization. Never imported by the live viewer."""
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from pyqt_mocap.quaternion_utils import from_xyzw, to_xyzw


def rotation(q):
    return Rotation.from_quat(to_xyzw(q))


def quaternion(r):
    return from_xyzw(r.as_quat())


def fit_mounting(sensor_quaternions, target_quaternions, world_alignment, initial_mounting):
    """One constant right-side mounting, with world alignment fixed explicitly."""
    sensors = rotation(sensor_quaternions)
    targets = rotation(target_quaternions)
    world = rotation(world_alignment)

    def residual(vector):
        return (targets.inv() * world * sensors * Rotation.from_rotvec(vector)).as_rotvec().ravel()

    fit = least_squares(residual, rotation(initial_mounting).as_rotvec(),
                        ftol=1e-12, xtol=1e-12, gtol=1e-12)
    mounting = Rotation.from_rotvec(fit.x)
    return {
        "q_sensor_segment": quaternion(mounting).tolist(),
        "success": bool(fit.success),
        "pose_orientation_error_deg": np.degrees(
            (targets.inv() * world * sensors * mounting).magnitude()).tolist(),
        "observability": "fixed world alignment and fully specified target frames required",
    }
