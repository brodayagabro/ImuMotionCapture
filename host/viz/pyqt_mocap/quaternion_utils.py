"""Quaternion convention: Hamilton, wxyz, active rotations of column vectors.

q_AB maps B coordinates into A. compose(q_AB, q_BC) = q_AC, right factor first.
relative(reference, target) = inverse(reference) * target (reference-local error).
world_delta(reference, target) = target * inverse(reference) (world-space motion).
SciPy is an independent optional diagnostic oracle, not a live-loop dependency.
"""
from __future__ import annotations
from typing import Iterable, Sequence
import math
import numpy as np
from numpy.typing import NDArray
Quaternion = NDArray[np.float64]
Vector = NDArray[np.float64]
Matrix3 = NDArray[np.float64]
IDENTITY_QUATERNION = np.array((1., 0., 0., 0.))


def from_xyzw(values):
    """The single external scalar-last -> internal scalar-first boundary."""
    values = np.asarray(values, dtype=float)
    if values.ndim == 0 or values.shape[-1] != 4:
        raise ValueError("quaternion array must end in four components")
    return values[..., [3, 0, 1, 2]]


def to_xyzw(values):
    values = np.asarray(values, dtype=float)
    if values.ndim == 0 or values.shape[-1] != 4:
        raise ValueError("quaternion array must end in four components")
    return values[..., [1, 2, 3, 0]]

def normalize_quaternion(values: Iterable[float]) -> Quaternion:
    """Return a finite unit quaternion in ``w, x, y, z`` order."""
    quaternion = np.asarray(tuple(values), dtype=float)
    if quaternion.shape != (4,):
        raise ValueError("quaternion must contain exactly four values")
    norm = float(np.linalg.norm(quaternion))
    if norm < 1.0e-9 or not math.isfinite(norm):
        raise ValueError("invalid quaternion norm")
    return quaternion / norm


def validate_input_quaternion(values: Iterable[float]) -> Quaternion:
    """Validate an MPU sample with the same norm bounds as the Blender driver."""
    quaternion = np.asarray(tuple(values), dtype=float)
    if quaternion.shape != (4,):
        raise ValueError("quaternion must contain exactly four values")
    norm_squared = float(np.dot(quaternion, quaternion))
    if not math.isfinite(norm_squared) or not 0.25 <= norm_squared <= 2.25:
        raise ValueError("invalid sensor quaternion norm")
    return quaternion / math.sqrt(norm_squared)


def quaternion_multiply(left: Quaternion, right: Quaternion) -> Quaternion:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return normalize_quaternion(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        )
    )


def quaternion_inverse(quaternion: Quaternion) -> Quaternion:
    normalized = normalize_quaternion(quaternion)
    return normalized * np.array((1.0, -1.0, -1.0, -1.0))


def quaternion_from_rotation_vector(rotation_vector: Iterable[float]) -> Quaternion:
    """Convert an axis-angle rotation vector in radians to a quaternion."""
    vector = np.asarray(tuple(rotation_vector), dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError("rotation vector must contain three finite values")
    angle = float(np.linalg.norm(vector))
    if angle < 1.0e-12:
        return IDENTITY_QUATERNION.copy()
    half_angle = angle * 0.5
    xyz = vector / angle * math.sin(half_angle)
    return normalize_quaternion((math.cos(half_angle), xyz[0], xyz[1], xyz[2]))


def quaternion_to_matrix(quaternion: Quaternion) -> Matrix3:
    w, x, y, z = normalize_quaternion(quaternion)
    return np.array(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=float,
    )


def matrix_to_quaternion(matrix: Matrix3) -> Quaternion:
    """Convert a proper 3x3 rotation matrix to a unit quaternion."""
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        values = (
            0.25 * scale,
            (matrix[2, 1] - matrix[1, 2]) / scale,
            (matrix[0, 2] - matrix[2, 0]) / scale,
            (matrix[1, 0] - matrix[0, 1]) / scale,
        )
    else:
        diagonal = np.diag(matrix)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = math.sqrt(max(0.0, 1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])) * 2.0
            values = (
                (matrix[2, 1] - matrix[1, 2]) / scale,
                0.25 * scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
            )
        elif index == 1:
            scale = math.sqrt(max(0.0, 1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])) * 2.0
            values = (
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                0.25 * scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
            )
        else:
            scale = math.sqrt(max(0.0, 1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])) * 2.0
            values = (
                (matrix[1, 0] - matrix[0, 1]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
                0.25 * scale,
            )
    return normalize_quaternion(values)


def quaternion_slerp(left: Quaternion, right: Quaternion, alpha: float) -> Quaternion:
    if alpha <= 0.0:
        return normalize_quaternion(left)
    if alpha >= 1.0:
        return normalize_quaternion(right)

    start = normalize_quaternion(left)
    target = normalize_quaternion(right)
    dot = float(np.dot(start, target))
    if dot < 0.0:
        target = -target
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        return normalize_quaternion(start + alpha * (target - start))
    angle = math.acos(dot)
    sine = math.sin(angle)
    return normalize_quaternion(
        math.sin((1.0 - alpha) * angle) / sine * start
        + math.sin(alpha * angle) / sine * target
    )


def quaternion_to_rotation_vector(quaternion: Sequence[float]) -> Vector:
    """Return the shortest axis-angle vector in radians."""
    value = normalize_quaternion(quaternion)
    if value[0] < 0.0:
        value = -value
    xyz_norm = float(np.linalg.norm(value[1:]))
    if xyz_norm < 1.0e-12:
        return np.zeros(3, dtype=float)
    angle = 2.0 * math.atan2(xyz_norm, min(1.0, max(-1.0, float(value[0]))))
    return value[1:] / xyz_norm * angle


normalize = normalize_quaternion
inverse = quaternion_inverse
compose = quaternion_multiply


def relative(q_reference, q_target):
    return compose(inverse(q_reference), q_target)


def world_delta(q_reference, q_target):
    return compose(q_target, inverse(q_reference))


def angular_distance(q1, q2):
    """Shortest SO(3) distance in radians; invariant to either quaternion sign."""
    delta = relative(q1, q2)
    return 2 * math.atan2(float(np.linalg.norm(delta[1:])), abs(float(delta[0])))


def rotate_vector(q, v):
    return quaternion_to_matrix(q) @ np.asarray(v, dtype=float)


def average_quaternions(values):
    """Sign-invariant chordal SO(3) mean (principal eigenvector / Markley)."""
    if len(values) == 0:
        raise ValueError("cannot average an empty quaternion sequence")
    quaternions = np.stack([normalize(q) for q in values])
    _, vectors = np.linalg.eigh(quaternions.T @ quaternions)
    result = normalize(vectors[:, -1])
    return -result if result @ quaternions[0] < 0 else result


def twist_angle(q, axis):
    """Signed twist in radians; None when the 180-degree swing is singular."""
    axis = np.array(axis, dtype=float, copy=True)
    length = float(np.linalg.norm(axis))
    if not math.isfinite(length) or length < 1e-9:
        raise ValueError("twist axis must be nonzero and finite")
    axis /= length
    q = normalize(q)
    projection = float(q[1:] @ axis)
    if math.hypot(float(q[0]), projection) < 1e-9:
        return None
    return math.remainder(2 * math.atan2(projection, float(q[0])), 2 * math.pi)
