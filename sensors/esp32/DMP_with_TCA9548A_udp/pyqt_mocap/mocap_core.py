"""Core UDP, quaternion, calibration, and mannequin math for the PyQt viewer.

This module deliberately has no Qt dependency.  The GUI can therefore keep all
socket activity outside the main thread, while the protocol and kinematics are
easy to exercise in headless tests.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Final, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray


Vector = NDArray[np.float64]
Quaternion = NDArray[np.float64]
Matrix3 = NDArray[np.float64]

IDENTITY_QUATERNION: Final[Quaternion] = np.array((1.0, 0.0, 0.0, 0.0))
SEGMENT_NAMES: Final = (
    "spine",
    "shoulder.L",
    "forearm.L",
    "shoulder.R",
    "forearm.R",
)
DEFAULT_ENABLED_SEGMENTS: Final = tuple(
    name for name in SEGMENT_NAMES if name != "spine"
)

SENSOR_MAPPING_REVISION: Final = 2
LEGACY_SENSOR_MAPPING: Final = {
    "shoulder.L": 0,
    "forearm.L": 1,
    "spine": 2,
    "forearm.R": 6,
    "shoulder.R": 7,
}
DEFAULT_SENSOR_MAPPING: Final = {
    "shoulder.L": 7,
    "forearm.L": 6,
    "spine": 2,
    "forearm.R": 1,
    "shoulder.R": 0,
}
PARENT_SEGMENT: Final = {
    "spine": None,
    "shoulder.L": "spine",
    "forearm.L": "shoulder.L",
    "shoulder.R": "spine",
    "forearm.R": "shoulder.R",
}
RAW_SENSOR_ID_PROFILES: Final = {
    "tca_channel": {0: 0, 1: 1, 2: 2, 6: 6, 7: 7},
    "sequential": {1: 0, 2: 1, 3: 2, 4: 6, 5: 7},
}
DEFAULT_AXIS_MAPS: Final = {
    "spine": ("+X", "-Z", "+Y"),
    "shoulder.L": ("+X", "+Y", "+Z"),
    "forearm.L": ("+X", "+Y", "+Z"),
    "shoulder.R": ("-X", "-Y", "+Z"),
    "forearm.R": ("-X", "-Y", "+Z"),
}


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


def axis_map_matrix(axis_spec: Sequence[str]) -> Matrix3:
    """Build Blender-compatible signed coordinate permutation matrix."""
    if len(axis_spec) != 3:
        raise ValueError("axis map must contain exactly three axes")
    axis_index = {"X": 0, "Y": 1, "Z": 2}
    used: set[int] = set()
    rows: list[list[float]] = []
    for raw_token in axis_spec:
        token = str(raw_token).strip().upper()
        if len(token) != 2 or token[0] not in "+-" or token[1] not in axis_index:
            raise ValueError("axis values must look like +X, -Y, or +Z")
        source_index = axis_index[token[1]]
        if source_index in used:
            raise ValueError("each source axis must occur exactly once")
        used.add(source_index)
        row = [0.0, 0.0, 0.0]
        row[source_index] = 1.0 if token[0] == "+" else -1.0
        rows.append(row)
    result = np.asarray(rows, dtype=float)
    if abs(abs(float(np.linalg.det(result))) - 1.0) > 1.0e-6:
        raise ValueError("axis map must be an orthogonal signed permutation")
    return result


def mapped_sensor_quaternion(values: Iterable[float], axis_spec: Sequence[str]) -> Quaternion:
    quaternion = validate_input_quaternion(values)
    basis = axis_map_matrix(axis_spec)
    return matrix_to_quaternion(basis @ quaternion_to_matrix(quaternion) @ basis.T)


@dataclass(frozen=True)
class SensorSample:
    quaternion: Quaternion
    received_s: float
    generation: int


@dataclass(frozen=True)
class DatagramResult:
    published: bool
    pose_changed: bool
    neutral_captured: bool
    messages: tuple[str, ...]
    frame_header: str | None
    invalid_lines: int


class MotionCaptureModel:
    """Stateful equivalent of the Blender UDP calibration/pose pipeline."""

    def __init__(
        self,
        sensor_mapping: Mapping[str, int] | None = None,
        axis_maps: Mapping[str, Sequence[str]] | None = None,
        sensor_id_mode: str = "auto",
        smooth_alpha: float = 0.65,
    ) -> None:
        self.sensor_mapping: dict[str, int] = {}
        self.axis_maps: dict[str, tuple[str, str, str]] = {}
        self.sensor_id_mode = "auto"
        self.enabled_segments = frozenset(DEFAULT_ENABLED_SEGMENTS)
        self.active_sensor_id_mode: str | None = None
        self.smooth_alpha = float(smooth_alpha)

        self.latest_samples: dict[int, SensorSample] = {}
        self.sensor_counts: dict[int, int] = {}
        self.neutral_orientation: dict[str, Quaternion] = {}
        self.axis_alignment_quaternion: dict[str, Quaternion] = {
            name: IDENTITY_QUATERNION.copy() for name in SEGMENT_NAMES
        }
        self.drift_rate_rad_s: dict[str, Vector] = {
            name: np.zeros(3, dtype=float) for name in SEGMENT_NAMES
        }
        self.drift_reference_s: float | None = None
        self.segment_delta: dict[str, Quaternion] = {
            name: IDENTITY_QUATERNION.copy() for name in SEGMENT_NAMES
        }
        self.applied_generation: dict[str, int] = {}
        self.neutral_pending = False
        self.sample_generation = 0
        self.udp_packets = 0
        self.sample_frames = 0
        self.quaternion_count = 0
        self.invalid_lines = 0
        self.last_frame_header = ""

        self.configure(
            sensor_mapping or DEFAULT_SENSOR_MAPPING,
            axis_maps or DEFAULT_AXIS_MAPS,
            sensor_id_mode,
        )

    def configure(
        self,
        sensor_mapping: Mapping[str, int],
        axis_maps: Mapping[str, Sequence[str]],
        sensor_id_mode: str,
    ) -> None:
        if set(sensor_mapping) != set(SEGMENT_NAMES):
            raise ValueError("sensor mapping must define all five body segments")
        mapping = {name: int(sensor_mapping[name]) for name in SEGMENT_NAMES}
        if any(sensor_id < 0 or sensor_id > 255 for sensor_id in mapping.values()):
            raise ValueError("sensor IDs must be in range 0..255")
        if len(set(mapping.values())) != len(mapping):
            raise ValueError("each body segment must use a different sensor ID")

        mode = str(sensor_id_mode).strip().lower()
        if mode not in {"auto", "tca_channel", "sequential", "raw"}:
            raise ValueError("unknown sensor ID mode")

        normalized_axis_maps: dict[str, tuple[str, str, str]] = {}
        for name in SEGMENT_NAMES:
            values = tuple(str(value).strip().upper() for value in axis_maps[name])
            axis_map_matrix(values)
            normalized_axis_maps[name] = values  # type: ignore[assignment]

        self.sensor_mapping = mapping
        self.axis_maps = normalized_axis_maps
        self.sensor_id_mode = mode
        self.axis_alignment_quaternion = {
            name: IDENTITY_QUATERNION.copy() for name in SEGMENT_NAMES
        }
        self.drift_rate_rad_s = {
            name: np.zeros(3, dtype=float) for name in SEGMENT_NAMES
        }
        self.drift_reference_s = None
        self.reset_tracking(clear_samples=True)

    def set_enabled_segments(self, enabled_segments: Iterable[str]) -> None:
        """Track only selected segments and hold every other segment neutral."""
        enabled = frozenset(str(name) for name in enabled_segments)
        unknown = enabled.difference(SEGMENT_NAMES)
        if unknown:
            raise ValueError(
                "unknown enabled body segments: " + ", ".join(sorted(unknown))
            )
        if not enabled:
            raise ValueError("at least one body segment must be enabled")
        self.enabled_segments = enabled
        self.reset_tracking(clear_samples=True)

    def reset_tracking(self, clear_samples: bool = False) -> None:
        if clear_samples:
            self.latest_samples.clear()
            self.sensor_counts.clear()
            self.active_sensor_id_mode = None
        self.neutral_orientation.clear()
        self.applied_generation.clear()
        self.neutral_pending = False
        self.segment_delta = {
            name: IDENTITY_QUATERNION.copy() for name in SEGMENT_NAMES
        }

    def request_neutral(self) -> None:
        self.neutral_orientation.clear()
        self.applied_generation.clear()
        self.segment_delta = {
            name: IDENTITY_QUATERNION.copy() for name in SEGMENT_NAMES
        }
        self.neutral_pending = True

    def set_drift_compensation(
        self,
        drift_rates_rad_s: Mapping[str, Sequence[float]],
    ) -> None:
        """Set stationary angular drift estimated by the guided calibration."""
        if set(drift_rates_rad_s) != set(SEGMENT_NAMES):
            raise ValueError("drift profile must define all five body segments")
        normalized: dict[str, Vector] = {}
        for name in SEGMENT_NAMES:
            values = np.asarray(tuple(drift_rates_rad_s[name]), dtype=float)
            if values.shape != (3,) or not np.all(np.isfinite(values)):
                raise ValueError("each drift rate must contain three finite values")
            normalized[name] = values
        self.drift_rate_rad_s = normalized
        self.drift_reference_s = None

    def set_axis_alignment(
        self,
        alignment_quaternions: Mapping[str, Sequence[float]],
    ) -> None:
        """Set continuous sensor-to-segment coordinate corrections."""
        if set(alignment_quaternions) != set(SEGMENT_NAMES):
            raise ValueError("axis alignment must define all five body segments")
        self.axis_alignment_quaternion = {
            name: normalize_quaternion(alignment_quaternions[name])
            for name in SEGMENT_NAMES
        }

    def _mapped_segment_quaternion(
        self,
        segment_name: str,
        values: Iterable[float],
    ) -> Quaternion:
        mapped = mapped_sensor_quaternion(values, self.axis_maps[segment_name])
        alignment = self.axis_alignment_quaternion[segment_name]
        return quaternion_multiply(
            quaternion_multiply(alignment, mapped),
            quaternion_inverse(alignment),
        )

    def set_guided_calibration(
        self,
        axis_maps: Mapping[str, Sequence[str]],
        drift_rates_rad_s: Mapping[str, Sequence[float]],
        neutral_raw_quaternions: Mapping[str, Sequence[float]],
        reference_s: float,
        alignment_quaternions: Mapping[str, Sequence[float]] | None = None,
    ) -> None:
        """Apply a three-pose result and use its averaged N-pose immediately."""
        self.configure(
            self.sensor_mapping,
            axis_maps,
            self.sensor_id_mode,
        )
        if alignment_quaternions is not None:
            self.set_axis_alignment(alignment_quaternions)
        self.set_drift_compensation(drift_rates_rad_s)
        captured = {
            name: self._mapped_segment_quaternion(
                name, neutral_raw_quaternions[name]
            )
            for name in SEGMENT_NAMES
        }
        self.neutral_orientation = captured
        self.drift_reference_s = float(reference_s)
        self.segment_delta = {
            name: IDENTITY_QUATERNION.copy() for name in SEGMENT_NAMES
        }
        self.applied_generation.clear()
        self.neutral_pending = False

    def _canonicalize(
        self, raw_samples: list[tuple[int, Quaternion]]
    ) -> list[tuple[int, Quaternion]]:
        if not raw_samples:
            return []
        enabled_sensor_ids = {
            self.sensor_mapping[name] for name in self.enabled_segments
        }
        if self.sensor_id_mode == "raw":
            self.active_sensor_id_mode = "raw"
            return [
                (sensor_id, quaternion)
                for sensor_id, quaternion in raw_samples
                if sensor_id in enabled_sensor_ids
            ]

        if self.active_sensor_id_mode is None:
            if self.sensor_id_mode != "auto":
                self.active_sensor_id_mode = self.sensor_id_mode
            else:
                raw_ids = {sensor_id for sensor_id, _quaternion in raw_samples}
                candidates = [
                    mode
                    for mode, profile in RAW_SENSOR_ID_PROFILES.items()
                    if raw_ids.issubset(profile)
                ]
                if len(candidates) != 1:
                    return []
                self.active_sensor_id_mode = candidates[0]

        profile = RAW_SENSOR_ID_PROFILES[self.active_sensor_id_mode]
        return [
            (profile[sensor_id], quaternion)
            for sensor_id, quaternion in raw_samples
            if sensor_id in profile and profile[sensor_id] in enabled_sensor_ids
        ]

    def handle_datagram(self, payload: bytes | str, received_s: float) -> DatagramResult:
        text = (
            payload.decode("utf-8", errors="replace")
            if isinstance(payload, bytes)
            else str(payload)
        ).strip("\x00\r\n")
        raw_samples: list[tuple[int, Quaternion]] = []
        messages: list[str] = []
        frame_header: str | None = None
        invalid_lines = 0

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) == 6 and parts[0].upper() in {"Q", "QUAT"}:
                try:
                    sensor_id = int(parts[1])
                    if not 0 <= sensor_id <= 255:
                        raise ValueError
                    quaternion = validate_input_quaternion(float(value) for value in parts[2:6])
                except (ValueError, OverflowError):
                    invalid_lines += 1
                else:
                    raw_samples.append((sensor_id, quaternion))
                continue
            if line.upper().startswith("FRAME "):
                frame_header = line
                continue
            if line.startswith(("ACK", "STATUS", "SENSOR", "PONG", "ERR")):
                messages.append(line)
                continue
            invalid_lines += 1

        samples = self._canonicalize(raw_samples)
        published = bool(samples)
        if published:
            self.sample_generation += 1
            generation = self.sample_generation
            self.sample_frames += 1
            for sensor_id, quaternion in samples:
                self.latest_samples[sensor_id] = SensorSample(
                    quaternion=quaternion,
                    received_s=float(received_s),
                    generation=generation,
                )
                self.sensor_counts[sensor_id] = self.sensor_counts.get(sensor_id, 0) + 1
                self.quaternion_count += 1

        self.udp_packets += 1
        self.invalid_lines += invalid_lines
        if frame_header is not None:
            self.last_frame_header = frame_header

        neutral_captured = self._try_capture_neutral(float(received_s))
        pose_changed = neutral_captured
        if not neutral_captured:
            pose_changed = self._update_segment_deltas() or pose_changed
        return DatagramResult(
            published=published,
            pose_changed=pose_changed,
            neutral_captured=neutral_captured,
            messages=tuple(messages),
            frame_header=frame_header,
            invalid_lines=invalid_lines,
        )

    def _try_capture_neutral(self, now_s: float) -> bool:
        if not self.neutral_pending:
            return False
        samples: dict[str, SensorSample] = {}
        for segment_name, sensor_id in self.sensor_mapping.items():
            if segment_name not in self.enabled_segments:
                continue
            sample = self.latest_samples.get(sensor_id)
            if sample is None:
                return False
            samples[segment_name] = sample
        timestamps = [sample.received_s for sample in samples.values()]
        if now_s - min(timestamps) > 0.40 or max(timestamps) - min(timestamps) > 0.25:
            return False
        if len({sample.generation for sample in samples.values()}) != 1:
            return False

        captured: dict[str, Quaternion] = {
            name: IDENTITY_QUATERNION.copy()
            for name in SEGMENT_NAMES if name not in self.enabled_segments
        }
        try:
            for segment_name, sample in samples.items():
                captured[segment_name] = self._mapped_segment_quaternion(
                    segment_name, sample.quaternion
                )
        except ValueError:
            return False

        self.neutral_orientation = captured
        self.drift_reference_s = float(sum(timestamps) / len(timestamps))
        self.segment_delta = {
            name: IDENTITY_QUATERNION.copy() for name in SEGMENT_NAMES
        }
        self.applied_generation = {
            name: sample.generation for name, sample in samples.items()
        }
        self.neutral_pending = False
        return True

    def _update_segment_deltas(self) -> bool:
        if self.neutral_pending or set(self.neutral_orientation) != set(SEGMENT_NAMES):
            return False
        changed = False
        for segment_name, sensor_id in self.sensor_mapping.items():
            if segment_name not in self.enabled_segments:
                continue
            sample = self.latest_samples.get(sensor_id)
            if sample is None or self.applied_generation.get(segment_name) == sample.generation:
                continue
            current = self._mapped_segment_quaternion(
                segment_name, sample.quaternion
            )
            if self.drift_reference_s is not None:
                elapsed_s = max(0.0, sample.received_s - self.drift_reference_s)
                drift = quaternion_from_rotation_vector(
                    self.drift_rate_rad_s[segment_name] * elapsed_s
                )
                current = quaternion_multiply(quaternion_inverse(drift), current)
            target = quaternion_multiply(
                current, quaternion_inverse(self.neutral_orientation[segment_name])
            )
            self.segment_delta[segment_name] = quaternion_slerp(
                self.segment_delta[segment_name], target, self.smooth_alpha
            )
            self.applied_generation[segment_name] = sample.generation
            changed = True
        return changed

    def orientations(self) -> dict[str, Quaternion]:
        return {name: quaternion.copy() for name, quaternion in self.segment_delta.items()}


@dataclass(frozen=True)
class BodyPose:
    tracked_segments: dict[str, tuple[Vector, Vector]]
    static_segments: tuple[tuple[Vector, Vector], ...]
    joints: tuple[Vector, ...]
    torso_faces: tuple[tuple[Vector, ...], ...]
    head_center: Vector
    axis_origins: dict[str, Vector]
    axis_frames: dict[str, Matrix3]


def _unit(vector: Vector) -> Vector:
    norm = float(np.linalg.norm(vector))
    if norm < 1.0e-9:
        raise ValueError("zero direction")
    return vector / norm


def _segment_rest_basis(direction: Vector) -> Matrix3:
    local_z = _unit(direction)
    reference_y = np.array((0.0, 1.0, 0.0))
    if abs(float(np.dot(reference_y, local_z))) > 0.95:
        reference_y = np.array((0.0, 0.0, 1.0))
    local_x = _unit(np.cross(reference_y, local_z))
    local_y = _unit(np.cross(local_z, local_x))
    return np.column_stack((local_x, local_y, local_z))

# Physical sensor frames in the reference N-pose.  Matrix columns are the
# sensor X/Y/Z directions in application coordinates (X right, Y forward,
# Z up).  The arm mounting was described with the arms pointing forward, so
# Rx(-90 deg) brings that frame back to the N-pose used by the mannequin.
_ARM_FORWARD_TO_NEUTRAL: Final[Matrix3] = np.array(
    ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0)),
    dtype=float,
)
_LEFT_ARM_FORWARD_FRAME: Final[Matrix3] = np.eye(3, dtype=float)
_RIGHT_ARM_FORWARD_FRAME: Final[Matrix3] = np.diag((-1.0, -1.0, 1.0))
SENSOR_NEUTRAL_AXIS_FRAMES: Final = {
    # Torso: X right, Y up, Z backward.
    "spine": np.array(
        ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
        dtype=float,
    ),
    # Left arm when forward: X right, Y toward fingers, Z up.
    "shoulder.L": _ARM_FORWARD_TO_NEUTRAL @ _LEFT_ARM_FORWARD_FRAME,
    "forearm.L": _ARM_FORWARD_TO_NEUTRAL @ _LEFT_ARM_FORWARD_FRAME,
    # Right arm when forward: X left, Y from fingers to elbow, Z up.
    "shoulder.R": _ARM_FORWARD_TO_NEUTRAL @ _RIGHT_ARM_FORWARD_FRAME,
    "forearm.R": _ARM_FORWARD_TO_NEUTRAL @ _RIGHT_ARM_FORWARD_FRAME,
}

def compute_body_pose(orientations: Mapping[str, Quaternion]) -> BodyPose:
    """Return an articulated upper-body mannequin for the five tracked segments."""
    rotations = {
        name: quaternion_to_matrix(orientations.get(name, IDENTITY_QUATERNION))
        for name in SEGMENT_NAMES
    }
    pelvis = np.array((0.0, 0.0, 0.92))
    spine_vector = np.array((0.0, 0.0, 0.58))
    chest = pelvis + rotations["spine"] @ spine_vector

    left_shoulder = pelvis + rotations["spine"] @ np.array((-0.29, 0.0, 0.52))
    right_shoulder = pelvis + rotations["spine"] @ np.array((0.29, 0.0, 0.52))
    left_upper_vector = np.array((-0.035, 0.0, -0.43))
    right_upper_vector = np.array((0.035, 0.0, -0.43))
    forearm_vector = np.array((0.0, 0.0, -0.39))

    left_elbow = left_shoulder + rotations["shoulder.L"] @ left_upper_vector
    right_elbow = right_shoulder + rotations["shoulder.R"] @ right_upper_vector
    left_wrist = left_elbow + rotations["forearm.L"] @ forearm_vector
    right_wrist = right_elbow + rotations["forearm.R"] @ forearm_vector

    tracked = {
        "spine": (pelvis, chest),
        "shoulder.L": (left_shoulder, left_elbow),
        "forearm.L": (left_elbow, left_wrist),
        "shoulder.R": (right_shoulder, right_elbow),
        "forearm.R": (right_elbow, right_wrist),
    }

    neck = pelvis + rotations["spine"] @ np.array((0.0, 0.0, 0.68))
    head_center = pelvis + rotations["spine"] @ np.array((0.0, 0.0, 0.84))
    left_hip = pelvis + np.array((-0.13, 0.0, -0.04))
    right_hip = pelvis + np.array((0.13, 0.0, -0.04))
    left_knee = np.array((-0.14, 0.0, 0.48))
    right_knee = np.array((0.14, 0.0, 0.48))
    left_ankle = np.array((-0.14, 0.0, 0.07))
    right_ankle = np.array((0.14, 0.0, 0.07))
    static_segments = (
        (chest, neck),
        (left_shoulder, right_shoulder),
        (pelvis, left_hip),
        (pelvis, right_hip),
        (left_hip, left_knee),
        (left_knee, left_ankle),
        (right_hip, right_knee),
        (right_knee, right_ankle),
        (left_ankle, left_ankle + np.array((0.0, -0.13, 0.0))),
        (right_ankle, right_ankle + np.array((0.0, -0.13, 0.0))),
    )

    neutral_corners = np.array(
        (
            (-0.17, -0.10, 0.0),
            (0.17, -0.10, 0.0),
            (0.17, 0.10, 0.0),
            (-0.17, 0.10, 0.0),
            (-0.28, -0.12, 0.52),
            (0.28, -0.12, 0.52),
            (0.28, 0.12, 0.52),
            (-0.28, 0.12, 0.52),
        )
    )
    corners = tuple(pelvis + rotations["spine"] @ corner for corner in neutral_corners)
    torso_faces = tuple(
        tuple(corners[index] for index in face)
        for face in (
            (0, 1, 2, 3),
            (4, 5, 6, 7),
            (0, 1, 5, 4),
            (1, 2, 6, 5),
            (2, 3, 7, 6),
            (3, 0, 4, 7),
        )
    )

    axis_origins = {
        name: (start + end) * 0.5 for name, (start, end) in tracked.items()
    }
    axis_frames = {
        name: rotations[name] @ SENSOR_NEUTRAL_AXIS_FRAMES[name]
        for name in SEGMENT_NAMES
    }
    joints = (
        pelvis,
        chest,
        neck,
        left_shoulder,
        right_shoulder,
        left_elbow,
        right_elbow,
        left_wrist,
        right_wrist,
        left_hip,
        right_hip,
        left_knee,
        right_knee,
        left_ankle,
        right_ankle,
    )
    return BodyPose(
        tracked_segments=tracked,
        static_segments=static_segments,
        joints=joints,
        torso_faces=torso_faces,
        head_center=head_center,
        axis_origins=axis_origins,
        axis_frames=axis_frames,
    )
