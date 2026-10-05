"""A -> T -> forward -> palms together -> A calibration and profiles."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import itertools
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from .quaternion_utils import average_quaternions, quaternion_to_rotation_vector
from .orientation_frames import SegmentCalibration
from numpy.typing import NDArray

from .mocap_core import (
    DEFAULT_AXIS_MAPS,
    SEGMENT_NAMES,
    axis_map_matrix,
    mapped_sensor_quaternion,
    matrix_to_quaternion,
    normalize_quaternion,
    quaternion_inverse,
    quaternion_from_rotation_vector,
    quaternion_multiply,
    quaternion_to_matrix,
)


Quaternion = NDArray[np.float64]
Vector = NDArray[np.float64]
PROFILE_SCHEMA = "neuromorph-pyqt-mocap-calibration"
PROFILE_VERSION = 7
SUPPORTED_PROFILE_VERSIONS = frozenset((1, 2, 3, 4, 5, 6, PROFILE_VERSION))
POSE_NAMES = ("a_pose", "t_pose", "forward_pose", "p_pose", "return_a_pose")
FIT_POSE_NAMES = ("t_pose", "forward_pose", "p_pose")
MIN_SAMPLES_PER_SEGMENT = 15
# A robust radius around the mean: ignores isolated spikes, catches movement
# or unstable drift during a pose that is explicitly required to be stationary.
MAX_POSE_SPREAD_DEG = 15.0
MAX_CLOSURE_ERROR_DEG = 30.0

NEUTRAL_DIRECTIONS = {
    "spine": np.array((0.0, 0.0, 1.0)),
    "shoulder.L": np.array((0.0, 0.0, -1.0)),
    "forearm.L": np.array((0.0, 0.0, -1.0)),
    "shoulder.R": np.array((0.0, 0.0, -1.0)),
    "forearm.R": np.array((0.0, 0.0, -1.0)),
}
TARGET_DIRECTIONS = {
    "t_pose": {
        "spine": np.array((0.0, 0.0, 1.0)),
        "shoulder.L": np.array((-1.0, 0.0, 0.0)),
        "forearm.L": np.array((-1.0, 0.0, 0.0)),
        "shoulder.R": np.array((1.0, 0.0, 0.0)),
        "forearm.R": np.array((1.0, 0.0, 0.0)),
    },
    "forward_pose": {
        name: (
            np.array((0.0, 0.0, 1.0))
            if name == "spine"
            else np.array((0.0, 1.0, 0.0))
        )
        for name in SEGMENT_NAMES
    },
    "return_a_pose": NEUTRAL_DIRECTIONS,
    # Upper arms down; forearms upward/inward, internal elbow angle 45 deg.
    # Targets describe segment directions, not anthropometric wrist positions.
    "p_pose": {
        **NEUTRAL_DIRECTIONS,
        "forearm.L": np.array((math.sqrt(.5), 0., math.sqrt(.5))),
        "forearm.R": np.array((-math.sqrt(.5), 0., math.sqrt(.5))),
    },
}

EXPECTED_ROTATION_VECTORS = {
    "t_pose": {
        "shoulder.L": np.array((0.0, math.pi / 2.0, 0.0)),
        "forearm.L": np.array((0.0, math.pi / 2.0, 0.0)),
        "shoulder.R": np.array((0.0, -math.pi / 2.0, 0.0)),
        "forearm.R": np.array((0.0, -math.pi / 2.0, 0.0)),
    },
    "forward_pose": {
        name: np.array((math.pi / 2.0, 0.0, 0.0))
        for name in SEGMENT_NAMES
        if name != "spine"
    },
}
MIN_ALIGNMENT_MOTION_RAD = math.radians(25.0)
MIN_ALIGNMENT_AXIS_SEPARATION = math.sin(math.radians(25.0))


@dataclass(frozen=True)
class CapturedPose:
    name: str
    average: dict[str, Quaternion]
    first: dict[str, Quaternion]
    last: dict[str, Quaternion]
    sample_counts: dict[str, int]
    started_s: float
    ended_s: float
    spread_deg: dict[str, float] = field(default_factory=dict)
    drift_interval_s: dict[str, float] = field(default_factory=dict)


class PoseRecorder:
    """Collect each sensor generation once during one stationary pose."""

    def __init__(self) -> None:
        self.values: dict[str, list[Quaternion]] = {
            name: [] for name in SEGMENT_NAMES
        }
        self.timestamps: dict[str, list[float]] = {
            name: [] for name in SEGMENT_NAMES
        }
        self.last_generation: dict[str, int] = {}
        self.started_s: float | None = None
        self.ended_s: float | None = None

    def add_snapshot(
        self,
        snapshot: Mapping[str, tuple[Sequence[float], float, int]],
    ) -> None:
        for name in SEGMENT_NAMES:
            if name not in snapshot:
                continue
            quaternion, timestamp_s, generation = snapshot[name]
            if self.last_generation.get(name) == generation:
                continue
            self.last_generation[name] = generation
            self.values[name].append(normalize_quaternion(quaternion))
            self.timestamps[name].append(float(timestamp_s))
            if self.started_s is None or timestamp_s < self.started_s:
                self.started_s = float(timestamp_s)
            if self.ended_s is None or timestamp_s > self.ended_s:
                self.ended_s = float(timestamp_s)

    def finish(self, name: str) -> CapturedPose:
        missing = [
            segment
            for segment in SEGMENT_NAMES
            if len(self.values[segment]) < MIN_SAMPLES_PER_SEGMENT
        ]
        if missing:
            details = ", ".join(
                f"{segment}: {len(self.values[segment])}"
                for segment in missing
            )
            raise ValueError(
                f"недостаточно кадров (нужно {MIN_SAMPLES_PER_SEGMENT}): {details}"
            )
        if self.started_s is None or self.ended_s is None:
            raise ValueError("нет временных меток калибровки")

        average: dict[str, Quaternion] = {}
        first: dict[str, Quaternion] = {}
        last: dict[str, Quaternion] = {}
        counts: dict[str, int] = {}
        spread: dict[str, float] = {}
        drift_intervals: dict[str, float] = {}
        for segment in SEGMENT_NAMES:
            values = self.values[segment]
            edge_count = max(3, len(values) // 5)
            average[segment] = average_quaternions(values)
            first[segment] = average_quaternions(values[:edge_count])
            last[segment] = average_quaternions(values[-edge_count:])
            counts[segment] = len(values)
            dots = np.clip(np.abs(np.stack(values) @ average[segment]), 0., 1.)
            spread[segment] = float(np.percentile(np.degrees(2 * np.arccos(dots)), 95))
            timestamps = self.timestamps[segment]
            drift_intervals[segment] = float(np.mean(timestamps[-edge_count:]) - np.mean(timestamps[:edge_count]))
        unstable = [segment for segment in SEGMENT_NAMES if spread[segment] > MAX_POSE_SPREAD_DEG]
        if unstable:
            details = ", ".join(f"{segment}: {spread[segment]:.1f}°" for segment in unstable)
            raise ValueError(
                f"Поза нестабильна: {details} (допуск {MAX_POSE_SPREAD_DEG:g}°). "
                "Повторите этот этап, удерживая корпус и руки неподвижно. "
                "Если показания меняются без движения, проверьте крепление и дрейф датчиков."
            )
        return CapturedPose(
            name=name,
            average=average,
            first=first,
            last=last,
            sample_counts=counts,
            started_s=self.started_s,
            ended_s=self.ended_s,
            spread_deg=spread,
            drift_interval_s=drift_intervals,
        )


@dataclass(frozen=True)
class CalibrationResult:
    axis_maps: dict[str, tuple[str, str, str]]
    axis_alignment_quaternions: dict[str, tuple[float, float, float, float]]
    drift_rates_rad_s: dict[str, tuple[float, float, float]]
    scores_deg: dict[str, float]
    preferred_scores_deg: dict[str, float]
    captures: dict[str, CapturedPose]
    reference_s: float
    created_at: str
    method: str = "guided_poses"
    repeatability_deg: dict[str, float] = field(default_factory=dict)
    closure_error_deg: dict[str, float] = field(default_factory=dict)
    pose_errors_deg: dict[str, dict[str, float]] = field(default_factory=dict)

    @property
    def max_drift_deg_s(self) -> float:
        return max(
            math.degrees(float(np.linalg.norm(rate)))
            for rate in self.drift_rates_rad_s.values()
        )

    @property
    def max_axis_alignment_deg(self) -> float:
        angles = []
        for quaternion in self.axis_alignment_quaternions.values():
            value = normalize_quaternion(quaternion)
            cosine = min(1.0, max(-1.0, abs(float(value[0]))))
            angles.append(math.degrees(2.0 * math.acos(cosine)))
        return max(angles)


def _candidate_axis_maps() -> tuple[tuple[str, str, str], ...]:
    candidates: list[tuple[str, str, str]] = []
    for permutation in itertools.permutations(("X", "Y", "Z")):
        for signs in itertools.product((1, -1), repeat=3):
            spec = tuple(
                ("+" if sign > 0 else "-") + axis
                for sign, axis in zip(signs, permutation, strict=True)
            )
            if float(np.linalg.det(axis_map_matrix(spec))) > 0.5:
                candidates.append(spec)  # type: ignore[arg-type]
    return tuple(candidates)


AXIS_MAP_CANDIDATES = _candidate_axis_maps()


def _aligned_sensor_quaternion(
    values: Sequence[float],
    spec: Sequence[str],
    alignment: NDArray[np.float64],
) -> Quaternion:
    mapped = mapped_sensor_quaternion(values, spec)
    return matrix_to_quaternion(
        alignment @ quaternion_to_matrix(mapped) @ alignment.T
    )


def _estimate_axis_alignment(
    segment: str,
    spec: Sequence[str],
    captures: Mapping[str, CapturedPose],
) -> NDArray[np.float64]:
    """Initialize from A→T/forward, then refine forearms using the P-pose."""
    if segment == "spine":
        return np.eye(3, dtype=float)

    neutral = mapped_sensor_quaternion(
        captures["a_pose"].average[segment], spec
    )
    observed_axes: list[Vector] = []
    target_axes: list[Vector] = []
    # Two non-collinear rotations estimate alignment; the final A-pose
    # independently validates return to the initial direction.
    for pose_name in ("t_pose", "forward_pose"):
        current = mapped_sensor_quaternion(
            captures[pose_name].average[segment], spec
        )
        delta = quaternion_multiply(current, quaternion_inverse(neutral))
        vector = quaternion_to_rotation_vector(delta)
        norm = float(np.linalg.norm(vector))
        if norm < MIN_ALIGNMENT_MOTION_RAD:
            raise ValueError(f"{segment}: недостаточно движения в {pose_name}; повторите калибровку")
        expected = EXPECTED_ROTATION_VECTORS[pose_name][segment]
        observed_axes.append(vector / norm)
        target_axes.append(expected / float(np.linalg.norm(expected)))

    if (
        float(np.linalg.norm(np.cross(*observed_axes)))
        < MIN_ALIGNMENT_AXIS_SEPARATION
    ):
        raise ValueError(f"{segment}: T-поза и руки вперёд не задают разные направления; повторите калибровку")

    source = np.column_stack(observed_axes)
    target = np.column_stack(target_axes)
    covariance = target @ source.T
    left, _singular_values, right_transposed = np.linalg.svd(covariance)
    correction = left @ right_transposed
    if float(np.linalg.det(correction)) < 0.0:
        left[:, -1] *= -1.0
        correction = left @ right_transposed
    if segment.startswith("forearm."):
        correction = _refine_forearm_alignment(segment, spec, captures, correction)
    return correction


def _refine_forearm_alignment(segment, spec, captures, initial):
    """Fit one rigid alignment to T/forward/P directions without assuming twist.

    Damped least squares on SO(3); each update is a proper rotation. The
    final A-pose stays an independent check and is not fitted here.
    """
    neutral = mapped_sensor_quaternion(captures["a_pose"].average[segment], spec)
    poses = ("t_pose", "forward_pose", "p_pose")
    deltas = [quaternion_to_matrix(quaternion_multiply(
        mapped_sensor_quaternion(captures[name].average[segment], spec),
        quaternion_inverse(neutral))) for name in poses]

    def residual(alignment):
        return np.concatenate([
            alignment @ delta @ alignment.T @ NEUTRAL_DIRECTIONS[segment]
            - TARGET_DIRECTIONS[name][segment]
            for name, delta in zip(poses, deltas)
        ])

    alignment = initial.copy()
    epsilon = 1e-5
    for _ in range(20):
        error = residual(alignment)
        if np.linalg.norm(error) < 1e-10:
            break
        jacobian = np.column_stack([
            (residual(quaternion_to_matrix(quaternion_from_rotation_vector(axis * epsilon))
                      @ alignment) - error) / epsilon
            for axis in np.eye(3)
        ])
        step = np.linalg.solve(jacobian.T @ jacobian + 1e-4 * np.eye(3),
                               -jacobian.T @ error)
        norm = float(np.linalg.norm(step))
        if norm < 1e-8:
            break
        step *= min(1., .15 / norm)
        for factor in (1., .5, .25, .125):
            candidate = quaternion_to_matrix(quaternion_from_rotation_vector(step * factor)) @ alignment
            if np.linalg.norm(residual(candidate)) < np.linalg.norm(error):
                alignment = candidate
                break
        else:
            break
    return alignment


def _pose_direction_errors_deg(
    segment: str,
    spec: Sequence[str],
    captures: Mapping[str, CapturedPose],
    alignment: NDArray[np.float64] | None = None,
    *, reference_pose: str = "a_pose", pose_names=POSE_NAMES[1:],
) -> dict[str, float]:
    if alignment is None:
        alignment = np.eye(3, dtype=float)
    neutral = _aligned_sensor_quaternion(
        captures[reference_pose].average[segment],
        spec,
        alignment,
    )
    errors: dict[str, float] = {}
    for pose_name in pose_names:
        current = _aligned_sensor_quaternion(
            captures[pose_name].average[segment],
            spec,
            alignment,
        )
        delta = quaternion_multiply(current, quaternion_inverse(neutral))
        predicted = quaternion_to_matrix(delta) @ NEUTRAL_DIRECTIONS[segment]
        target = TARGET_DIRECTIONS[pose_name][segment]
        cosine = min(1.0, max(-1.0, float(np.dot(predicted, target))))
        errors[pose_name] = math.degrees(math.acos(cosine))
    return errors


def _direction_error_deg(segment, spec, captures, alignment=None, *,
                         reference_pose="a_pose", pose_names=POSE_NAMES[1:]) -> float:
    errors = _pose_direction_errors_deg(segment, spec, captures, alignment,
        reference_pose=reference_pose, pose_names=pose_names)
    return math.sqrt(sum(error * error for error in errors.values()) / len(errors))


def _select_axis_map(
    segment: str,
    captures: Mapping[str, CapturedPose],
    preferred: tuple[str, str, str],
) -> tuple[tuple[str, str, str], float, float]:
    preferred_score = _direction_error_deg(segment, preferred, captures, pose_names=FIT_POSE_NAMES)
    if segment == "spine":
        return preferred, preferred_score, preferred_score
    ranked = sorted(
        (
            (_direction_error_deg(segment, candidate, captures, pose_names=FIT_POSE_NAMES), candidate)
            for candidate in AXIS_MAP_CANDIDATES
        ),
        key=lambda item: item[0],
    )
    best_score, best = ranked[0]
    # Keep the known mounting description unless real poses improve it clearly.
    if best_score + 3.0 >= preferred_score:
        return preferred, preferred_score, preferred_score
    return best, best_score, preferred_score


def calibrate_five_poses(
    captures: Mapping[str, CapturedPose],
    preferred_axis_maps: Mapping[str, Sequence[str]] = DEFAULT_AXIS_MAPS,
    *, enabled_segments=None,
) -> CalibrationResult:
    """Fit A/T/forward/P, validate return to A and estimate stationary drift."""
    if set(captures) != set(POSE_NAMES):
        raise ValueError(
            "calibration requires initial A, T, forward, P, and final A captures"
        )

    enabled = set(SEGMENT_NAMES if enabled_segments is None else enabled_segments)
    if enabled.difference(SEGMENT_NAMES):
        raise ValueError("Unknown enabled calibration segment")
    axis_maps: dict[str, tuple[str, str, str]] = {}
    preferred_scores: dict[str, float] = {}
    for segment in SEGMENT_NAMES:
        preferred = tuple(preferred_axis_maps[segment])
        if segment not in enabled:
            axis_maps[segment], preferred_scores[segment] = preferred, 0.
            continue
        selected, _score, preferred_score = _select_axis_map(
            segment, captures, preferred  # type: ignore[arg-type]
        )
        axis_maps[segment] = selected
        preferred_scores[segment] = preferred_score

    alignment_matrices: dict[str, NDArray[np.float64]] = {}
    axis_alignment_quaternions: dict[
        str, tuple[float, float, float, float]
    ] = {}
    scores: dict[str, float] = {}
    closure: dict[str, float] = {}
    pose_errors: dict[str, dict[str, float]] = {}
    for segment in SEGMENT_NAMES:
        if segment not in enabled:
            alignment_matrices[segment] = np.eye(3)
            axis_alignment_quaternions[segment] = (1., 0., 0., 0.)
            scores[segment] = closure[segment] = 0.
            pose_errors[segment] = {}
            continue
        alignment = _estimate_axis_alignment(
            segment, axis_maps[segment], captures
        )
        unaligned_score = _direction_error_deg(
            segment, axis_maps[segment], captures, pose_names=FIT_POSE_NAMES
        )
        aligned_score = _direction_error_deg(
            segment, axis_maps[segment], captures, alignment, pose_names=FIT_POSE_NAMES
        )
        if aligned_score > unaligned_score:
            alignment = np.eye(3, dtype=float)
            aligned_score = unaligned_score
        alignment_matrices[segment] = alignment
        quaternion = matrix_to_quaternion(alignment)
        axis_alignment_quaternions[segment] = tuple(
            float(value) for value in quaternion
        )
        # Final A is held out from fitting. It becomes the live neutral, so
        # evaluate the deployed transform rather than the initial reference.
        closure[segment] = _pose_direction_errors_deg(segment, axis_maps[segment], captures,
            alignment, pose_names=("return_a_pose",))["return_a_pose"]
        if closure[segment] > MAX_CLOSURE_ERROR_DEG:
            raise ValueError(f"{segment}: возврат в A-позу отличается на {closure[segment]:.1f}°. "
                "Повторите калибровку, не поворачивая корпус и возвращая руки в исходную позу.")
        pose_errors[segment] = _pose_direction_errors_deg(segment, axis_maps[segment], captures,
            alignment, reference_pose="return_a_pose", pose_names=FIT_POSE_NAMES)
        scores[segment] = _direction_error_deg(segment, axis_maps[segment], captures, alignment,
            reference_pose="return_a_pose", pose_names=FIT_POSE_NAMES)
        preferred_scores[segment] = _direction_error_deg(segment, preferred_axis_maps[segment], captures,
            reference_pose="return_a_pose", pose_names=FIT_POSE_NAMES)

    neutral_capture = captures["return_a_pose"]
    duration_s = neutral_capture.ended_s - neutral_capture.started_s
    if duration_s < 1.0:
        raise ValueError("Final A-pose capture is too short to estimate drift")
    drift_rates: dict[str, tuple[float, float, float]] = {}
    for segment in SEGMENT_NAMES:
        if segment not in enabled:
            drift_rates[segment] = (0., 0., 0.)
            continue
        first = _aligned_sensor_quaternion(
            neutral_capture.first[segment],
            axis_maps[segment],
            alignment_matrices[segment],
        )
        last = _aligned_sensor_quaternion(
            neutral_capture.last[segment],
            axis_maps[segment],
            alignment_matrices[segment],
        )
        drift = quaternion_multiply(last, quaternion_inverse(first))
        interval = neutral_capture.drift_interval_s.get(segment, duration_s)
        if interval <= 0. or not math.isfinite(interval):
            raise ValueError(f"{segment}: некорректный интервал оценки дрейфа")
        rate = quaternion_to_rotation_vector(drift) / interval
        drift_rates[segment] = tuple(float(value) for value in rate)

    reference_s = (neutral_capture.started_s + neutral_capture.ended_s) * 0.5
    return CalibrationResult(
        axis_maps=axis_maps,
        axis_alignment_quaternions=axis_alignment_quaternions,
        drift_rates_rad_s=drift_rates,
        scores_deg=scores,
        preferred_scores_deg=preferred_scores,
        captures=dict(captures),
        reference_s=reference_s,
        created_at=datetime.now(timezone.utc).isoformat(),
        closure_error_deg=closure,
        pose_errors_deg=pose_errors,
    )


def captured_pose_document(capture: CapturedPose) -> dict[str, object]:
    """Serializable raw pose evidence, including attempts that were rejected."""
    return {
        "average_quaternions_wxyz": {s: [float(v) for v in capture.average[s]] for s in SEGMENT_NAMES},
        "first_quaternions_wxyz": {s: [float(v) for v in capture.first[s]] for s in SEGMENT_NAMES},
        "last_quaternions_wxyz": {s: [float(v) for v in capture.last[s]] for s in SEGMENT_NAMES},
        "sample_counts": dict(capture.sample_counts),
        "spread_deg": dict(capture.spread_deg),
        "drift_interval_s": dict(capture.drift_interval_s),
        "started_s": capture.started_s,
        "ended_s": capture.ended_s,
    }


def profile_document(
    application_config: Mapping[str, object],
    result: CalibrationResult,
) -> dict[str, object]:
    """Build a versioned, human-readable calibration profile."""
    poses = {name: captured_pose_document(capture) for name, capture in result.captures.items()}
    return {
        "schema": PROFILE_SCHEMA,
        "version": PROFILE_VERSION,
        "created_at": result.created_at,
        "application": dict(application_config),
        "calibration": {
            "method": result.method,
            "quaternion_convention": "Hamilton wxyz; q_AB maps B to A; right factor first",
            "drift_compensation_applied": False,
            "axis_alignment_role": "world_basis_alignment_after_axis_map",
            "segment_calibrations": {
                s: {
                    "q_world_alignment": list(transform.q_world_alignment),
                    "q_sensor_segment": list(transform.q_sensor_segment),
                }
                for s in SEGMENT_NAMES
                for transform in (SegmentCalibration.from_pose(
                    mapped_sensor_quaternion(result.captures["return_a_pose"].average[s], result.axis_maps[s]),
                    q_world_alignment=result.axis_alignment_quaternions[s]),)
            },
            "neutral_raw_quaternions_wxyz": {
                s: [float(v) for v in result.captures["return_a_pose"].average[s]] for s in SEGMENT_NAMES
            },
            "repeatability_deg": result.repeatability_deg,
            "closure_error_deg": result.closure_error_deg,
            "pose_errors_deg": result.pose_errors_deg,
            "score_reference_pose": "return_a_pose",
            "axis_maps": {
                name: list(result.axis_maps[name]) for name in SEGMENT_NAMES
            },
            "axis_alignment_quaternions_wxyz": {
                name: list(result.axis_alignment_quaternions[name])
                for name in SEGMENT_NAMES
            },
            "drift_rates_rad_s": {
                name: list(result.drift_rates_rad_s[name])
                for name in SEGMENT_NAMES
            },
            "direction_error_deg": result.scores_deg,
            "preferred_error_deg": result.preferred_scores_deg,
            "reference_s": result.reference_s,
            "poses": poses,
        },
    }


def save_profile(path: str | Path, document: Mapping[str, object]) -> None:
    destination = Path(path)
    destination.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_profile(path: str | Path) -> dict[str, object]:
    source = Path(path)
    document = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("корень профиля должен быть JSON-объектом")
    if document.get("schema") != PROFILE_SCHEMA:
        raise ValueError("это не профиль Neuromorph PyQt mocap")
    if document.get("version") not in SUPPORTED_PROFILE_VERSIONS:
        raise ValueError("неподдерживаемая версия профиля")
    application = document.get("application")
    calibration = document.get("calibration")
    if not isinstance(application, dict) or not isinstance(calibration, dict):
        raise ValueError("в профиле отсутствуют application/calibration")
    axis_maps = calibration.get("axis_maps")
    alignment_quaternions = calibration.get(
        "axis_alignment_quaternions_wxyz"
    )
    drift_rates = calibration.get("drift_rates_rad_s")
    if not isinstance(axis_maps, dict) or not isinstance(drift_rates, dict):
        raise ValueError("в профиле отсутствуют оси или оценка дрейфа")
    if alignment_quaternions is not None and not isinstance(
        alignment_quaternions, dict
    ):
        raise ValueError("некорректное согласование осей в профиле")
    for name in SEGMENT_NAMES:
        axis_map_matrix(axis_maps[name])
        if alignment_quaternions is not None:
            normalize_quaternion(alignment_quaternions[name])
        values = np.asarray(drift_rates[name], dtype=float)
        if values.shape != (3,) or not np.all(np.isfinite(values)):
            raise ValueError(f"некорректная оценка дрейфа для {name}")
    return document


def restore_profile_calibration(model, document, *, reference_s=0.):
    """Restore the saved reference rather than recapturing the current live pose.

    Old profiles without a saved neutral need one explicit fresh reference.
    Stored host drift estimates are retained for diagnosis, not extrapolated.
    """
    calibration = document["calibration"]
    neutral = calibration.get("neutral_raw_quaternions_wxyz")
    if neutral is None:
        neutral = calibration.get("poses", {}).get("return_a_pose", {}).get("average_quaternions_wxyz")
    if neutral is None:
        model.set_axis_alignment(calibration.get("axis_alignment_quaternions_wxyz") or
                                 {s: (1., 0., 0., 0.) for s in SEGMENT_NAMES})
        model.set_drift_compensation(calibration["drift_rates_rad_s"])
        model.drift_compensation_enabled = False
        model.request_neutral()
        return False
    # Validate all before mutating state.
    neutral = {s: normalize_quaternion(neutral[s]) for s in SEGMENT_NAMES}
    model.set_guided_calibration(
        calibration["axis_maps"], calibration["drift_rates_rad_s"], neutral, reference_s,
        calibration.get("axis_alignment_quaternions_wxyz"), apply_drift_compensation=False)
    return True
