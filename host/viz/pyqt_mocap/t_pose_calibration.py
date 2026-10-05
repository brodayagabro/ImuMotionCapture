"""A -> T -> A calibration using the sensor's longitudinal Y axis."""

from datetime import datetime, timezone
import math

import numpy as np

from .calibration import (
    CalibrationResult, MAX_CLOSURE_ERROR_DEG, NEUTRAL_DIRECTIONS, TARGET_DIRECTIONS,
    MIN_ALIGNMENT_AXIS_SEPARATION, MIN_ALIGNMENT_MOTION_RAD, _pose_direction_errors_deg,
    quaternion_to_rotation_vector,
)
from .mocap_core import (
    DEFAULT_AXIS_MAPS, SEGMENT_NAMES, axis_map_matrix, mapped_sensor_quaternion,
    matrix_to_quaternion, quaternion_inverse, quaternion_multiply,
    quaternion_to_matrix,
)


T_POSE_NAMES = ("a_pose", "t_pose", "return_a_pose")
# Physical local sensor direction from the proximal to the distal joint.
# Left +Y points towards the wrist; right +Y points towards the shoulder.
ARM_DISTAL_SENSOR_AXES = {
    "shoulder.L": np.array((0., 1., 0.)),
    "forearm.L": np.array((0., 1., 0.)),
    "shoulder.R": np.array((0., -1., 0.)),
    "forearm.R": np.array((0., -1., 0.)),
}


class ReturnPoseMismatch(ValueError):
    """A recoverable mismatch; keep all measurements for diagnosis and retry."""

    def __init__(self, errors_deg):
        self.errors_deg = dict(errors_deg)
        failed = [f"{s}: {error:.1f}°" for s, error in errors_deg.items()
                  if error > MAX_CLOSURE_ERROR_DEG]
        super().__init__(
            "Не принят возврат в A-позу: " + "; ".join(failed)
            + f" (предел {MAX_CLOSURE_ERROR_DEG:g}°). "
            "Это расхождение направлений по данным датчиков; его причина по одному углу неизвестна. "
            "Опустите и выпрямите руки, верните ладони к бёдрам и повторите последнюю A-позу. "
            "Если расхождение сохраняется при той же позе, начните калибровку заново."
        )


def arm_return_errors_deg(captures):
    """Compare raw physical Y directions, independent of fitted world frames.

    A local-Y twist or a quaternion sign change must never look like arm swing.
    For this A/T fit this is algebraically equal to the rendered closure check.
    """
    errors = {}
    for segment, axis in ARM_DISTAL_SENSOR_AXES.items():
        initial, final = (quaternion_to_matrix(captures[p].average[segment]) @ axis
                          for p in ("a_pose", "return_a_pose"))
        errors[segment] = math.degrees(math.atan2(float(np.linalg.norm(np.cross(initial, final))),
                                                 float(np.clip(initial @ final, -1., 1.))))
    return errors


def _arm_alignment(segment, captures, axis_spec):
    """Map measured arm directions in A/T into the anatomical reference frame.

    The quaternion maps sensor-local axes into its world frame. Axis mapping
    changes that world frame, so the physical Y vector is B @ R(raw) @ local_Y,
    not the Y column of B @ R(raw) @ B.T. Keeping A exact also makes rotation
    about the arm's local Y a twist instead of a spurious swing of the arm.
    """
    basis = axis_map_matrix(axis_spec)
    local_distal = ARM_DISTAL_SENSOR_AXES[segment]
    source_a, source_t = (
        basis @ quaternion_to_matrix(captures[name].average[segment]) @ local_distal
        for name in ("a_pose", "t_pose")
    )
    cosine = float(np.clip(source_a @ source_t, -1., 1.))
    if math.acos(cosine) < MIN_ALIGNMENT_MOTION_RAD:
        raise ValueError(f"{segment}: недостаточно движения в T-позе; повторите калибровку")
    source_side = source_t - cosine * source_a
    length = float(np.linalg.norm(source_side))
    if length < MIN_ALIGNMENT_AXIS_SEPARATION:
        raise ValueError(f"{segment}: A- и T-позы не задают плоскость подъёма; "
                         "в T-позе разведите прямые руки горизонтально в стороны")
    source_side /= length
    source_frame = np.column_stack((source_a, source_side, np.cross(source_a, source_side)))
    target_a = NEUTRAL_DIRECTIONS[segment]
    target_t = TARGET_DIRECTIONS["t_pose"][segment]
    target_frame = np.column_stack((target_a, target_t, np.cross(target_a, target_t)))
    return target_frame @ source_frame.T


def calibrate_t_pose(captures, preferred_axis_maps=DEFAULT_AXIS_MAPS,
                     prior_alignment=None, enabled_segments=None):
    """Align the sensor-local arm axis in A and T with both target directions.

    Known longitudinal Y mounting supplies the constraint missing from the
    rotation axis alone. Prior alignment is kept only for the torso and disabled
    sensors. Final A validates closure, measures drift and sets neutral.
    """
    if set(captures) != set(T_POSE_NAMES):
        raise ValueError("Нужны начальная A-поза, T-поза и конечная A-поза")
    enabled = set(SEGMENT_NAMES if enabled_segments is None else enabled_segments)
    if enabled.difference(SEGMENT_NAMES):
        raise ValueError("Unknown enabled calibration segment")
    prior_alignment = prior_alignment or {}
    maps = {s: tuple(preferred_axis_maps[s]) for s in SEGMENT_NAMES}
    final = captures["return_a_pose"]
    duration = final.ended_s - final.started_s
    if not math.isfinite(duration) or duration < 1.:
        raise ValueError("Конечную A-позу необходимо удерживать не менее секунды")
    alignments, scores, preferred_scores, rates, closure, pose_errors = {}, {}, {}, {}, {}, {}
    raw_arm_closure = arm_return_errors_deg(captures)
    for segment in SEGMENT_NAMES:
        prior = quaternion_to_matrix(prior_alignment.get(segment, (1., 0., 0., 0.)))
        alignment = prior.copy()
        if segment in enabled and segment != "spine":
            alignment = _arm_alignment(segment, captures, maps[segment])
        alignments[segment] = tuple(float(v) for v in matrix_to_quaternion(alignment))
        if segment not in enabled:
            scores[segment] = preferred_scores[segment] = closure[segment] = 0.
            pose_errors[segment] = {}
            rates[segment] = (0., 0., 0.)
            continue
        closure[segment] = raw_arm_closure[segment] if segment in raw_arm_closure else _pose_direction_errors_deg(
            segment, maps[segment], captures, alignment,
            pose_names=("return_a_pose",))["return_a_pose"]
        pose_errors[segment] = _pose_direction_errors_deg(
            segment, maps[segment], captures, alignment,
            reference_pose="return_a_pose", pose_names=("t_pose",))
        scores[segment] = pose_errors[segment]["t_pose"]
        preferred_scores[segment] = _pose_direction_errors_deg(
            segment, maps[segment], captures, prior,
            reference_pose="return_a_pose", pose_names=("t_pose",))["t_pose"]
        first = mapped_sensor_quaternion(final.first[segment], maps[segment])
        last = mapped_sensor_quaternion(final.last[segment], maps[segment])
        interval = final.drift_interval_s.get(segment, duration)
        if interval <= 0. or not math.isfinite(interval):
            raise ValueError(f"{segment}: некорректный интервал оценки дрейфа")
        rate = alignment @ quaternion_to_rotation_vector(
            quaternion_multiply(last, quaternion_inverse(first))) / interval
        rates[segment] = tuple(float(v) for v in rate)
    if any(error > MAX_CLOSURE_ERROR_DEG for error in closure.values()):
        raise ReturnPoseMismatch(closure)
    return CalibrationResult(
        axis_maps=maps, axis_alignment_quaternions=alignments, drift_rates_rad_s=rates,
        scores_deg=scores, preferred_scores_deg=preferred_scores, captures=dict(captures),
        reference_s=(final.started_s + final.ended_s) / 2,
        created_at=datetime.now(timezone.utc).isoformat(),
        method="a_t_a_sensor_y", closure_error_deg=closure, pose_errors_deg=pose_errors,
    )
