"""Planar arm calibration; preserve the unobservable twist from the prior."""

from datetime import datetime, timezone
import math

import numpy as np

from .calibration import (
    CalibrationResult, NEUTRAL_DIRECTIONS, quaternion_to_rotation_vector,
)
from .mocap_core import (
    DEFAULT_AXIS_MAPS, SEGMENT_NAMES, mapped_sensor_quaternion,
    matrix_to_quaternion, quaternion_from_rotation_vector,
    quaternion_inverse, quaternion_multiply, quaternion_to_matrix,
)

LETTERS = ("sem_a", "sem_u", "sem_zh", "sem_e", "sem_d")
SEMAPHORE_POSES = ("a_pose",) + LETTERS + ("return_a_pose",)
LABELS = dict(zip(LETTERS, ("А", "У", "Ж", "Е", "Д")))
INSTRUCTIONS = {
    "sem_a": "Обе прямые руки в стороны и вниз под 45° к корпусу.",
    "sem_u": "Обе прямые руки в стороны и вверх под 45° к горизонту.",
    "sem_zh": "Левая рука горизонтально влево; правая вверх-вправо под 45°. Локти прямые.",
    "sem_e": "Левая рука вниз; правая вверх-вправо под 45°. Локти прямые.",
    "sem_d": "Левая рука горизонтально влево. Правая вниз-влево через корпус под 45°, локоть слегка согнут. Правое предплечье в этой букве не оценивается.",
}
# Signed rotations about +Y, from arms down, in the preview's XZ plane.
ANGLES = {
    "sem_a": (45., -45.), "sem_u": (135., -135.),
    "sem_zh": (90., -135.), "sem_e": (0., -135.),
    "sem_d": (90., 45.),
}


def base_pose(name):
    return name.rsplit("_", 1)[0] if name.startswith("sem_") and name.endswith(("_1", "_2")) else name


def target_direction(pose, segment):
    base = base_pose(pose)
    if segment == "spine" or base not in ANGLES:
        return NEUTRAL_DIRECTIONS[segment].copy()
    if base == "sem_d" and segment == "forearm.R":
        return None  # Slight flexion is not a measured target angle.
    angle = math.radians(ANGLES[base][0 if segment.endswith(".L") else 1])
    return np.array((-math.sin(angle), 0., -math.cos(angle)))


def _shortest_alignment(source, target):
    """Minimal proper rotation; leaves the unconstrained twist unchanged."""
    cosine = float(np.clip(source @ target, -1., 1.))
    cross = np.cross(source, target)
    sine = float(np.linalg.norm(cross))
    if sine < 1e-10:
        if cosine > 0:
            return np.eye(3)
        axis = np.cross(source, np.eye(3)[np.argmin(np.abs(source))])
        axis /= np.linalg.norm(axis)
        return quaternion_to_matrix(quaternion_from_rotation_vector(axis * math.pi))
    return quaternion_to_matrix(quaternion_from_rotation_vector(
        cross / sine * math.atan2(sine, cosine)))


def calibrate_semaphore(captures, preferred_axis_maps=DEFAULT_AXIS_MAPS,
                        prior_alignment=None, enabled_segments=None):
    """Fit segment directions in one round, without inventing missing twist.

    Axis permutations remain unchanged. Initial A is the reference; final A
    only checks closure and measures stationary drift. D's right forearm is
    excluded because its flexion angle is unspecified.
    """
    if set(captures) != set(SEMAPHORE_POSES):
        raise ValueError("Нужны начальная A-поза, А/У/Ж/Е/Д и конечная A-поза")
    enabled = set(SEGMENT_NAMES if enabled_segments is None else enabled_segments)
    prior_alignment = prior_alignment or {}
    maps = {s: tuple(preferred_axis_maps[s]) for s in SEGMENT_NAMES}
    alignments, scores, preferred_scores, rates, closure = {}, {}, {}, {}, {}
    final = captures["return_a_pose"]
    duration = final.ended_s - final.started_s
    if duration < 1.:
        raise ValueError("Конечную A-позу необходимо удерживать не менее секунды")
    for segment in SEGMENT_NAMES:
        prior = quaternion_to_matrix(prior_alignment.get(segment, (1., 0., 0., 0.)))
        neutral = mapped_sensor_quaternion(captures["a_pose"].average[segment], maps[segment])
        deltas = {
            name: quaternion_multiply(mapped_sensor_quaternion(pose.average[segment], maps[segment]),
                                      quaternion_inverse(neutral))
            for name, pose in captures.items()
        }
        alignment = prior.copy()
        vectors = []
        if segment in enabled and segment != "spine":
            for name in SEMAPHORE_POSES[1:-1]:
                if target_direction(name, segment) is None:
                    continue
                angle = ANGLES[base_pose(name)][0 if segment.endswith(".L") else 1]
                vector = quaternion_to_rotation_vector(deltas[name])
                if abs(angle) >= 25 and np.linalg.norm(vector) >= math.radians(25):
                    vectors.append(vector * np.sign(angle))
            if len(vectors) < 3:
                raise ValueError(f"{segment}: недостаточно движений; проверьте датчик и повторите жесты")
            mean = np.mean(vectors, axis=0)
            if np.linalg.norm(mean) < math.radians(15):
                raise ValueError(f"{segment}: направления жестов противоречат друг другу")
            source = prior @ (mean / np.linalg.norm(mean))
            alignment = _shortest_alignment(source, np.array((0., 1., 0.))) @ prior

        def direction(name, matrix):
            return matrix @ quaternion_to_matrix(deltas[name]) @ matrix.T @ NEUTRAL_DIRECTIONS[segment]

        training_names = [name for name in LETTERS
                          if target_direction(name, segment) is not None]

        def residual(matrix):
            return np.concatenate([direction(name, matrix) - target_direction(name, segment)
                                   for name in training_names])

        if segment in enabled and segment != "spine":
            # Refine the normal (two degrees of freedom), not a full 3D basis.
            # Raw rotation axes alone are biased by natural forearm twist.
            normal = alignment.T @ np.array((0., 1., 0.))

            def from_normal(value):
                value = value / np.linalg.norm(value)
                return _shortest_alignment(prior @ value, np.array((0., 1., 0.))) @ prior

            for _ in range(30):
                current = residual(alignment)
                tangent = np.cross(normal, np.eye(3)[np.argmin(np.abs(normal))])
                tangent /= np.linalg.norm(tangent)
                basis = np.column_stack((tangent, np.cross(normal, tangent)))
                epsilon = 1e-5
                jacobian = np.column_stack([
                    (residual(from_normal(normal + epsilon * axis)) - current) / epsilon
                    for axis in basis.T
                ])
                step = np.linalg.solve(jacobian.T @ jacobian + 1e-4 * np.eye(2),
                                       -jacobian.T @ current)
                length = float(np.linalg.norm(step))
                if length < 1e-8:
                    break
                step *= min(1., .15 / length)
                for factor in (1., .5, .25, .125):
                    candidate_normal = normal + factor * basis @ step
                    candidate_normal /= np.linalg.norm(candidate_normal)
                    candidate = from_normal(candidate_normal)
                    if np.linalg.norm(residual(candidate)) < np.linalg.norm(current):
                        normal, alignment = candidate_normal, candidate
                        break
                else:
                    break

        def error(matrix, use_final=False):
            errors = []
            neutral_rotation = quaternion_to_matrix(deltas["return_a_pose"]).T if use_final else np.eye(3)
            for name in training_names:
                target = target_direction(name, segment)
                if target is not None:
                    predicted = (matrix @ quaternion_to_matrix(deltas[name]) @ neutral_rotation
                                 @ matrix.T @ NEUTRAL_DIRECTIONS[segment])
                    dot = float(predicted @ target)
                    errors.append(math.degrees(math.acos(np.clip(dot, -1., 1.))))
            return float(np.sqrt(np.mean(np.square(errors))))

        before, after = error(prior), error(alignment)
        if after > before:
            alignment, after = prior, before
        # Report the error for the final neutral that the application WILL use.
        scores[segment] = error(alignment, use_final=True) if segment in enabled else 0.
        preferred_scores[segment] = error(prior, use_final=True) if segment in enabled else 0.
        closure[segment] = math.degrees(math.acos(np.clip(
            direction("return_a_pose", alignment) @ NEUTRAL_DIRECTIONS[segment], -1., 1.))) if segment in enabled else 0.
        if closure[segment] > 30.:
            raise ValueError(f"{segment}: возврат в A-позу отличается на {closure[segment]:.1f}°. Повторите калибровку, не поворачивая корпус.")
        alignments[segment] = tuple(float(v) for v in matrix_to_quaternion(alignment))
        first = mapped_sensor_quaternion(final.first[segment], maps[segment])
        last = mapped_sensor_quaternion(final.last[segment], maps[segment])
        rate = alignment @ quaternion_to_rotation_vector(
            quaternion_multiply(last, quaternion_inverse(first))) / duration
        rates[segment] = tuple(float(v) for v in rate) if segment in enabled else (0., 0., 0.)
    return CalibrationResult(
        axis_maps=maps, axis_alignment_quaternions=alignments, drift_rates_rad_s=rates,
        scores_deg=scores, preferred_scores_deg=preferred_scores, captures=dict(captures),
        reference_s=(final.started_s + final.ended_s) / 2,
        created_at=datetime.now(timezone.utc).isoformat(),
        method="semaphore_xz", closure_error_deg=closure,
    )
