"""Five real poses, one shared letter dictionary, and a fixed sensor-Y fit."""
from dataclasses import replace
import math
import numpy as np

from pyqt_mocap.calibration import NEUTRAL_DIRECTIONS
from pyqt_mocap.mocap_core import DEFAULT_AXIS_MAPS, SEGMENT_NAMES, axis_map_matrix, matrix_to_quaternion
from pyqt_mocap.orientation_frames import SegmentCalibration
from pyqt_mocap.quaternion_utils import angular_distance, quaternion_to_matrix, rotate_vector
from pyqt_mocap.mocap_core import mapped_sensor_quaternion
from pyqt_mocap.t_pose_calibration import ARM_DISTAL_SENSOR_AXES, calibrate_t_pose
from .semaphore_model import SEMAPHORE_POSES, NEUTRAL

POSE_SEQUENCE = ('neutral_a_pose', 'semaphore_a', 'semaphore_t', 'semaphore_u', 'neutral_a_pose_final')
LETTERS = {'semaphore_a': 'А', 'semaphore_t': 'Т', 'semaphore_u': 'У'}
POSE_TITLES = {'neutral_a_pose': 'A-поза — руки вниз', 'semaphore_a': 'Буква А',
               'semaphore_t': 'Буква Т', 'semaphore_u': 'Буква У', 'neutral_a_pose_final': 'A-поза — снова руки вниз'}


def displayed_pose(name):
    return SEMAPHORE_POSES[LETTERS[name]] if name in LETTERS else NEUTRAL


def target_direction(name, segment):
    if segment == 'spine':
        return NEUTRAL_DIRECTIONS[segment]
    pose = displayed_pose(name)
    angle = math.radians(pose.left_angle_deg if segment.endswith('.L') else pose.right_angle_deg)
    return np.array((math.cos(angle), 0., math.sin(angle)))


def calibrate_series(captures, preferred_axis_maps=DEFAULT_AXIS_MAPS, prior_alignment=None, enabled_segments=None):
    if tuple(captures) != POSE_SEQUENCE:
        raise ValueError('Нужны пять положений: A-поза → буква А → буква Т → буква У → A-поза')
    # Reuse closure, degeneracy checks, diagnostic rate and neutral model.
    aliases = {'a_pose': captures[POSE_SEQUENCE[0]], 't_pose': captures['semaphore_t'],
               'return_a_pose': captures[POSE_SEQUENCE[-1]]}
    # Duration and minimum measurement count are validated by FramePoseRecorder.
    # Keep the fitter compatible with earlier frame-count-based profiles.
    base = calibrate_t_pose(aliases, preferred_axis_maps, prior_alignment, SEGMENT_NAMES, minimum_duration_s=0.)
    alignments = dict(base.axis_alignment_quaternions)
    errors, scores = {}, {}
    for segment, local_y in ARM_DISTAL_SENSOR_AXES.items():
        basis = axis_map_matrix(preferred_axis_maps[segment])
        a = basis @ rotate_vector(captures[POSE_SEQUENCE[0]].average[segment], local_y)
        side = np.zeros(3)
        target_a = NEUTRAL_DIRECTIONS[segment]
        target_t = target_direction('semaphore_t', segment)
        for name in LETTERS:
            observed = basis @ rotate_vector(captures[name].average[segment], local_y)
            weight = target_direction(name, segment) @ target_t
            side += weight * (observed - (observed @ a) * a)
        if np.linalg.norm(side) < 1e-8:
            raise ValueError(f'{segment}: противоречивые направления букв А, Т, У')
        side /= np.linalg.norm(side)
        source = np.column_stack((a, side, np.cross(a, side)))
        target = np.column_stack((target_a, target_t, np.cross(target_a, target_t)))
        alignments[segment] = tuple(matrix_to_quaternion(target @ source.T))
    for segment in SEGMENT_NAMES:
        frame = SegmentCalibration.from_pose(
            mapped_sensor_quaternion(captures[POSE_SEQUENCE[-1]].average[segment], preferred_axis_maps[segment]),
            q_world_alignment=alignments[segment])
        errors[segment] = {}
        for name in LETTERS:
            q = frame.apply(mapped_sensor_quaternion(captures[name].average[segment], preferred_axis_maps[segment]))
            dot = rotate_vector(q, NEUTRAL_DIRECTIONS[segment]) @ target_direction(name, segment)
            errors[segment][name] = math.degrees(math.acos(float(np.clip(dot, -1, 1))))
        scores[segment] = max(errors[segment].values())
    rates = {s: tuple(quaternion_to_matrix(alignments[s]) @
                     quaternion_to_matrix(base.axis_alignment_quaternions[s]).T @
                     np.asarray(base.drift_rates_rad_s[s])) for s in SEGMENT_NAMES}
    return replace(base, captures=dict(captures), method='neutral_a_semaphore_a_t_u_neutral_a_sensor_y',
                   neutral_pose_name=POSE_SEQUENCE[-1], axis_alignment_quaternions=alignments,
                   drift_rates_rad_s=rates,
                   scores_deg=scores, pose_errors_deg=errors,
                   repeatability_deg={s: math.degrees(angular_distance(captures[POSE_SEQUENCE[0]].average[s],
                       captures[POSE_SEQUENCE[-1]].average[s])) for s in SEGMENT_NAMES})
