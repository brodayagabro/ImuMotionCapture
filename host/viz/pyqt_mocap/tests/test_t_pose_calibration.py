"""Physical sensor mounting, arm swing/twist, closure and three-pose profiles."""
import math

import numpy as np
import pytest

from pyqt_mocap.calibration import (
    CapturedPose, NEUTRAL_DIRECTIONS, PoseRecorder, TARGET_DIRECTIONS,
    profile_document, save_profile, load_profile,
)
from pyqt_mocap.mocap_core import (
    DEFAULT_AXIS_MAPS, DEFAULT_SENSOR_MAPPING, MotionCaptureModel, SEGMENT_NAMES,
    SENSOR_NEUTRAL_AXIS_FRAMES, axis_map_matrix, matrix_to_quaternion, quaternion_to_matrix,
)
from pyqt_mocap.t_pose_calibration import (
    T_POSE_NAMES, ReturnPoseMismatch, arm_return_errors_deg, calibrate_t_pose,
)
from pyqt_mocap.tests.test_calibration import rotation_x, rotation_y, rotation_z


ARMS = tuple(s for s in SEGMENT_NAMES if s != 'spine')


def physical_captures(world_frames=None, t_twists=None):
    """Absolute IMU rotations with local +Y distal on left, proximal on right."""
    world_frames = world_frames or {}
    t_twists = t_twists or {}
    captures = {}
    for index, name in enumerate(T_POSE_NAMES):
        values = {}
        for segment in SEGMENT_NAMES:
            body = np.eye(3)
            if name == 't_pose' and segment != 'spine':
                body = rotation_y(math.pi / 2 * (1 if segment.endswith('.L') else -1))
            rotation = world_frames.get(segment, np.eye(3)) @ body @ SENSOR_NEUTRAL_AXIS_FRAMES[segment]
            if name == 't_pose':
                rotation = rotation @ rotation_y(t_twists.get(segment, 0.))
            values[segment] = matrix_to_quaternion(rotation)
        captures[name] = CapturedPose(
            name, {s: q.copy() for s, q in values.items()},
            {s: q.copy() for s, q in values.items()}, {s: q.copy() for s, q in values.items()},
            {s: 50 for s in SEGMENT_NAMES}, 100. + 10 * index, 105. + 10 * index)
    return captures


@pytest.fixture
def captures():
    return physical_captures()


def calibrated_model(result):
    model = MotionCaptureModel(DEFAULT_SENSOR_MAPPING, result.axis_maps, 'raw', smooth_alpha=1.)
    model.set_enabled_segments(SEGMENT_NAMES)
    model.set_guided_calibration(result.axis_maps, result.drift_rates_rad_s,
        result.captures['return_a_pose'].average, result.reference_s, result.axis_alignment_quaternions)
    return model


def directions_after_packet(model, raw_values, timestamp):
    packet = ['FRAME 1 0 5']
    packet.extend(f'Q {sid} ' + ' '.join(map(str, raw_values[name]))
                  for name, sid in DEFAULT_SENSOR_MAPPING.items())
    model.handle_datagram('\n'.join(packet), timestamp)
    return {s: quaternion_to_matrix(q) @ NEUTRAL_DIRECTIONS[s]
            for s, q in model.orientations().items()}


@pytest.mark.parametrize('axis_maps', [DEFAULT_AXIS_MAPS, {s: ('+Y', '+Z', '+X') for s in SEGMENT_NAMES}])
def test_sensor_y_corrects_world_frames_and_forward_elbow_flexion(axis_maps, tmp_path):
    # Independent IMU headings, including a large rotation the old one-axis fit
    # could leave undetermined. Twisting a wrist during T must not change the fit.
    worlds = {s: rotation_z(.3 + i * .7) @ rotation_y(1.2 - i * .4) @ rotation_x(.2)
              for i, s in enumerate(ARMS)}
    captures = physical_captures(worlds, {s: .6 for s in ARMS})
    result = calibrate_t_pose(captures, axis_maps)
    assert result.method == 'a_t_a_sensor_y'
    assert result.axis_maps == axis_maps
    assert max(result.scores_deg.values()) < 2e-6
    assert max(result.closure_error_deg.values()) < 2e-6
    for segment in ARMS:
        alignment = quaternion_to_matrix(result.axis_alignment_quaternions[segment])
        np.testing.assert_allclose(alignment @ axis_map_matrix(axis_maps[segment]) @ worlds[segment],
                                   np.eye(3), atol=1e-8)
        assert np.linalg.det(alignment) == pytest.approx(1.)
    model = calibrated_model(result)
    directions = directions_after_packet(model, captures['t_pose'].average, result.reference_s)
    for segment in ARMS:
        np.testing.assert_allclose(directions[segment], TARGET_DIRECTIONS['t_pose'][segment], atol=1e-8)
    for flexion in (math.pi / 4, math.pi / 2):
        values = dict(captures['a_pose'].average)
        for segment in ('forearm.L', 'forearm.R'):
            values[segment] = matrix_to_quaternion(worlds[segment] @ rotation_x(flexion)
                                                  @ SENSOR_NEUTRAL_AXIS_FRAMES[segment])
        directions = directions_after_packet(model, values, result.reference_s)
        for segment in ('forearm.L', 'forearm.R'):
            np.testing.assert_allclose(directions[segment], [0., math.sin(flexion), -math.cos(flexion)], atol=1e-8)
        for segment in ('shoulder.L', 'shoulder.R'):
            np.testing.assert_allclose(directions[segment], [0., 0., -1.], atol=1e-8)
    path = tmp_path / 'profile.json'
    save_profile(path, profile_document({}, result))
    saved = load_profile(path)['calibration']
    assert saved['method'] == 'a_t_a_sensor_y'
    assert tuple(saved['poses']) == T_POSE_NAMES


def test_bad_previous_arm_alignment_is_replaced_and_torso_prior_preserved(captures):
    prior = {s: matrix_to_quaternion(rotation_y(1.1)) for s in SEGMENT_NAMES}
    result = calibrate_t_pose(captures, prior_alignment=prior, enabled_segments=set(SEGMENT_NAMES) - {'shoulder.R'})
    for segment in ('shoulder.L', 'forearm.L', 'forearm.R'):
        alignment = quaternion_to_matrix(result.axis_alignment_quaternions[segment])
        np.testing.assert_allclose(alignment @ axis_map_matrix(DEFAULT_AXIS_MAPS[segment]), np.eye(3), atol=1e-8)
    for segment in ('spine', 'shoulder.R'):
        np.testing.assert_allclose(quaternion_to_matrix(result.axis_alignment_quaternions[segment]),
                                   quaternion_to_matrix(prior[segment]), atol=1e-8)


@pytest.mark.parametrize('twist', [-math.pi / 2, math.pi / 2])
def test_local_y_twist_does_not_swing_either_arm(captures, twist):
    result = calibrate_t_pose(captures)
    model = calibrated_model(result)
    for pose_name in ('a_pose', 't_pose'):
        values = dict(captures[pose_name].average)
        for segment in ARMS:
            values[segment] = matrix_to_quaternion(quaternion_to_matrix(values[segment]) @ rotation_y(twist))
        directions = directions_after_packet(model, values, result.reference_s)
        for segment in ARMS:
            expected = NEUTRAL_DIRECTIONS[segment] if pose_name == 'a_pose' else TARGET_DIRECTIONS['t_pose'][segment]
            np.testing.assert_allclose(directions[segment], expected, atol=1e-8)


@pytest.mark.parametrize('twist', [0., math.pi / 2])
def test_missing_arm_lift_is_rejected_even_if_sensor_twists(captures, twist):
    segment = 'forearm.L'
    captures['t_pose'].average[segment] = matrix_to_quaternion(
        quaternion_to_matrix(captures['a_pose'].average[segment]) @ rotation_y(twist))
    with pytest.raises(ValueError, match='forearm.L: недостаточно движения'):
        calibrate_t_pose(captures)
    result = calibrate_t_pose(captures, enabled_segments=set(SEGMENT_NAMES) - {segment})
    assert result.scores_deg[segment] == 0.
    assert result.pose_errors_deg[segment] == {}
    assert result.drift_rates_rad_s[segment] == (0., 0., 0.)


def test_final_a_checks_closure_and_does_not_change_fitted_axes(captures):
    before = calibrate_t_pose(captures)
    segment = 'forearm.L'
    captures['return_a_pose'].average[segment] = matrix_to_quaternion(rotation_y(.2) @ SENSOR_NEUTRAL_AXIS_FRAMES[segment])
    after = calibrate_t_pose(captures)
    assert after.axis_alignment_quaternions == before.axis_alignment_quaternions
    assert after.closure_error_deg[segment] == pytest.approx(math.degrees(.2))
    assert after.scores_deg[segment] == pytest.approx(math.degrees(.2))
    captures['return_a_pose'].average[segment] = matrix_to_quaternion(rotation_y(.8) @ SENSOR_NEUTRAL_AXIS_FRAMES[segment])
    with pytest.raises(ValueError, match='возврат в A-позу'):
        calibrate_t_pose(captures)


def test_stationary_drift_uses_measured_edge_interval(captures):
    recorder = PoseRecorder()
    for index in range(101):
        time_s = index * .05
        values = {}
        for segment in SEGMENT_NAMES:
            motion = rotation_x(math.radians(.5) * time_s) if segment == 'forearm.L' else np.eye(3)
            values[segment] = (matrix_to_quaternion(motion @ SENSOR_NEUTRAL_AXIS_FRAMES[segment]), 140. + time_s, index)
        recorder.add_snapshot(values)
    captures['return_a_pose'] = recorder.finish('return_a_pose')
    result = calibrate_t_pose(captures)
    np.testing.assert_allclose(result.drift_rates_rad_s['forearm.L'], [math.radians(.5), 0., 0.], atol=1e-9)
    captures['return_a_pose'].drift_interval_s['forearm.L'] = 0.
    with pytest.raises(ValueError, match='интервал оценки дрейфа'):
        calibrate_t_pose(captures)


def test_missing_stage_is_rejected(captures):
    del captures['return_a_pose']
    with pytest.raises(ValueError, match='начальная A-поза, T-поза и конечная A-поза'):
        calibrate_t_pose(captures)


@pytest.mark.parametrize('sign', [-1., 1.])
def test_return_check_measures_raw_y_swing_not_twist_or_world_heading(sign):
    worlds = {s: rotation_z(.7 * i) @ rotation_x(.3) for i, s in enumerate(ARMS)}
    captures = physical_captures(worlds)
    for segment in ARMS:
        # Any local Y twist preserves arm direction; only right forearm swings.
        swing = rotation_x(math.radians(40.3)) if segment == 'forearm.R' else np.eye(3)
        captures['return_a_pose'].average[segment] = sign * matrix_to_quaternion(
            worlds[segment] @ swing @ SENSOR_NEUTRAL_AXIS_FRAMES[segment] @ rotation_y(1.4))
    errors = arm_return_errors_deg(captures)
    assert errors['forearm.R'] == pytest.approx(40.3)
    assert max(errors[s] for s in ARMS if s != 'forearm.R') < 1e-10
    with pytest.raises(ReturnPoseMismatch) as caught:
        calibrate_t_pose(captures)
    assert caught.value.errors_deg['forearm.R'] == pytest.approx(40.3)
    assert len(caught.value.errors_deg) == len(SEGMENT_NAMES)
    # Cross-check against the original rendered closure definition, independently
    # of the direct-Y path used for reporting/rejection.
    from pyqt_mocap.calibration import _pose_direction_errors_deg
    from pyqt_mocap.t_pose_calibration import _arm_alignment
    for segment in ARMS:
        alignment = _arm_alignment(segment, captures, DEFAULT_AXIS_MAPS[segment])
        rendered = _pose_direction_errors_deg(segment, DEFAULT_AXIS_MAPS[segment], captures,
                                             alignment, pose_names=('return_a_pose',))
        assert errors[segment] == pytest.approx(rendered['return_a_pose'], abs=2e-6)
