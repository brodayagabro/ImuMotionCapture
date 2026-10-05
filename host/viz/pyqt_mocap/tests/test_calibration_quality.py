"""Failures that previously produced a plausible but unreliable profile."""
import math

import numpy as np
import pytest

from pyqt_mocap.calibration import (
    FIT_POSE_NAMES, NEUTRAL_DIRECTIONS, PoseRecorder, TARGET_DIRECTIONS,
    calibrate_five_poses, profile_document,
)
from pyqt_mocap.mocap_core import (
    DEFAULT_SENSOR_MAPPING, MotionCaptureModel, SEGMENT_NAMES, quaternion_to_matrix,
)
from pyqt_mocap.tests import test_calibration as fixtures
from pyqt_mocap.tests.test_calibration import raw_quaternion_for, rotation_x


def captures():
    fixture = fixtures.GuidedCalibrationTests()
    fixture.setUp()
    return fixture.captures


def recorded_pose(motion):
    recorder = PoseRecorder()
    for index in range(101):
        time_s = index * .05
        snapshot = {}
        for segment in SEGMENT_NAMES:
            angle = motion(index, time_s) if segment == 'forearm.L' else 0.
            q = raw_quaternion_for(rotation_x(angle), segment)
            snapshot[segment] = (q * (-1 if index % 2 else 1), 140. + time_s, index)
        recorder.add_snapshot(snapshot)
    return recorder


def test_movement_during_capture_is_rejected_even_with_enough_samples():
    recorder = recorded_pose(lambda i, t: math.radians(7.) * t)
    with pytest.raises(ValueError, match='Поза нестабильна.*forearm.L'):
        recorder.finish('return_a_pose')


def test_stationary_noise_sign_flips_and_one_outlier_are_tolerated():
    recorder = recorded_pose(lambda i, t: math.radians(90 if i == 50 else .3 * math.sin(i)))
    pose = recorder.finish('return_a_pose')
    assert pose.spread_deg['forearm.L'] < 2.
    assert pose.sample_counts['forearm.L'] == 101


def test_drift_uses_time_between_averaged_edges_not_whole_window():
    pose = recorded_pose(lambda i, t: math.radians(.5) * t).finish('return_a_pose')
    data = captures()
    data['return_a_pose'] = pose
    result = calibrate_five_poses(data)
    np.testing.assert_allclose(result.drift_rates_rad_s['forearm.L'], [math.radians(.5), 0, 0], atol=1e-9)
    saved = profile_document({}, result)['calibration']['poses']['return_a_pose']
    assert 3.9 < saved['drift_interval_s']['forearm.L'] < 4.2
    assert saved['first_quaternions_wxyz'] != saved['last_quaternions_wxyz']
    assert saved['spread_deg']['forearm.L'] < 2.


def test_reported_pose_errors_match_model_using_final_a():
    data = captures()
    segment = 'forearm.L'
    before = calibrate_five_poses(data)
    final = data['return_a_pose']
    final.average[segment] = raw_quaternion_for(rotation_x(.25), segment)
    result = calibrate_five_poses(data)
    assert result.axis_maps == before.axis_maps
    for name in SEGMENT_NAMES:
        np.testing.assert_allclose(result.axis_alignment_quaternions[name], before.axis_alignment_quaternions[name], atol=1e-8)
    assert result.closure_error_deg[segment] == pytest.approx(math.degrees(.25))
    model = MotionCaptureModel(DEFAULT_SENSOR_MAPPING, result.axis_maps, 'raw', smooth_alpha=1.)
    model.set_guided_calibration(result.axis_maps, result.drift_rates_rad_s,
        final.average, result.reference_s, result.axis_alignment_quaternions)
    errors = []
    for index, pose in enumerate(FIT_POSE_NAMES):
        packet = [f'FRAME {index} 0 5']
        packet.extend(f'Q {sid} ' + ' '.join(map(str, data[pose].average[name]))
                      for name, sid in DEFAULT_SENSOR_MAPPING.items())
        model.handle_datagram('\n'.join(packet), result.reference_s)
        direction = quaternion_to_matrix(model.orientations()[segment]) @ NEUTRAL_DIRECTIONS[segment]
        error = math.degrees(math.acos(np.clip(direction @ TARGET_DIRECTIONS[pose][segment], -1, 1)))
        errors.append(error)
        assert result.pose_errors_deg[segment][pose] == pytest.approx(error)
    assert result.scores_deg[segment] == pytest.approx(np.sqrt(np.mean(np.square(errors))))


def test_bad_return_to_a_does_not_produce_a_new_profile():
    data = captures()
    data['return_a_pose'].average['forearm.L'] = raw_quaternion_for(rotation_x(.8), 'forearm.L')
    with pytest.raises(ValueError, match='возврат в A-позу'):
        calibrate_five_poses(data)


def test_insufficient_or_collinear_motion_is_rejected_but_disabled_sensor_is_skipped():
    data = captures()
    segment = 'forearm.L'
    data['forward_pose'].average[segment] = data['t_pose'].average[segment].copy()
    with pytest.raises(ValueError, match='разные направления'):
        calibrate_five_poses(data)
    for pose in data.values():
        pose.average[segment] = data['a_pose'].average[segment].copy()
    with pytest.raises(ValueError, match='недостаточно движения'):
        calibrate_five_poses(data)
    result = calibrate_five_poses(data, enabled_segments=set(SEGMENT_NAMES)-{segment})
    assert result.scores_deg[segment] == 0.
    assert result.drift_rates_rad_s[segment] == (0., 0., 0.)
