from dataclasses import replace
import math
import numpy as np
import pytest

from pyqt_mocap.calibration import CapturedPose, profile_document, restore_profile_calibration
from scripts.ER2026.experiments.semaphore.frame_calibration import FramePoseRecorder
from pyqt_mocap.mocap_core import SEGMENT_NAMES, DEFAULT_SENSOR_MAPPING, DEFAULT_AXIS_MAPS, SENSOR_NEUTRAL_AXIS_FRAMES, MotionCaptureModel, matrix_to_quaternion
from pyqt_mocap.quaternion_utils import angular_distance, quaternion_from_rotation_vector, quaternion_to_matrix, rotate_vector
from pyqt_mocap.t_pose_calibration import ReturnPoseMismatch
from scripts.ER2026.experiments.semaphore.calibration_protocol import POSE_SEQUENCE, target_direction, calibrate_series
from scripts.ER2026.diagnostics.orientation.calibration_analyzer import analyze_profile


def physical_series_captures(worlds=None):
    worlds = worlds or {}
    result = {}
    for i, name in enumerate(POSE_SEQUENCE):
        values = {}
        for s in SEGMENT_NAMES:
            target = target_direction(name, s)
            angle = 0. if s == 'spine' else math.atan2(-target[0], -target[2])
            body = quaternion_to_matrix(quaternion_from_rotation_vector((0., angle, 0.)))
            values[s] = matrix_to_quaternion(worlds.get(s, np.eye(3)) @ body @ SENSOR_NEUTRAL_AXIS_FRAMES[s])
        result[name] = CapturedPose(name, values, values, values, {s: 100 for s in SEGMENT_NAMES},
                                   100. + i * 6, 101.98 + i * 6)
    return result


def test_five_poses_use_shared_targets_and_recover_independent_world_frames():
    worlds = {s: quaternion_to_matrix(quaternion_from_rotation_vector((.3*i, .5, .2)))
              for i, s in enumerate(SEGMENT_NAMES) if s != 'spine'}
    captures = physical_series_captures(worlds)
    result = calibrate_series(captures)
    assert tuple(result.captures) == POSE_SEQUENCE
    assert max(result.scores_deg.values()) < 2e-6
    model = MotionCaptureModel()
    doc = profile_document({}, result)
    assert doc['version'] == 8 and doc['calibration']['calibration_id']
    assert restore_profile_calibration(model, doc)
    for name in POSE_SEQUENCE:
        for s in SEGMENT_NAMES:
            from pyqt_mocap.mocap_core import mapped_sensor_quaternion
            q = model.segment_calibrations[s].apply(mapped_sensor_quaternion(captures[name].average[s], DEFAULT_AXIS_MAPS[s]))
            axis = [0., 0., 1. if s == 'spine' else -1.]
            np.testing.assert_allclose(rotate_vector(q, axis), target_direction(name, s), atol=1e-8)
    assert max(r['axis_direction_error_deg'] for r in analyze_profile(doc)['residuals']) < 2e-6


def test_right_forearm_return_mismatch_still_rejected_without_special_coefficients():
    captures = physical_series_captures()
    values = dict(captures[POSE_SEQUENCE[-1]].average)
    values['forearm.R'] = matrix_to_quaternion(quaternion_to_matrix(quaternion_from_rotation_vector((math.radians(40.3), 0, 0))) @ SENSOR_NEUTRAL_AXIS_FRAMES['forearm.R'])
    captures[POSE_SEQUENCE[-1]] = replace(captures[POSE_SEQUENCE[-1]], average=values)
    with pytest.raises(ReturnPoseMismatch) as error:
        calibrate_series(captures)
    assert error.value.errors_deg['forearm.R'] == pytest.approx(40.3)


def test_a_and_u_influence_plane_fit_and_preserve_neutral_direction():
    captures = physical_series_captures()
    original = calibrate_series(captures)
    yaw = quaternion_to_matrix(quaternion_from_rotation_vector((0., 0., .2)))
    for name in ('semaphore_a', 'semaphore_u'):
        values = dict(captures[name].average)
        values['forearm.R'] = matrix_to_quaternion(yaw @ quaternion_to_matrix(values['forearm.R']))
        captures[name] = replace(captures[name], average=values)
    updated = calibrate_series(captures)
    assert not np.allclose(original.axis_alignment_quaternions['forearm.R'], updated.axis_alignment_quaternions['forearm.R'])
    assert set(updated.pose_errors_deg['forearm.R']) == {'semaphore_a', 'semaphore_t', 'semaphore_u'}
    assert 0 < updated.scores_deg['forearm.R'] < 15


@pytest.mark.parametrize('rate', [20, 50, 100])
def test_1500_ms_of_complete_frames_counted_once_including_hidden_spine(rate):
    model = MotionCaptureModel(sensor_id_mode='tca_channel')
    values = physical_series_captures()[POSE_SEQUENCE[0]].average
    recorder = FramePoseRecorder(POSE_SEQUENCE[0], 0.)
    final = None
    total_frames = 15 + round(1.5 * rate)
    for i in range(1, total_frames + 1):
        text = f'FRAME {i} {round(i*1000/rate)} 5\n' + '\n'.join(f'Q {sid} ' + ' '.join(map(str, values[s])) for s, sid in DEFAULT_SENSOR_MAPPING.items())
        parsed = model.handle_datagram(text, i/rate)
        assert set(parsed.calibration_samples) == set(SEGMENT_NAMES)
        final = recorder.add_frame(parsed.calibration_samples, parsed.measurement_key, i/rate)
        recorder.add_frame(parsed.calibration_samples, parsed.measurement_key, i/rate)
        if i < total_frames:
            assert final is None
    assert final is not None and recorder.accepted == round(1.5 * rate) + 1
    assert set(final.sample_counts.values()) == {round(1.5 * rate) + 1}
    assert final.frame_diagnostics['required_duration_s'] == 1.5
    assert final.frame_diagnostics['accepted_frequency_hz'] == pytest.approx(rate)
    assert final.frame_diagnostics['capture_duration_s'] == pytest.approx(1.5)
    assert final.frame_diagnostics['rejection_reasons']['duplicate_or_out_of_order'] == total_frames - 1
    for bad in (text.replace('Q 2 ', 'Q 200 '), text.replace('Q 2 ', 'Q 1 '), text.replace('Q 2 ', 'Q 2 nan ')):
        assert not model.handle_datagram(bad, 21.).calibration_samples


def test_short_capture_finishes_with_spine_drift_without_resetting():
    values = physical_series_captures()[POSE_SEQUENCE[0]].average
    recorder = FramePoseRecorder(POSE_SEQUENCE[0], 0.)
    for i in range(1, 91):
        # 2 deg/s would cross the 15-degree limit during the former 20-second capture.
        drift = quaternion_to_matrix(quaternion_from_rotation_vector((0., 0., math.radians(2 * i * .02))))
        samples = {**values, 'spine': matrix_to_quaternion(drift @ quaternion_to_matrix(values['spine']))}
        final = recorder.add_frame(samples, (i, i*20), i*.02)
    assert final is not None
    assert final.ended_s - final.started_s == pytest.approx(1.5)
    assert recorder.rejected['unstable_or_discarded'] == 0
    assert math.degrees(angular_distance(final.first['spine'], final.last['spine'])) > 2.


def test_packet_gap_cannot_complete_a_timed_capture():
    values = physical_series_captures()[POSE_SEQUENCE[0]].average
    recorder = FramePoseRecorder(POSE_SEQUENCE[0], 0.)
    for i in range(1, 31):
        assert recorder.add_frame(values, (i,i), i*.02) is None
    before = recorder.elapsed_s
    recorder.add_frame({}, None, .8)
    recorder.add_frame(values, (30,30), .9)
    assert recorder.elapsed_s == before and not recorder.completed
    with pytest.raises(ValueError, match='Прервался поток'):
        recorder.add_frame(values, (100,100), 2.1)
    assert not recorder.completed


def test_timed_capture_still_requires_enough_real_samples():
    values = physical_series_captures()[POSE_SEQUENCE[0]].average
    recorder = FramePoseRecorder(POSE_SEQUENCE[0], 0.)
    with pytest.raises(ValueError, match='недостаточно кадров'):
        for i in range(1, 24):
            recorder.add_frame(values, (i,i), i*.2)
    assert not recorder.completed


def test_movement_previous_pose_and_timeout_do_not_count_or_create_frames():
    captures = physical_series_captures()
    initial = captures[POSE_SEQUENCE[0]]
    recorder = FramePoseRecorder('semaphore_a', 0., previous=initial)
    for i in range(1, 30):
        recorder.add_frame(initial.average, (i,i), i*.02)
    assert recorder.accepted == 0
    for i in range(30, 60):
        recorder.add_frame(captures['semaphore_a'].average, (i,i), i*.02)
    assert recorder.accepted > 0
    recorder.add_frame(initial.average, (60,60), 1.2)
    assert recorder.accepted == 0
    recorder.add_frame({}, None, 1.3)
    with pytest.raises(ValueError, match='Тайм-аут'):
        recorder.timeout(121.)


def test_100_frames_at_supported_100_hz_do_not_fail_legacy_one_second_rule():
    captures = {name: replace(pose, ended_s=pose.started_s + .99)
                for name, pose in physical_series_captures().items()}
    result = calibrate_series(captures)
    assert max(result.scores_deg.values()) < 2e-6
