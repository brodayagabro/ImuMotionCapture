"""Independent SO(3) oracle and downstream calibration regression tests."""
import json
import math

import numpy as np
import pytest

from pyqt_mocap import quaternion_utils as q
from pyqt_mocap.orientation_frames import SegmentCalibration
from pyqt_mocap.calibration import profile_document, load_profile, save_profile, restore_profile_calibration
from pyqt_mocap.mocap_core import (
    DEFAULT_AXIS_MAPS, DEFAULT_SENSOR_MAPPING, SEGMENT_NAMES, MotionCaptureModel,
    mapped_sensor_quaternion,
)
from pyqt_mocap.t_pose_calibration import calibrate_t_pose
from pyqt_mocap.tests.test_t_pose_calibration import physical_captures, calibrated_model, directions_after_packet


def test_math_against_scipy_random_rotations_and_signs():
    Rotation = pytest.importorskip('scipy.spatial.transform').Rotation
    Slerp = pytest.importorskip('scipy.spatial.transform').Slerp
    rng = np.random.default_rng(174)
    values = Rotation.random(120, random_state=rng)
    for i in range(0, 120, 2):
        a, b = q.from_xyzw(values[i:i+2].as_quat())
        np.testing.assert_allclose(q.quaternion_to_matrix(q.compose(a, b)), (values[i]*values[i+1]).as_matrix(), atol=1e-12)
        assert q.angular_distance(q.compose(a, q.inverse(a)), (1,0,0,0)) < 1e-12
        assert q.angular_distance(q.compose(a, (1,0,0,0)), a) < 1e-12
        assert q.angular_distance(q.compose((1,0,0,0), a), a) < 1e-12
        assert q.angular_distance(a, -a) < 1e-12
        assert q.angular_distance(a, b) == pytest.approx((values[i].inv()*values[i+1]).magnitude(), abs=1e-12)
        np.testing.assert_allclose(q.quaternion_to_matrix(q.quaternion_slerp(a,-b,.31)),
                                   Slerp([0,1], values[i:i+2])([.31]).as_matrix()[0], atol=1e-7)
        np.testing.assert_allclose(q.rotate_vector(a,[1,2,3]), values[i].apply([1,2,3]), atol=1e-12)
        assert q.angular_distance(q.matrix_to_quaternion(values[i].as_matrix()), a) < 1e-12


@pytest.mark.parametrize('axis,source,target', [([1,0,0],[0,1,0],[0,0,1]),
    ([0,1,0],[0,0,1],[1,0,0]), ([0,0,1],[1,0,0],[0,1,0])])
def test_known_right_handed_rotations(axis,source,target):
    rotation = q.quaternion_from_rotation_vector(np.array(axis)*math.pi/2)
    np.testing.assert_allclose(q.rotate_vector(rotation,source), target, atol=1e-12)


def test_rotation_average_is_sign_and_order_invariant_scipy_mean():
    Rotation = pytest.importorskip('scipy.spatial.transform').Rotation
    values = Rotation.from_euler('xyz', [[1,2,3],[100,30,20],[-80,10,35],[4,50,-8]], degrees=True)
    quats = q.from_xyzw(values.as_quat())
    expected = q.from_xyzw(values.mean().as_quat())
    quats[::2] *= -1
    assert q.angular_distance(q.average_quaternions(quats), expected) < 1e-12
    assert q.angular_distance(q.average_quaternions(quats[::-1]), expected) < 1e-12


def test_single_pose_mounting_recovers_multiple_noncommuting_poses():
    Rotation = pytest.importorskip('scipy.spatial.transform').Rotation
    # Known 30-degree X / 20-degree Z extrinsic mounting, distinct world frame.
    mounting = Rotation.from_euler('xz',[30,20],degrees=True)
    world = Rotation.from_euler('xyz',[15,-35,63],degrees=True)
    targets = Rotation.from_euler('xyz',[[20,10,-15],[90,0,0],[0,-90,0],[40,60,-80],[-20,130,70]],degrees=True)
    sensors = world.inv()*targets*mounting.inv()
    frame = SegmentCalibration.from_pose(q.from_xyzw(sensors[0].as_quat()), q.from_xyzw(targets[0].as_quat()), q.from_xyzw(world.as_quat()))
    assert q.angular_distance(frame.q_sensor_segment, q.from_xyzw(mounting.as_quat())) < 1e-12
    for sensor,target in zip(sensors,targets):
        assert q.angular_distance(frame.apply(q.from_xyzw(sensor.as_quat())), q.from_xyzw(target.as_quat())) < 1e-12
    # Same frame through the production model, whose reference target is identity.
    neutral = world.inv()*mounting.inv()
    align = {s: q.from_xyzw(world.as_quat()) for s in SEGMENT_NAMES}
    raw0 = {s: q.from_xyzw(neutral.as_quat()) for s in SEGMENT_NAMES}
    maps = {s: ('+X','+Y','+Z') for s in SEGMENT_NAMES}
    model = MotionCaptureModel(DEFAULT_SENSOR_MAPPING,maps,'raw',smooth_alpha=1.)
    model.set_guided_calibration(maps,{s:(0,0,0) for s in SEGMENT_NAMES},raw0,0.,align)
    for sensor,target in zip(sensors,targets):
        values = {s:q.from_xyzw(sensor.as_quat()) for s in SEGMENT_NAMES}
        directions_after_packet(model,values,1.)
        assert q.angular_distance(model.orientations()['forearm.L'],q.from_xyzw(target.as_quat())) < 1e-12


def test_stable_input_stays_stable_despite_nonzero_estimated_rate():
    result = calibrate_t_pose(physical_captures())
    model = calibrated_model(result)
    model.set_drift_compensation({s:(.02,-.01,.03) for s in SEGMENT_NAMES})
    model.drift_reference_s = 100.
    signature = model.calibration_signature()
    for timestamp in (100.,130.,1000.):
        directions_after_packet(model,result.captures['return_a_pose'].average,timestamp)
        assert q.angular_distance(model.orientations()['forearm.L'],(1,0,0,0)) < 1e-12
        assert model.calibration_signature() == signature
    model.axis_alignment_quaternion['forearm.L'] = q.quaternion_from_rotation_vector((0,.2,0))
    with pytest.raises(AssertionError, match='outside an explicit'):
        model.assert_calibration_invariant()


def test_profile_reload_restores_same_pose_and_reference_with_sign_changes(tmp_path):
    result = calibrate_t_pose(physical_captures())
    original = calibrated_model(result)
    document = profile_document({'sensor_mapping':DEFAULT_SENSOR_MAPPING,'enabled_segments':list(SEGMENT_NAMES)},result)
    # A sign change in saved quaternions must not change the rotation.
    for field in ('axis_alignment_quaternions_wxyz','neutral_raw_quaternions_wxyz'):
        for segment in SEGMENT_NAMES:
            document['calibration'][field][segment] = [-x for x in document['calibration'][field][segment]]
    path=tmp_path/'profile.json'
    save_profile(path,document)
    loaded=load_profile(path)
    restored=MotionCaptureModel(DEFAULT_SENSOR_MAPPING,DEFAULT_AXIS_MAPS,'raw',smooth_alpha=1.)
    restored.set_enabled_segments(SEGMENT_NAMES)
    assert restore_profile_calibration(restored,loaded,reference_s=999.)
    assert not restored.neutral_pending
    for pose in result.captures.values():
        directions_after_packet(original,pose.average,1000.)
        directions_after_packet(restored,pose.average,1000.)
        for segment in SEGMENT_NAMES:
            assert q.angular_distance(original.orientations()[segment],restored.orientations()[segment]) < 1e-12
    assert restored.sensor_mapping == loaded['application']['sensor_mapping']


def test_constant_twist_has_no_longitudinal_swing_and_twist_error_is_visible():
    swing=q.quaternion_from_rotation_vector((.6,0,0))
    twist=q.quaternion_from_rotation_vector((0,0,.7))
    combined=q.compose(swing,twist)
    np.testing.assert_allclose(q.rotate_vector(combined,[0,0,-1]),q.rotate_vector(swing,[0,0,-1]),atol=1e-12)
    assert abs(q.twist_angle(q.relative(swing,combined),[0,0,-1])) == pytest.approx(.7)
