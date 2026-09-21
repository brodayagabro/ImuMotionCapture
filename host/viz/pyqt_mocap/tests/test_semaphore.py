"""Synthetic planar motions, neutral closure, and one-round semaphore capture."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from pyqt_mocap import guided_dialog
from pyqt_mocap.calibration import load_profile, profile_document, save_profile
from pyqt_mocap.mocap_core import SEGMENT_NAMES, matrix_to_quaternion, quaternion_to_matrix
from pyqt_mocap.semaphore_calibration import (
    ANGLES, SEMAPHORE_POSES, base_pose, calibrate_semaphore, target_direction,
)
from pyqt_mocap.tests.test_calibration import captured_pose, rotation_x, rotation_y, rotation_z


def semaphore_captures(mount=None):
    mount = np.eye(3) if mount is None else mount
    captures = {}
    for index, name in enumerate(SEMAPHORE_POSES):
        rotations = {}
        for segment in SEGMENT_NAMES:
            angle = 0.
            if base_pose(name) in ANGLES and segment != "spine":
                angle = ANGLES[base_pose(name)][0 if segment.endswith(".L") else 1]
            rotations[segment] = mount.T @ rotation_y(math.radians(angle)) @ mount
        captures[name] = captured_pose(name, rotations, 100. + index * 10.)
    return captures


def test_semaphore_angles_and_recovery():
    np.testing.assert_allclose(target_direction("sem_zh", "shoulder.L"), (-1, 0, 0), atol=1e-12)
    np.testing.assert_allclose(target_direction("sem_e", "shoulder.L"), (0, 0, -1))
    np.testing.assert_allclose(target_direction("sem_d", "shoulder.R"), (-math.sqrt(.5), 0, -math.sqrt(.5)))
    np.testing.assert_allclose(target_direction("sem_d", "shoulder.L"), (-1, 0, 0), atol=1e-12)
    assert target_direction("sem_d", "forearm.R") is None
    result = calibrate_semaphore(semaphore_captures(rotation_x(.3) @ rotation_z(.5)))
    assert result.method == "semaphore_xz"
    assert max(result.scores_deg.values()) < 2e-6
    assert not result.repeatability_deg
    assert max(result.closure_error_deg.values()) < 2e-6
    for q in result.axis_alignment_quaternions.values():
        assert np.linalg.det(quaternion_to_matrix(q)) == pytest.approx(1.)


def test_unknown_d_flexion_does_not_change_right_forearm_fit():
    captures = semaphore_captures(rotation_z(.3))
    before = calibrate_semaphore(captures)
    captures["sem_d"].average["forearm.R"] = matrix_to_quaternion(rotation_x(.9))
    after = calibrate_semaphore(captures)
    np.testing.assert_allclose(after.axis_alignment_quaternions["forearm.R"],
                               before.axis_alignment_quaternions["forearm.R"])
    assert after.scores_deg["forearm.R"] == pytest.approx(before.scores_deg["forearm.R"])


def test_final_a_checks_closure_without_fitting_it():
    captures = semaphore_captures(rotation_z(.3))
    before = calibrate_semaphore(captures)
    from pyqt_mocap.tests.test_calibration import raw_quaternion_for
    captures["return_a_pose"].average["shoulder.L"] = raw_quaternion_for(
        rotation_z(-.3) @ rotation_y(.3) @ rotation_z(.3), "shoulder.L")
    after = calibrate_semaphore(captures)
    np.testing.assert_allclose(after.axis_alignment_quaternions["shoulder.L"],
                               before.axis_alignment_quaternions["shoulder.L"])
    assert after.scores_deg["shoulder.L"] > 1.


def test_prior_unobservable_twist_is_preserved():
    prior = {s: matrix_to_quaternion(rotation_y(.4)) for s in SEGMENT_NAMES}
    result = calibrate_semaphore(semaphore_captures(), prior_alignment=prior)
    for segment in SEGMENT_NAMES:
        np.testing.assert_allclose(result.axis_alignment_quaternions[segment], prior[segment], atol=1e-10)


def test_axial_twist_does_not_bias_direction_fit():
    from pyqt_mocap.tests.test_calibration import raw_quaternion_for
    mount = rotation_z(.4)
    captures = semaphore_captures(mount)
    for index, pose in enumerate(ANGLES):
        for segment in SEGMENT_NAMES:
            if segment == "spine":
                continue
            angle = math.radians(ANGLES[pose][0 if segment.endswith(".L") else 1])
            # Local Z twist leaves a straight segment's direction unchanged,
            # but DOES tilt the quaternion's net axis of rotation.
            rotation = rotation_y(angle) @ rotation_z((index - 2) * .15)
            captures[pose].average[segment] = raw_quaternion_for(mount.T @ rotation @ mount, segment)
    result = calibrate_semaphore(captures)
    assert max(result.scores_deg.values()) < .1


def test_reported_error_uses_deployed_final_neutral():
    from pyqt_mocap.calibration import NEUTRAL_DIRECTIONS
    from pyqt_mocap.mocap_core import mapped_sensor_quaternion
    from pyqt_mocap.tests.test_calibration import raw_quaternion_for
    captures = semaphore_captures(rotation_z(.3))
    segment = "shoulder.L"
    captures["return_a_pose"].average[segment] = raw_quaternion_for(
        rotation_z(-.3) @ rotation_y(.2) @ rotation_z(.3), segment)
    result = calibrate_semaphore(captures)
    correction = quaternion_to_matrix(result.axis_alignment_quaternions[segment])
    neutral = quaternion_to_matrix(mapped_sensor_quaternion(
        captures["return_a_pose"].average[segment], result.axis_maps[segment]))
    errors = []
    for pose in ANGLES:
        rotation = quaternion_to_matrix(mapped_sensor_quaternion(
            captures[pose].average[segment], result.axis_maps[segment]))
        predicted = correction @ rotation @ neutral.T @ correction.T @ NEUTRAL_DIRECTIONS[segment]
        errors.append(math.degrees(math.acos(np.clip(predicted @ target_direction(pose, segment), -1, 1))))
    assert result.scores_deg[segment] == pytest.approx(float(np.sqrt(np.mean(np.square(errors)))))
    assert result.closure_error_deg[segment] == pytest.approx(math.degrees(.2))


def test_large_failure_to_return_to_a_is_rejected():
    from pyqt_mocap.tests.test_calibration import raw_quaternion_for
    captures = semaphore_captures()
    captures["return_a_pose"].average["shoulder.L"] = raw_quaternion_for(rotation_y(.8), "shoulder.L")
    with pytest.raises(ValueError, match="возврат в A-позу"):
        calibrate_semaphore(captures)


def test_single_round_profile_roundtrip(tmp_path):
    result = calibrate_semaphore(semaphore_captures())
    path = tmp_path / "semaphore.json"
    save_profile(path, profile_document({}, result))
    data = load_profile(path)["calibration"]
    assert data["method"] == "semaphore_xz"
    assert len(data["poses"]) == 7
    assert not data["repeatability_deg"]
    assert data["closure_error_deg"]["shoulder.L"] == pytest.approx(0.)


def test_missing_motion_and_disabled_segments():
    captures = semaphore_captures()
    for capture in captures.values():
        capture.average["shoulder.L"] = np.array((1., 0., 0., 0.))
    with pytest.raises(ValueError, match="недостаточно движений"):
        calibrate_semaphore(captures)
    result = calibrate_semaphore(captures, enabled_segments=set(SEGMENT_NAMES) - {"shoulder.L"})
    assert result.scores_deg["shoulder.L"] == 0.


@pytest.mark.gui
def test_automatic_semaphore_wizard(monkeypatch):
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    clock = [100.]
    monkeypatch.setattr(guided_dialog, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    captures = semaphore_captures()
    generation = [0]

    def snapshot():
        generation[0] += 1
        pose = captures[SEMAPHORE_POSES[min(dialog.stage_index, len(SEMAPHORE_POSES) - 1)]]
        return {s: (q, clock[0], generation[0]) for s, q in pose.average.items()}

    dialog = guided_dialog.GuidedCalibrationDialog(snapshot, semaphore=True)
    results = []
    dialog.result_ready.connect(results.append)
    dialog.show()
    try:
        dialog.capture_button.click()
        for stage in range(7):
            assert dialog.stage_index == stage
            clock[0] = dialog.phase_started_s + 2.5
            dialog._tick()
            app.processEvents()  # Opposite directions and bent D must also paint.
            clock[0] = dialog.phase_started_s + 5.
            dialog._tick()
            start = clock[0]
            for index in range(1, 51):
                clock[0] = start + index * .1
                dialog._tick()
        assert dialog.phase == "finished"
        assert len(results) == 1
        assert len(results[0].captures) == 7
        assert max(results[0].scores_deg.values()) < 2e-6
        assert not dialog.timer.isActive()
    finally:
        dialog.reject()
