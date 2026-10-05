from dataclasses import replace
import math
import numpy as np
import pytest
from pyqt_mocap.mocap_core import compute_body_pose, quaternion_from_rotation_vector, quaternion_to_matrix
from scripts.noitom_comparison.noitom_adapter import (
    NoitomAdapter, convert_noitom_position, convert_noitom_rotation, rotation_between, world_transforms,
    SDK_TO_PROJECT_BASIS, sdk_rotation_to_project, sdk_position_to_project, torso_heading,
)
from scripts.noitom_comparison.noitom_client import FakeNoitomClient, NoitomClient, NoitomError
from scripts.noitom_comparison.config import ComparisonConfig


def test_coordinate_change_applies_to_rotations_and_positions():
    # Right-handed Y-up to Z-up, including forward sign.
    basis = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    q = quaternion_from_rotation_vector((.3, -.6, .9))
    converted = convert_noitom_rotation(q[[1, 2, 3, 0]], basis)
    point = np.array([1., 2., 3.])
    np.testing.assert_allclose(quaternion_to_matrix(converted) @ convert_noitom_position(point, basis),
                               basis @ quaternion_to_matrix(q) @ point, atol=1e-12)
    np.testing.assert_allclose(convert_noitom_rotation([0, 0, 0, 1]), [1, 0, 0, 0])


def test_parent_rotation_is_composed_before_retargeting():
    frame = FakeNoitomClient.frame(0, 123)
    # Rotating the pelvis must rotate the entire upper body, not just spine.
    q = quaternion_from_rotation_vector((0, 0, math.pi / 2))
    rotated = replace(frame, joints={**frame.joints, "Hips": replace(frame.joints["Hips"], rotation=q)})
    adapter = NoitomAdapter()
    before, after = adapter.orientations(frame), adapter.orientations(rotated)
    for name in before:
        np.testing.assert_allclose(quaternion_to_matrix(after[name]),
            quaternion_to_matrix(q) @ quaternion_to_matrix(before[name]), atol=1e-12)


def test_source_t_pose_offsets_become_downward_a_pose_without_live_calibration():
    pose = compute_body_pose(NoitomAdapter().orientations(FakeNoitomClient.frame(0, 10)))
    for name in ("shoulder.L", "shoulder.R", "forearm.L", "forearm.R"):
        start, end = pose.tracked_segments[name]
        assert end[2] < start[2]
        assert abs(end[0] - start[0]) < 1e-10
    np.testing.assert_allclose(pose.tracked_segments["spine"], compute_body_pose({}).tracked_segments["spine"])


def test_root_translation_has_no_effect_on_mannequin():
    frame = FakeNoitomClient.frame(7, 10)
    moved = replace(frame, joints={**frame.joints, "Hips": replace(frame.joints["Hips"], position=np.ones(3) * 99)})
    before, after = NoitomAdapter().orientations(frame), NoitomAdapter().orientations(moved)
    for name in before:
        np.testing.assert_allclose(before[name], after[name])


def test_hierarchy_cycle_and_missing_offsets_fail_explicitly():
    frame = FakeNoitomClient.frame(0, 10)
    broken = replace(frame, joints={**frame.joints, "Hips": replace(frame.joints["Hips"], parent="Neck")})
    with pytest.raises(ValueError, match="Cycle"):
        world_transforms(broken)
    broken = replace(frame, joints={**frame.joints, "LeftHand": replace(frame.joints["LeftHand"], default_position=None, position=None)})
    with pytest.raises(ValueError, match="offset"):
        NoitomAdapter().orientations(broken)


def test_disabled_segment_offset_does_not_block_selected_arm_capture():
    frame = FakeNoitomClient.frame(17, 10)
    broken = replace(frame, joints={**frame.joints,
        "Neck": replace(frame.joints["Neck"], default_position=None, position=None)})
    adapter = NoitomAdapter()
    expected = adapter.orientations(frame)
    selected = set(expected) - {"spine"}
    actual = adapter.orientations(broken, selected)
    assert set(actual) == selected
    for name in selected:
        np.testing.assert_allclose(actual[name], expected[name])
    assert adapter.orientations(broken, ()) == {}


def test_stream_without_default_offsets_uses_parent_local_child_direction():
    frame = FakeNoitomClient.frame(17, 10)
    streamed = replace(frame, joints={name: replace(joint, default_position=None) for name, joint in frame.joints.items()})
    adapter = NoitomAdapter()
    expected, actual = adapter.orientations(frame), adapter.orientations(streamed)
    for name in expected:
        np.testing.assert_allclose(actual[name], expected[name])


def test_sdk_default_basis_and_centimetres_match_project_axes():
    np.testing.assert_allclose(sdk_position_to_project([100, 200, 300]), [-1, 3, 2])
    for vector in [(0, 0, 0), (.2, -.9, .6), (1.1, .4, -2.3)]:
        q = quaternion_from_rotation_vector(vector)
        actual = sdk_rotation_to_project(q[[1, 2, 3, 0]])
        expected = convert_noitom_rotation(q[[1, 2, 3, 0]], SDK_TO_PROJECT_BASIS)
        np.testing.assert_allclose(quaternion_to_matrix(actual), quaternion_to_matrix(expected), atol=1e-12)


def test_antiparallel_rest_vector():
    np.testing.assert_allclose(rotation_between([0, 0, -1], [0, 0, 1]) @ [0, 0, -1], [0, 0, 1])


def test_heading_alignment_removes_reference_yaw_and_preserves_later_turns():
    frame = FakeNoitomClient.frame(17, 123)
    def yawed(angle):
        return replace(frame, joints={**frame.joints, "Hips": replace(frame.joints["Hips"],
            rotation=quaternion_from_rotation_vector((0, 0, angle)))})
    adapter = NoitomAdapter()
    reference = yawed(-.9)
    offset = -torso_heading(reference)
    assert offset == pytest.approx(.9)
    selected = {"shoulder.L", "forearm.L"}  # reference works with spine unchecked
    baseline = adapter.orientations(frame, selected)
    aligned = adapter.orientations(reference, selected, heading_offset_rad=offset)
    later = adapter.orientations(yawed(-.9 + .3), selected, heading_offset_rad=offset)
    turn = quaternion_to_matrix(quaternion_from_rotation_vector((0, 0, .3)))
    for name in selected:
        np.testing.assert_allclose(quaternion_to_matrix(aligned[name]), quaternion_to_matrix(baseline[name]), atol=1e-12)
        np.testing.assert_allclose(quaternion_to_matrix(later[name]), turn @ quaternion_to_matrix(baseline[name]), atol=1e-12)
    # One common rotation cannot change the elbow's internal angle.
    raw = adapter.orientations(reference, selected)
    dots = []
    for pose in (raw, aligned):
        vectors = [quaternion_to_matrix(pose[name]) @ [0, 0, -1] for name in sorted(selected)]
        dots.append(vectors[0] @ vectors[1])
    assert dots[0] == pytest.approx(dots[1])


def test_heading_alignment_rejects_unavailable_shoulder_positions():
    frame = FakeNoitomClient.frame(0, 123)
    broken = replace(frame, joints={**frame.joints, "LeftArm": replace(frame.joints["LeftArm"], position=None)})
    with pytest.raises(ValueError, match="положений"):
        torso_heading(broken)


def test_missing_sdk_is_lazy_and_actionable(tmp_path):
    client = NoitomClient(ComparisonConfig(sdk_library=str(tmp_path / "missing.dll")))
    with pytest.raises(NoitomError, match="sdk-library"):
        client.connect()
    client.disconnect()
