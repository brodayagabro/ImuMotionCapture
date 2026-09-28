import numpy as np
import pytest

from pyqt_mocap.human_canvas_gl import (
    _segment_positions, _static_bone_segments, bone_mesh, OpenGLHumanCanvas,
)
from pyqt_mocap.mocap_core import (
    DEFAULT_SENSOR_MAPPING, SEGMENT_NAMES, compute_body_pose,
    quaternion_from_rotation_vector, quaternion_to_matrix,
)


def test_segment_positions_flattens_pairs_for_gl_lines() -> None:
    segments = (
        (np.array((0.0, 1.0, 2.0)), np.array((3.0, 4.0, 5.0))),
        (np.array((6.0, 7.0, 8.0)), np.array((9.0, 10.0, 11.0))),
    )

    positions = _segment_positions(segments)

    assert positions.dtype == np.float32
    assert positions.shape == (4, 3)
    np.testing.assert_array_equal(
        positions,
        np.array(
            (
                (0.0, 1.0, 2.0),
                (3.0, 4.0, 5.0),
                (6.0, 7.0, 8.0),
                (9.0, 10.0, 11.0),
            ),
            dtype=np.float32,
        ),
    )


@pytest.mark.parametrize("end", [(0, 0, 1), (0, 0, -1), (1, 0, 0), (0, 1, 0), (.3, -.7, .4)])
def test_bone_mesh_has_joint_tips_and_outward_faces(end):
    start = np.array((.1, .2, .3))
    end = start + end
    vertices, faces = bone_mesh(start, end)
    assert vertices.shape == (6, 3)
    assert faces.shape == (8, 3)
    np.testing.assert_allclose(vertices[:2], (start, end))
    center = start + .2 * (end - start)
    np.testing.assert_allclose(vertices[2:].mean(axis=0), center, atol=1e-7)
    for face in faces:
        triangle = vertices[face]
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        assert normal @ (triangle.mean(axis=0) - center) > 0


def test_zero_length_bone_is_empty():
    vertices, faces = bone_mesh((0, 0, 0), (0, 0, 0))
    assert vertices.shape == faces.shape == (0, 3)


def test_cross_section_rotates_with_segment():
    start = np.zeros(3)
    end = np.array((.08, .17, .7))
    rotation = quaternion_to_matrix(quaternion_from_rotation_vector((.3, -.8, .4)))
    vertices, faces = bone_mesh(start, end)
    transformed, transformed_faces = bone_mesh(rotation @ start, rotation @ end, rotation)
    np.testing.assert_allclose(transformed, vertices @ rotation.T, atol=1e-7)
    np.testing.assert_array_equal(faces, transformed_faces)


@pytest.mark.gui
def test_gl_scene_updates_existing_bone_meshes():
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    canvas = OpenGLHumanCanvas()
    try:
        items = tuple(canvas.view.items)
        orientations = {name: quaternion_from_rotation_vector((.3, .5, -.2))
                        for name in SEGMENT_NAMES}
        pose = compute_body_pose(orientations)
        canvas.update_pose(orientations, DEFAULT_SENSOR_MAPPING, {"shoulder.L"})
        assert tuple(canvas.view.items) == items
        assert len(canvas.static_bones) == len(pose.static_segments) + 1
        for bone, segment in zip(canvas.static_bones, _static_bone_segments(pose), strict=True):
            np.testing.assert_allclose(bone.opts["meshdata"].vertexes()[:2], segment, atol=1e-7)
        for name, bone in canvas.tracked_bones.items():
            np.testing.assert_allclose(bone.opts["meshdata"].vertexes()[:2],
                                       pose.tracked_segments[name], atol=1e-7)
            assert not bone.opts["smooth"]
            assert bone.opts["drawEdges"]
        assert canvas.tracked_bones["spine"].opts["color"] == (.70, .74, .79, 1.)
        assert canvas.segment_labels["forearm.L"].text == ""
    finally:
        canvas.close()


@pytest.mark.parametrize("rotation", [(0., 0., 0.), (.4, -.6, .8)])
def test_clavicles_share_center_and_follow_torso(rotation):
    pose = compute_body_pose({"spine": quaternion_from_rotation_vector(rotation)})
    segments = _static_bone_segments(pose)
    left = pose.tracked_segments["shoulder.L"][0]
    right = pose.tracked_segments["shoulder.R"][0]
    center = (left + right) / 2
    np.testing.assert_allclose(segments[1], (center, left))
    np.testing.assert_allclose(segments[2], (center, right))
    np.testing.assert_allclose(segments[0], pose.static_segments[0])
    np.testing.assert_allclose(segments[3:], pose.static_segments[2:])
