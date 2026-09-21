"""Read exported BVH and compare forward kinematics with the displayed body."""

import math
import re
import shutil
import subprocess

import numpy as np
import pytest

from storage.bvh import TO_BVH, euler_zxy
from pyqt_mocap.window_recording import create_recording, export_geometry
from pyqt_mocap.mocap_core import (
    SEGMENT_NAMES, compute_body_pose, quaternion_from_rotation_vector, quaternion_to_matrix,
)


def rotation(axis, angle):
    c, s = math.cos(angle), math.sin(angle)
    if axis == "X":
        return np.array(((1, 0, 0), (0, c, -s), (0, s, c)))
    if axis == "Y":
        return np.array(((c, 0, s), (0, 1, 0), (-s, 0, c)))
    return np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))


def read_bvh(path):
    hierarchy, motion = path.read_text().split("MOTION\n")
    tokens = iter(re.findall(r"[^\s{}]+|[{}]", hierarchy))
    assert next(tokens) == "HIERARCHY"
    joints = []

    def joint(parent):
        name = next(tokens)
        assert next(tokens) == "{"
        assert next(tokens) == "OFFSET"
        offset = np.array([float(next(tokens)) for _ in range(3)])
        assert next(tokens) == "CHANNELS"
        channels = [next(tokens) for _ in range(int(next(tokens)))]
        index = len(joints)
        joints.append((name, parent, offset, channels))
        while (token := next(tokens)) != "}":
            if token == "JOINT":
                joint(index)
            else:
                assert token == "End" and next(tokens) == "Site"
                assert next(tokens) == "{"
                assert next(tokens) == "OFFSET"
                for _ in range(3):
                    float(next(tokens))
                assert next(tokens) == "}"

    assert next(tokens) == "ROOT"
    joint(None)
    lines = motion.splitlines()
    count = int(lines[0].split(":")[1])
    dt = float(lines[1].split(":")[1])
    frames = np.array([[float(v) for v in line.split()] for line in lines[2:]])
    assert frames.shape == (count, sum(len(j[3]) for j in joints))
    return joints, frames, dt


def positions(joints, frame):
    world = []
    result = {}
    cursor = 0
    for name, parent, offset, channels in joints:
        r = np.eye(3)
        offset = offset.copy()
        for channel in channels:
            value = frame[cursor]
            cursor += 1
            if channel.endswith("position"):
                offset["XYZ".index(channel[0])] += value
            else:
                r = r @ rotation(channel[0], math.radians(value))
        if parent is None:
            position = offset
        else:
            parent_r, parent_p = world[parent]
            position = parent_p + parent_r @ offset
            r = parent_r @ r
        world.append((r, position))
        result[name] = position
    return result


def test_export_reconstructs_viewer_joints(tmp_path):
    recording = create_recording(20)
    rng = np.random.default_rng(71)
    poses = []
    for index in range(20):
        quats = {name: quaternion_from_rotation_vector(rng.normal(size=3))
                 for name in SEGMENT_NAMES}
        poses.append(compute_body_pose(quats))
        recording.append(index / 20, {n: quaternion_to_matrix(q) for n, q in quats.items()})
    path = tmp_path / "motion.bvh"
    assert recording.save(path) == 20
    joints, frames, dt = read_bvh(path)
    assert dt == pytest.approx(.05)
    names = ("Hips", "Chest", "Head", "LeftArm", "RightArm", "LeftForeArm",
             "RightForeArm", "LeftHand", "RightHand", "LeftUpLeg", "RightUpLeg",
             "LeftLeg", "RightLeg", "LeftFoot", "RightFoot")
    for pose, frame in zip(poses, frames):
        actual = positions(joints, frame)
        for name, position in zip(names, pose.joints):
            scale, origin = export_geometry()
            np.testing.assert_allclose(actual[name], TO_BVH @ ((position - origin) * scale), atol=1e-6)


def test_export_height_is_170cm_and_feet_are_on_floor(tmp_path):
    recording = create_recording(20)
    recording.append(0., {name: np.eye(3) for name in SEGMENT_NAMES})
    path = tmp_path / "height.bvh"
    recording.save(path)
    joints, frames, _ = read_bvh(path)
    points = positions(joints, frames[0])
    # Read the actual serialized head end, rather than assuming its length.
    head = path.read_text().split("JOINT Head", 1)[1].split("End Site", 1)[1]
    tip_offset = np.array([float(v) for v in re.search(r"OFFSET\s+([^\n]+)", head)[1].split()])
    top = points["Head"] + tip_offset
    assert points["LeftFoot"][1] == pytest.approx(0., abs=1e-7)
    assert points["RightFoot"][1] == pytest.approx(0., abs=1e-7)
    assert top[1] == pytest.approx(1.70, abs=1e-7)
    assert recording.units_per_meter == 1.


def test_leaf_end_sites_have_explicit_lengths(tmp_path):
    recording = create_recording(40)
    recording.append(0., {name: np.eye(3) for name in SEGMENT_NAMES})
    path = tmp_path / "leaf_lengths.bvh"
    recording.save(path)
    text = path.read_text()
    for name in ("Head", "LeftHand", "RightHand", "LeftFoot", "RightFoot"):
        end = text.split("JOINT " + name, 1)[1].split("End Site", 1)[1]
        vector = np.array([float(v) for v in re.search(r"OFFSET\s+([^\n]+)", end)[1].split()])
        assert np.linalg.norm(vector) > .01
        if name.endswith("Hand"):
            np.testing.assert_allclose(vector, (0., -.18, 0.))


@pytest.mark.skipif(shutil.which("blender") is None, reason="Blender not installed")
def test_blender_import_preserves_hand_size_height_and_motion(tmp_path):
    recording = create_recording(40)
    neutral = {name: np.eye(3) for name in SEGMENT_NAMES}
    recording.append(0., neutral)
    moved = dict(neutral)
    moved["shoulder.L"] = rotation("Y", .7)
    moved["forearm.L"] = rotation("X", -.5) @ rotation("Y", 1.)
    recording.append(.025, moved)
    path = tmp_path / "synthetic_blender_test.bvh"
    recording.save(path)
    joints, frames, _ = read_bvh(path)
    # Convert BVH Y-up to Blender Z-up, independently of the exporter.
    conversion = np.array(((1, 0, 0), (0, 0, -1), (0, 1, 0)))
    expected = [{name: (conversion @ value).tolist()
                 for name, value in positions(joints, frame).items()} for frame in frames]
    script = f"""
import bpy
from mathutils import Vector
bpy.ops.import_anim.bvh(filepath={str(path)!r}, global_scale=1.,
                        update_scene_fps=True, update_scene_duration=True)
obj = bpy.context.object
for name in ('LeftHand', 'RightHand'):
    assert abs(obj.data.bones[name].length - .18) < 1e-6
expected = {expected!r}
for index, pose in enumerate(expected):
    bpy.context.scene.frame_set(index + 1)
    for name, point in pose.items():
        assert (obj.matrix_world @ obj.pose.bones[name].head - Vector(point)).length < 1e-5, name
bpy.context.scene.frame_set(1)
top = obj.matrix_world @ obj.pose.bones['Head'].tail
foot = obj.matrix_world @ obj.pose.bones['LeftFoot'].head
assert abs(top.z - foot.z - 1.70) < 1e-6
assert abs(bpy.context.scene.render.fps / bpy.context.scene.render.fps_base - 40.) < 1e-6
print('BVH_IMPORT_VERIFIED')
"""
    result = subprocess.run(
        [shutil.which("blender"), "--background", "--factory-startup",
         "--python-exit-code", "1", "--python-expr", script],
        capture_output=True, text=True, timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "zero length" not in result.stdout.lower()
    assert "BVH_IMPORT_VERIFIED" in result.stdout


@pytest.mark.parametrize("units", [0., -1., float("nan"), float("inf")])
def test_invalid_unit_scale_is_rejected(units):
    from storage.bvh import BvhRecording, Joint
    with pytest.raises(ValueError, match="units_per_meter"):
        BvhRecording([Joint("Root", None, (0., 0., 0.))], (0., 0., 0.), units_per_meter=units)


@pytest.mark.parametrize("x", [90, -90, 89.999, -89.999, 0, 180])
def test_euler_at_singularities(x):
    matrix = rotation("Z", .7) @ rotation("X", math.radians(x)) @ rotation("Y", -.4)
    z, x, y = np.radians(euler_zxy(matrix))
    np.testing.assert_allclose(rotation("Z", z) @ rotation("X", x) @ rotation("Y", y),
                               matrix, atol=1e-7)


def test_resample_and_failed_empty_save(tmp_path):
    path = tmp_path / "capture.bvh"
    path.write_text("existing")
    recording = create_recording(10)
    with pytest.raises(ValueError):
        recording.save(path)
    assert path.read_text() == "existing"
    neutral = {n: np.eye(3) for n in SEGMENT_NAMES}
    recording.append(100., neutral)
    recording.append(100., neutral)  # Duplicate timestamp is ignored.
    recording.append(100.35, neutral)
    assert recording.save(path) == 4
    joints, frames, dt = read_bvh(path)
    assert dt == pytest.approx(.1)
    assert len(frames) == 4
    np.testing.assert_allclose(frames, np.tile(frames[0], (4, 1)))
