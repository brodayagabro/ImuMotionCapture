"""Headless tests for the PyQt viewer's protocol and kinematics."""

from __future__ import annotations

import math
import unittest

import numpy as np

from mocap_core import (
    DEFAULT_AXIS_MAPS,
    DEFAULT_SENSOR_MAPPING,
    IDENTITY_QUATERNION,
    MotionCaptureModel,
    axis_map_matrix,
    compute_body_pose,
    quaternion_to_matrix,
)


SENSOR_IDS = (0, 1, 2, 6, 7)


def quaternion(axis: tuple[float, float, float], angle_deg: float) -> np.ndarray:
    vector = np.asarray(axis, dtype=float)
    vector /= np.linalg.norm(vector)
    half = math.radians(angle_deg) * 0.5
    return np.concatenate(([math.cos(half)], vector * math.sin(half)))


def frame(sequence: int, samples: dict[int, np.ndarray]) -> bytes:
    lines = [f"FRAME {sequence} {sequence * 100} {len(samples)}"]
    for sensor_id, values in samples.items():
        lines.append(
            f"Q {sensor_id} {values[0]:.9f} {values[1]:.9f} "
            f"{values[2]:.9f} {values[3]:.9f}"
        )
    return ("\n".join(lines) + "\n").encode("ascii")


def quaternion_distance(left: np.ndarray, right: np.ndarray) -> float:
    dot = min(1.0, max(-1.0, abs(float(np.dot(left, right)))))
    return 2.0 * math.acos(dot)


class QuaternionTests(unittest.TestCase):
    def test_default_sensor_mapping_has_swapped_arm_groups(self) -> None:
        self.assertEqual(
            DEFAULT_SENSOR_MAPPING,
            {
                "shoulder.L": 7,
                "forearm.L": 6,
                "spine": 2,
                "forearm.R": 1,
                "shoulder.R": 0,
            },
        )

    def test_default_axis_maps_match_body_mounting(self) -> None:
        self.assertEqual(DEFAULT_AXIS_MAPS["spine"], ("+X", "-Z", "+Y"))
        self.assertEqual(DEFAULT_AXIS_MAPS["shoulder.L"], ("+X", "+Y", "+Z"))
        self.assertEqual(DEFAULT_AXIS_MAPS["forearm.L"], ("+X", "+Y", "+Z"))
        self.assertEqual(DEFAULT_AXIS_MAPS["shoulder.R"], ("-X", "-Y", "+Z"))
        self.assertEqual(DEFAULT_AXIS_MAPS["forearm.R"], ("-X", "-Y", "+Z"))

    def test_axis_map_is_blender_compatible_signed_permutation(self) -> None:
        reflected = axis_map_matrix(("+X", "+Z", "-Y"))
        self.assertAlmostEqual(abs(float(np.linalg.det(reflected))), 1.0)
        with self.assertRaises(ValueError):
            axis_map_matrix(("+X", "+X", "+Z"))

    def test_quaternion_matrix_rotates_vector(self) -> None:
        rotation = quaternion((0.0, 0.0, 1.0), 90.0)
        rotated = quaternion_to_matrix(rotation) @ np.array((1.0, 0.0, 0.0))
        np.testing.assert_allclose(rotated, (0.0, 1.0, 0.0), atol=1.0e-7)


class ProtocolAndCalibrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = MotionCaptureModel(smooth_alpha=1.0)
        self.identity_samples = {
            sensor_id: IDENTITY_QUATERNION for sensor_id in SENSOR_IDS
        }

    def test_tca_frame_captures_one_atomic_neutral_pose(self) -> None:
        self.model.request_neutral()
        result = self.model.handle_datagram(frame(1, self.identity_samples), 10.0)
        self.assertTrue(result.published)
        self.assertTrue(result.neutral_captured)
        self.assertEqual(self.model.active_sensor_id_mode, "tca_channel")
        self.assertEqual(set(self.model.neutral_orientation), set(DEFAULT_SENSOR_MAPPING))
        self.assertFalse(self.model.neutral_pending)

    def test_sequential_firmware_ids_are_canonicalized(self) -> None:
        sequential = {
            raw_id: IDENTITY_QUATERNION for raw_id in (1, 2, 3, 4, 5)
        }
        self.model.handle_datagram(frame(1, sequential), 10.0)
        self.assertEqual(self.model.active_sensor_id_mode, "sequential")
        self.assertEqual(set(self.model.latest_samples), set(SENSOR_IDS))

    def test_neutral_rejects_samples_from_mixed_frames(self) -> None:
        self.model.request_neutral()
        self.model.handle_datagram(
            frame(1, {sensor_id: IDENTITY_QUATERNION for sensor_id in (0, 1, 2, 6)}),
            10.0,
        )
        self.model.handle_datagram(frame(2, {7: IDENTITY_QUATERNION}), 10.05)
        self.assertTrue(self.model.neutral_pending)
        self.assertFalse(self.model.neutral_orientation)

    def test_isolated_forearm_motion_changes_only_left_forearm(self) -> None:
        self.model.request_neutral()
        self.model.handle_datagram(frame(1, self.identity_samples), 10.0)
        moved = dict(self.identity_samples)
        moved[DEFAULT_SENSOR_MAPPING["forearm.L"]] = quaternion(
            (1.0, 0.0, 0.0), 40.0
        )
        result = self.model.handle_datagram(frame(2, moved), 10.1)
        self.assertTrue(result.pose_changed)
        orientations = self.model.orientations()
        self.assertGreater(
            quaternion_distance(orientations["forearm.L"], IDENTITY_QUATERNION),
            math.radians(35.0),
        )
        for name in ("spine", "shoulder.L", "shoulder.R", "forearm.R"):
            self.assertLess(
                quaternion_distance(orientations[name], IDENTITY_QUATERNION),
                1.0e-6,
            )

    def test_control_reply_is_not_counted_as_invalid(self) -> None:
        result = self.model.handle_datagram(
            b"ACK SET_RATE rate_hz=25 frame_ms=40\n", 10.0
        )
        self.assertEqual(result.messages, ("ACK SET_RATE rate_hz=25 frame_ms=40",))
        self.assertEqual(result.invalid_lines, 0)


class MannequinTests(unittest.TestCase):
    def test_left_forearm_rotation_moves_only_left_wrist(self) -> None:
        neutral = compute_body_pose({})
        moved = compute_body_pose(
            {"forearm.L": quaternion((1.0, 0.0, 0.0), 70.0)}
        )
        neutral_left_wrist = neutral.tracked_segments["forearm.L"][1]
        moved_left_wrist = moved.tracked_segments["forearm.L"][1]
        neutral_right_wrist = neutral.tracked_segments["forearm.R"][1]
        moved_right_wrist = moved.tracked_segments["forearm.R"][1]
        self.assertGreater(
            float(np.linalg.norm(moved_left_wrist - neutral_left_wrist)), 0.2
        )
        np.testing.assert_allclose(
            moved_right_wrist, neutral_right_wrist, atol=1.0e-8
        )

    def test_all_five_blender_names_have_axes(self) -> None:
        pose = compute_body_pose({})
        self.assertEqual(set(pose.axis_frames), set(DEFAULT_AXIS_MAPS))
        for matrix in pose.axis_frames.values():
            np.testing.assert_allclose(matrix.T @ matrix, np.eye(3), atol=1.0e-8)


if __name__ == "__main__":
    unittest.main()
