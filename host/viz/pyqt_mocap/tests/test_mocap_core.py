"""Protocol, physical mapping, quaternion, and kinematics tests."""

from __future__ import annotations

import math
import unittest

import numpy as np

from pyqt_mocap.mocap_core import (
    DEFAULT_AXIS_MAPS,
    DEFAULT_SENSOR_MAPPING,
    IDENTITY_QUATERNION,
    RAW_SENSOR_ID_PROFILES,
    SEGMENT_NAMES,
    MotionCaptureModel,
    axis_map_matrix,
    compute_body_pose,
    quaternion_to_matrix,
)


SENSOR_IDS = (0, 1, 2, 6, 7)


def quaternion(axis: tuple[float, float, float], angle_deg: float) -> np.ndarray:
    vector = np.asarray(axis, dtype=float)
    vector /= np.linalg.norm(vector)
    half_angle = math.radians(angle_deg) * 0.5
    return np.concatenate(
        ([math.cos(half_angle)], vector * math.sin(half_angle))
    )


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


class CanonicalPhysicalMappingTests(unittest.TestCase):
    def test_mapping_matches_the_approved_body_mounting(self) -> None:
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

    def test_id_profiles_resolve_to_physical_tca_channels(self) -> None:
        self.assertEqual(
            RAW_SENSOR_ID_PROFILES["tca_channel"],
            {0: 0, 1: 1, 2: 2, 6: 6, 7: 7},
        )
        self.assertEqual(
            RAW_SENSOR_ID_PROFILES["sequential"],
            {1: 0, 2: 1, 3: 2, 4: 6, 5: 7},
        )


class ProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = MotionCaptureModel(smooth_alpha=1.0)
        self.model.set_enabled_segments(SEGMENT_NAMES)
        self.identity_samples = {
            sensor_id: IDENTITY_QUATERNION for sensor_id in SENSOR_IDS
        }

    def test_one_tca_datagram_captures_an_atomic_neutral_pose(self) -> None:
        self.model.request_neutral()
        result = self.model.handle_datagram(frame(1, self.identity_samples), 10.0)
        self.assertTrue(result.published)
        self.assertTrue(result.neutral_captured)
        self.assertEqual(self.model.active_sensor_id_mode, "tca_channel")
        self.assertEqual(set(self.model.neutral_orientation), set(SEGMENT_NAMES))

    def test_sequential_ids_are_canonicalized(self) -> None:
        samples = {
            raw_id: IDENTITY_QUATERNION for raw_id in (1, 2, 3, 4, 5)
        }
        self.model.handle_datagram(frame(1, samples), 10.0)
        self.assertEqual(self.model.active_sensor_id_mode, "sequential")
        self.assertEqual(set(self.model.latest_samples), set(SENSOR_IDS))

    def test_channel_6_moves_only_the_left_forearm(self) -> None:
        self.model.request_neutral()
        self.model.handle_datagram(frame(1, self.identity_samples), 10.0)
        moved = dict(self.identity_samples)
        moved[6] = quaternion((1.0, 0.0, 0.0), 40.0)
        self.model.handle_datagram(frame(2, moved), 10.1)
        orientations = self.model.orientations()
        self.assertGreater(
            quaternion_distance(
                orientations["forearm.L"], IDENTITY_QUATERNION
            ),
            math.radians(35.0),
        )
        for segment in ("spine", "shoulder.L", "shoulder.R", "forearm.R"):
            self.assertLess(
                quaternion_distance(orientations[segment], IDENTITY_QUATERNION),
                1.0e-6,
            )

    def test_control_reply_is_not_counted_as_invalid(self) -> None:
        result = self.model.handle_datagram(
            b"ACK SET_RATE rate_hz=25 frame_ms=40\n", 10.0
        )
        self.assertEqual(result.messages, ("ACK SET_RATE rate_hz=25 frame_ms=40",))
        self.assertEqual(result.invalid_lines, 0)

    def test_malformed_quaternion_is_rejected(self) -> None:
        result = self.model.handle_datagram(
            b"FRAME 1 100 1\nQ 6 0 0 0 0\n", 10.0
        )
        self.assertFalse(result.published)
        self.assertEqual(result.invalid_lines, 1)


class QuaternionAndKinematicsTests(unittest.TestCase):
    def test_axis_map_is_a_signed_permutation(self) -> None:
        matrix = axis_map_matrix(("+X", "+Z", "-Y"))
        self.assertAlmostEqual(abs(float(np.linalg.det(matrix))), 1.0)
        with self.assertRaises(ValueError):
            axis_map_matrix(("+X", "+X", "+Z"))

    def test_quaternion_matrix_rotates_a_vector(self) -> None:
        rotation = quaternion((0.0, 0.0, 1.0), 90.0)
        rotated = quaternion_to_matrix(rotation) @ np.array((1.0, 0.0, 0.0))
        np.testing.assert_allclose(rotated, (0.0, 1.0, 0.0), atol=1.0e-7)

    def test_left_forearm_motion_does_not_move_the_right_wrist(self) -> None:
        neutral = compute_body_pose({})
        moved = compute_body_pose(
            {"forearm.L": quaternion((1.0, 0.0, 0.0), 70.0)}
        )
        self.assertGreater(
            float(
                np.linalg.norm(
                    moved.tracked_segments["forearm.L"][1]
                    - neutral.tracked_segments["forearm.L"][1]
                )
            ),
            0.2,
        )
        np.testing.assert_allclose(
            moved.tracked_segments["forearm.R"][1],
            neutral.tracked_segments["forearm.R"][1],
            atol=1.0e-8,
        )

    def test_every_segment_has_an_axis_frame(self) -> None:
        pose = compute_body_pose({})
        self.assertEqual(set(pose.axis_frames), set(DEFAULT_AXIS_MAPS))
        for matrix in pose.axis_frames.values():
            np.testing.assert_allclose(
                matrix.T @ matrix, np.eye(3), atol=1.0e-8
            )

    def test_head_is_centered_on_the_top_of_the_neck(self) -> None:
        pose = compute_body_pose({})
        _chest, neck = pose.static_segments[0]
        np.testing.assert_allclose(pose.head_center, neck, atol=1.0e-8)

    def test_feet_point_forward(self) -> None:
        pose = compute_body_pose({})
        for ankle, toes in pose.static_segments[-2:]:
            np.testing.assert_allclose(
                toes - ankle,
                (0.0, 0.13, 0.0),
                atol=1.0e-8,
            )


if __name__ == "__main__":
    unittest.main()
