"""Headless tests for mounting axes, guided calibration, and profiles."""

from __future__ import annotations

import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pyqt_mocap.calibration import (
    CapturedPose,
    calibrate_three_poses,
    load_profile,
    profile_document,
    save_profile,
)
from pyqt_mocap.mocap_core import (
    DEFAULT_AXIS_MAPS,
    DEFAULT_ENABLED_SEGMENTS,
    DEFAULT_SENSOR_MAPPING,
    IDENTITY_QUATERNION,
    SEGMENT_NAMES,
    MotionCaptureModel,
    axis_map_matrix,
    compute_body_pose,
    matrix_to_quaternion,
    quaternion_to_matrix,
    quaternion_from_rotation_vector,
)


def rotation_x(angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.array(((1, 0, 0), (0, cosine, -sine), (0, sine, cosine)))


def rotation_y(angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.array(((cosine, 0, sine), (0, 1, 0), (-sine, 0, cosine)))


def rotation_z(angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.array(((cosine, -sine, 0), (sine, cosine, 0), (0, 0, 1)))


def raw_quaternion_for(mapped_rotation: np.ndarray, segment: str) -> np.ndarray:
    basis = axis_map_matrix(DEFAULT_AXIS_MAPS[segment])
    return matrix_to_quaternion(basis.T @ mapped_rotation @ basis)


def captured_pose(
    name: str,
    rotations: dict[str, np.ndarray],
    started_s: float,
) -> CapturedPose:
    values = {
        segment: raw_quaternion_for(rotations[segment], segment)
        for segment in SEGMENT_NAMES
    }
    return CapturedPose(
        name=name,
        average={name: value.copy() for name, value in values.items()},
        first={name: value.copy() for name, value in values.items()},
        last={name: value.copy() for name, value in values.items()},
        sample_counts={name: 50 for name in SEGMENT_NAMES},
        started_s=started_s,
        ended_s=started_s + 5.0,
    )


class MountingAxisTests(unittest.TestCase):
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

    def test_disabled_spine_is_ignored_and_held_neutral(self) -> None:
        self.assertNotIn("spine", DEFAULT_ENABLED_SEGMENTS)
        model = MotionCaptureModel(
            DEFAULT_SENSOR_MAPPING, DEFAULT_AXIS_MAPS, "raw", smooth_alpha=1.0
        )
        model.request_neutral()
        payload = "FRAME 1 100 4\n" + "".join(
            f"Q {sensor_id} 1 0 0 0\n"
            for sensor_id in (0, 1, 6, 7)
        )
        result = model.handle_datagram(payload, 10.0)
        self.assertTrue(result.neutral_captured)
        self.assertEqual(set(model.latest_samples), {0, 1, 6, 7})
        self.assertNotIn(2, model.latest_samples)
        np.testing.assert_allclose(
            model.orientations()["spine"], IDENTITY_QUATERNION, atol=1e-8
        )

        model.set_enabled_segments(SEGMENT_NAMES)
        model.handle_datagram("FRAME 2 200 1\nQ 2 1 0 0 0\n", 11.0)
        self.assertIn(2, model.latest_samples)

    def test_requested_default_axis_maps(self) -> None:
        self.assertEqual(DEFAULT_AXIS_MAPS["shoulder.L"], ("+X", "+Y", "+Z"))
        self.assertEqual(DEFAULT_AXIS_MAPS["forearm.L"], ("+X", "+Y", "+Z"))
        self.assertEqual(DEFAULT_AXIS_MAPS["shoulder.R"], ("-X", "-Y", "+Z"))
        self.assertEqual(DEFAULT_AXIS_MAPS["forearm.R"], ("-X", "-Y", "+Z"))
        self.assertEqual(DEFAULT_AXIS_MAPS["spine"], ("+X", "+Z", "-Y"))

    def test_drawn_axes_match_mounting_when_arms_point_forward(self) -> None:
        forward = matrix_to_quaternion(rotation_x(math.pi / 2.0))
        pose = compute_body_pose(
            {
                "shoulder.L": forward,
                "forearm.L": forward,
                "shoulder.R": forward,
                "forearm.R": forward,
            }
        )
        np.testing.assert_allclose(pose.axis_frames["shoulder.L"], np.eye(3), atol=1e-8)
        np.testing.assert_allclose(pose.axis_frames["forearm.L"], np.eye(3), atol=1e-8)
        right_frame = np.diag((-1.0, -1.0, 1.0))
        np.testing.assert_allclose(pose.axis_frames["shoulder.R"], right_frame, atol=1e-8)
        np.testing.assert_allclose(pose.axis_frames["forearm.R"], right_frame, atol=1e-8)
        torso_frame = np.array(((1, 0, 0), (0, 0, 1), (0, -1, 0)))
        np.testing.assert_allclose(pose.axis_frames["spine"], torso_frame, atol=1e-8)


class GuidedCalibrationTests(unittest.TestCase):
    def setUp(self) -> None:
        identity = {name: np.eye(3) for name in SEGMENT_NAMES}
        t_pose = dict(identity)
        forward_pose = dict(identity)
        for name in ("shoulder.L", "forearm.L"):
            t_pose[name] = rotation_y(math.pi / 2.0)
            forward_pose[name] = rotation_x(math.pi / 2.0)
        for name in ("shoulder.R", "forearm.R"):
            t_pose[name] = rotation_y(-math.pi / 2.0)
            forward_pose[name] = rotation_x(math.pi / 2.0)
        self.captures = {
            "n_pose": captured_pose("n_pose", identity, 100.0),
            "t_pose": captured_pose("t_pose", t_pose, 110.0),
            "forward_pose": captured_pose("forward_pose", forward_pose, 120.0),
        }

    def test_three_poses_keep_known_mounting_and_have_zero_error(self) -> None:
        result = calibrate_three_poses(self.captures)
        self.assertEqual(result.axis_maps, DEFAULT_AXIS_MAPS)
        for score in result.scores_deg.values():
            self.assertLess(score, 1e-6)
        for rate in result.drift_rates_rad_s.values():
            np.testing.assert_allclose(rate, (0.0, 0.0, 0.0), atol=1e-10)

    def test_three_poses_align_a_rotated_arm_sensor_frame(self) -> None:
        segment = "shoulder.L"
        expected_alignment = rotation_z(math.radians(18.0))
        expected_rotations = {
            "n_pose": np.eye(3),
            "t_pose": rotation_y(math.pi / 2.0),
            "forward_pose": rotation_x(math.pi / 2.0),
        }
        observed_rotations = {}
        for pose_name, expected in expected_rotations.items():
            observed = expected_alignment.T @ expected @ expected_alignment
            observed_rotations[pose_name] = observed
            raw = raw_quaternion_for(observed, segment)
            capture = self.captures[pose_name]
            capture.average[segment] = raw.copy()
            capture.first[segment] = raw.copy()
            capture.last[segment] = raw.copy()

        result = calibrate_three_poses(self.captures)
        actual_alignment = quaternion_to_matrix(
            np.asarray(result.axis_alignment_quaternions[segment])
        )
        self.assertLess(result.scores_deg[segment], 1.0e-6)
        for pose_name in ("t_pose", "forward_pose"):
            corrected = (
                actual_alignment
                @ observed_rotations[pose_name]
                @ actual_alignment.T
            )
            np.testing.assert_allclose(
                corrected, expected_rotations[pose_name], atol=1.0e-8
            )

        self.assertEqual(
            result.axis_maps[segment], DEFAULT_AXIS_MAPS[segment]
        )
        live_expected = rotation_y(math.radians(32.0))
        live_observed = expected_alignment.T @ live_expected @ expected_alignment
        live_raw = raw_quaternion_for(live_observed, segment)
        neutral_raw = {
            name: self.captures["n_pose"].average[name]
            for name in SEGMENT_NAMES
        }
        model = MotionCaptureModel(
            DEFAULT_SENSOR_MAPPING,
            result.axis_maps,
            "raw",
            smooth_alpha=1.0,
        )
        model.set_guided_calibration(
            result.axis_maps,
            result.drift_rates_rad_s,
            neutral_raw,
            result.reference_s,
            result.axis_alignment_quaternions,
        )
        lines = ["FRAME 1 10000 5"]
        for name, sensor_id in DEFAULT_SENSOR_MAPPING.items():
            quaternion = (
                live_raw if name == segment else neutral_raw[name]
            )
            lines.append(
                f"Q {sensor_id} {quaternion[0]} {quaternion[1]} "
                f"{quaternion[2]} {quaternion[3]}"
            )
        model.handle_datagram("\n".join(lines), result.reference_s)
        np.testing.assert_allclose(
            quaternion_to_matrix(model.orientations()[segment]),
            live_expected,
            atol=1.0e-8,
        )

    def test_profile_round_trip(self) -> None:
        result = calibrate_three_poses(self.captures)
        document = profile_document(
            {
                "device_ip": "192.168.1.117",
                "device_port": 4210,
                "stream_rate_hz": 10,
                "render_fps": 30,
                "sensor_id_mode": "auto",
                "sensor_mapping": DEFAULT_SENSOR_MAPPING,
            },
            result,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            save_profile(path, document)
            loaded = load_profile(path)
        self.assertEqual(loaded["schema"], document["schema"])
        self.assertEqual(
            loaded["calibration"]["axis_maps"],
            document["calibration"]["axis_maps"],
        )
        self.assertEqual(
            loaded["calibration"]["axis_alignment_quaternions_wxyz"],
            document["calibration"]["axis_alignment_quaternions_wxyz"],
        )

    def test_stationary_drift_is_removed_from_model(self) -> None:
        identity_maps = {name: ("+X", "+Y", "+Z") for name in SEGMENT_NAMES}
        rates = {name: (0.0, 0.0, 0.1) for name in SEGMENT_NAMES}
        neutral = {name: IDENTITY_QUATERNION for name in SEGMENT_NAMES}
        model = MotionCaptureModel(
            DEFAULT_SENSOR_MAPPING, identity_maps, "raw", smooth_alpha=1.0
        )
        model.set_guided_calibration(identity_maps, rates, neutral, 0.0)
        drifted = quaternion_from_rotation_vector((0.0, 0.0, 1.0))
        lines = ["FRAME 1 10000 5"]
        for sensor_id in DEFAULT_SENSOR_MAPPING.values():
            lines.append(
                f"Q {sensor_id} {drifted[0]} {drifted[1]} "
                f"{drifted[2]} {drifted[3]}"
            )
        model.handle_datagram("\n".join(lines), 10.0)
        for orientation in model.orientations().values():
            np.testing.assert_allclose(orientation, IDENTITY_QUATERNION, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
