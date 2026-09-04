import math
import unittest

from capture_gravity import (
    FACE_ORDER,
    SensorRecord,
    analyze,
    calculate_axis_calibration,
    parse_frame,
    quaternion_gravity,
    summarize_face,
)


def make_record(
    face,
    accel,
    quaternion=(1.0, 0.0, 0.0, 0.0),
    sensor_id=0,
    index=0,
):
    return SensorRecord(
        host_time_s=float(index),
        device_ms=index * 50,
        sequence=index,
        face=face,
        sensor_id=sensor_id,
        ax=int(accel[0]),
        ay=int(accel[1]),
        az=int(accel[2]),
        gx=0,
        gy=0,
        gz=0,
        qw=quaternion[0],
        qx=quaternion[1],
        qy=quaternion[2],
        qz=quaternion[3],
    )


class FrameParserTests(unittest.TestCase):
    def test_parses_sensor_row(self):
        payload = (
            b"GRAVITY_FRAME 42 1234 1\n"
            b"S 3 10 -20 16384 1 2 3 1.0 0.0 0.0 0.0\n"
        )
        records = parse_frame(payload, 5.0, "+Z")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].sequence, 42)
        self.assertEqual(records[0].sensor_id, 3)
        self.assertEqual((records[0].ax, records[0].ay, records[0].az), (10, -20, 16384))

    def test_ignores_command_reply(self):
        self.assertEqual(parse_frame(b"ACK START\n", 1.0, "+Z"), [])

    def test_rejects_truncated_frame(self):
        with self.assertRaisesRegex(RuntimeError, "обещает 2"):
            parse_frame(
                b"GRAVITY_FRAME 1 2 2\nS 0 0 0 1 0 0 0 1 0 0 0\n",
                1.0,
                "+Z",
            )


class QuaternionTests(unittest.TestCase):
    def test_identity_gravity_is_positive_z(self):
        gravity = quaternion_gravity((1.0, 0.0, 0.0, 0.0))
        self.assertEqual(gravity, (0.0, 0.0, 1.0))

    def test_half_turn_about_x_gravity_is_negative_z(self):
        gravity = quaternion_gravity((0.0, 1.0, 0.0, 0.0))
        self.assertAlmostEqual(gravity[0], 0.0)
        self.assertAlmostEqual(gravity[1], 0.0)
        self.assertAlmostEqual(gravity[2], -1.0)


class AnalysisTests(unittest.TestCase):
    def test_ideal_positive_z_face_passes(self):
        records = [
            make_record("+Z", (0, 0, 16384), index=index)
            for index in range(20)
        ]
        summary = summarize_face(records)
        self.assertTrue(summary["passed"])
        self.assertAlmostEqual(summary["face_error_deg"], 0.0)
        self.assertAlmostEqual(summary["dmp_accel_mismatch_deg"], 0.0)

    def test_yaw_change_does_not_look_like_gravity_drift(self):
        records = []
        for index in range(20):
            yaw = math.radians(60.0 * index / 19.0)
            quaternion = (
                math.cos(yaw / 2.0),
                0.0,
                0.0,
                math.sin(yaw / 2.0),
            )
            records.append(
                make_record("+Z", (0, 0, 16384), quaternion, index=index)
            )
        summary = summarize_face(records)
        self.assertGreater(summary["quaternion_drift_deg"], 40.0)
        self.assertAlmostEqual(summary["dmp_gravity_drift_deg"], 0.0)
        self.assertTrue(summary["passed"])

    def test_six_face_bias_and_scale_estimate(self):
        bias = {"x": 100.0, "y": -50.0, "z": 25.0}
        scale = {"x": 16000.0, "y": 16500.0, "z": 16300.0}
        means = {
            "+X": (bias["x"] + scale["x"], bias["y"], bias["z"]),
            "-X": (bias["x"] - scale["x"], bias["y"], bias["z"]),
            "+Y": (bias["x"], bias["y"] + scale["y"], bias["z"]),
            "-Y": (bias["x"], bias["y"] - scale["y"], bias["z"]),
            "+Z": (bias["x"], bias["y"], bias["z"] + scale["z"]),
            "-Z": (bias["x"], bias["y"], bias["z"] - scale["z"]),
        }
        face_summaries = [
            {
                "sensor_id": 2,
                "face": face,
                "mean_accel_counts": {
                    "x": means[face][0],
                    "y": means[face][1],
                    "z": means[face][2],
                },
            }
            for face in FACE_ORDER
        ]
        calibration = calculate_axis_calibration(face_summaries)[0]["axes"]
        for axis in ("x", "y", "z"):
            self.assertAlmostEqual(calibration[axis]["bias_counts"], bias[axis])
            self.assertAlmostEqual(calibration[axis]["scale_counts_per_g"], scale[axis])

    def test_full_ideal_six_face_analysis_is_complete(self):
        quaternions = {
            "+Z": (1.0, 0.0, 0.0, 0.0),
            "-Z": (0.0, 1.0, 0.0, 0.0),
            "+X": (math.sqrt(0.5), 0.0, -math.sqrt(0.5), 0.0),
            "-X": (math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0),
            "+Y": (math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0),
            "-Y": (math.sqrt(0.5), -math.sqrt(0.5), 0.0, 0.0),
        }
        accel = {
            face: tuple(int(component * 16384) for component in {
                "+X": (1, 0, 0),
                "-X": (-1, 0, 0),
                "+Y": (0, 1, 0),
                "-Y": (0, -1, 0),
                "+Z": (0, 0, 1),
                "-Z": (0, 0, -1),
            }[face])
            for face in FACE_ORDER
        }
        records = [
            make_record(face, accel[face], quaternions[face], index=index)
            for face in FACE_ORDER
            for index in range(20)
        ]
        result = analyze(records)
        self.assertTrue(result["complete"])
        self.assertTrue(result["passed"])


if __name__ == "__main__":
    unittest.main()
