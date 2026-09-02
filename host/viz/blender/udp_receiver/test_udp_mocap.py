"""Headless integration tests for Human_spine_UDP.blend.

Usage:
    blender --background Human_spine_UDP.blend --python test_udp_mocap.py -- \
        udp_mocap.py
"""

import math
import os
import re
import socket
import sys
import time

import bpy
from mathutils import Quaternion


EXPECTED_OBJECTS = {"Camera", "Human_Rig", "Light"}
EXPECTED_TEXTS = {"UDP_MOCAP_STATUS", "udp_mocap.py"}
EXPECTED_BONES = {
    "spine": None,
    "chest": "spine",
    "head": "chest",
    "clavicle.L": "chest",
    "shoulder.L": "clavicle.L",
    "forearm.L": "shoulder.L",
    "clavicle.R": "chest",
    "shoulder.R": "clavicle.R",
    "forearm.R": "shoulder.R",
}
SENSOR_IDS = (0, 1, 2, 6, 7)
IDENTITY_VALUES = (1.0, 0.0, 0.0, 0.0)
TOLERANCE_RAD = 1.0e-5
checks = 0


def check(condition, message):
    global checks
    if not condition:
        raise AssertionError(message)
    checks += 1


def expect_value_error(callback, message):
    try:
        callback()
    except ValueError:
        check(True, message)
    else:
        raise AssertionError(message)


def quaternion_distance(left, right):
    dot = min(1.0, max(-1.0, abs(left.dot(right))))
    return 2.0 * math.acos(dot)


def quaternion_values(quaternion):
    return (quaternion.w, quaternion.x, quaternion.y, quaternion.z)


def make_frame(sequence, values_by_sensor, declared_count=None):
    # The current firmware may report zero in the header count, so the driver
    # intentionally trusts the actual Q lines.
    if declared_count is None:
        declared_count = len(values_by_sensor)
    lines = ["FRAME %d %d %d" % (sequence, sequence * 100, declared_count)]
    for sensor_id in sorted(values_by_sensor):
        values = values_by_sensor[sensor_id]
        lines.append(
            "Q %d %.9f %.9f %.9f %.9f"
            % (sensor_id, values[0], values[1], values[2], values[3])
        )
    return ("\n".join(lines) + "\n").encode("ascii")


def driver_path_from_arguments():
    if "--" not in sys.argv:
        raise SystemExit("expected udp_mocap.py after --")
    values = sys.argv[sys.argv.index("--") + 1 :]
    if len(values) != 1:
        raise SystemExit("expected exactly one udp_mocap.py path after --")
    return os.path.abspath(values[0])


driver_path = driver_path_from_arguments()
with open(driver_path, "r", encoding="utf-8") as source_file:
    driver_source = source_file.read()

check(set(bpy.data.objects.keys()) == EXPECTED_OBJECTS, "clean object set")
check(set(bpy.data.texts.keys()) == EXPECTED_TEXTS, "clean text block set")
check(
    bpy.data.texts["udp_mocap.py"].as_string().rstrip("\r\n")
    == driver_source.rstrip("\r\n"),
    "embedded driver matches external source",
)
check(bpy.context.scene.get("neuromorph_udp_mocap") is True, "scene marker")
check(bpy.context.scene.get("udp_mocap_build_version") == 3, "build version")

rig = bpy.data.objects.get("Human_Rig")
check(rig is not None and rig.type == "ARMATURE", "Human_Rig armature")
actual_bones = {
    bone.name: bone.parent.name if bone.parent else None for bone in rig.data.bones
}
check(actual_bones == EXPECTED_BONES, "rig bone hierarchy")
check("Bone" not in actual_bones, "unused default bone removed")

# Execute the real external driver with networking disabled, preserving every
# other setting exactly as shipped.
test_source, replacements = re.subn(
    r"(?m)^AUTO_START = True$",
    "AUTO_START = False",
    driver_source,
)
check(replacements == 1, "AUTO_START test override")
runtime = {"__file__": driver_path, "__name__": "udp_mocap_headless_test"}
exec(compile(test_source, driver_path, "exec"), runtime)
runtime["SMOOTH_ALPHA"] = 1.0
runtime["TAG_VIEWPORT_REDRAW"] = False
runtime["_prepare_rig"]()
runtime["reset_pose"]()

check(runtime["_EXPECTED_SENSOR_IDS"] == SENSOR_IDS, "sensor ID set")
check(
    runtime["SENSOR_TO_BONE"]
    == {
        0: "shoulder.L",
        1: "forearm.L",
        2: "spine",
        6: "forearm.R",
        7: "shoulder.R",
    },
    "sensor-to-bone map",
)
expect_value_error(
    lambda: runtime["_axis_matrix"](("+X", "+X", "+Z")),
    "duplicate source axis rejected",
)
expect_value_error(
    lambda: runtime["_normalized_quaternion"]((0.0, 0.0, 0.0, 0.0)),
    "zero quaternion rejected",
)
reflected = runtime["_axis_matrix"](("+X", "+Z", "-Y"))
check(abs(abs(reflected.determinant()) - 1.0) < 1.0e-6, "reflected axis map")

neutral_values = {sensor_id: IDENTITY_VALUES for sensor_id in SENSOR_IDS}
runtime["calibrate_neutral_pose"]()
runtime["_handle_datagram"](make_frame(1, neutral_values, declared_count=0))
neutral_samples = runtime["_snapshot_samples"]()
check(set(neutral_samples) == set(SENSOR_IDS), "all neutral samples parsed")
check(
    len({sample[5] for sample in neutral_samples.values()}) == 1,
    "one atomic generation per UDP frame",
)
check(runtime["_sample_frames"] == 1, "sample frame counter")
check(runtime["_quaternion_packets"] == 5, "quaternion counter")
check(runtime["_last_frame_header"] == "FRAME 1 100 0", "frame header")

mixed_samples = dict(neutral_samples)
mixed = list(mixed_samples[7])
mixed[5] += 1
mixed_samples[7] = tuple(mixed)
check(
    runtime["_try_capture_neutral"](mixed_samples) is False,
    "mixed frame neutral pose rejected",
)
check(runtime["_neutral_pending"] is True, "neutral remains pending")
check(
    runtime["_try_capture_neutral"](neutral_samples) is True,
    "coherent neutral pose captured",
)
check(runtime["_neutral_pending"] is False, "neutral completed")
check(runtime["_active_sensor_id_mode"] == "tca_channel", "TCA ID mode detected")

# The controller currently deployed at 192.168.1.117 reports IDs 1..5 and
# includes the physical TCA channel separately in STATUS. Verify that this
# legacy/sequential frame is converted to the same canonical channel IDs.
runtime["_active_sensor_id_mode"] = None
with runtime["_sample_lock"]:
    runtime["_latest_samples"].clear()
sequential_values = {
    raw_sensor_id: IDENTITY_VALUES for raw_sensor_id in (1, 2, 3, 4, 5)
}
runtime["_handle_datagram"](make_frame(11, sequential_values, declared_count=0))
check(
    runtime["_active_sensor_id_mode"] == "sequential",
    "sequential ID mode detected",
)
check(
    set(runtime["_snapshot_samples"]()) == set(SENSOR_IDS),
    "sequential IDs mapped to canonical channels",
)
runtime["set_sensor_id_mode"]("auto")
check(runtime["SENSOR_ID_MODE"] == "auto", "automatic ID mode restored")

# A common global rotation should move only the torso basis. Child sensor
# deltas cancel against their parent sensor and therefore remain local identity.
global_rotation = Quaternion((0.0, 0.0, 1.0), math.radians(30.0))
global_values = {
    sensor_id: quaternion_values(global_rotation) for sensor_id in SENSOR_IDS
}
runtime["_handle_datagram"](make_frame(2, global_values))
check(
    runtime["_update_sensor_deltas"](runtime["_snapshot_samples"]()) is True,
    "new global frame accepted",
)
spine_target = runtime["_target_basis_quaternion"](2)
check(
    quaternion_distance(spine_target, Quaternion((1.0, 0.0, 0.0, 0.0)))
    > math.radians(20.0),
    "torso receives global rotation",
)
for sensor_id in (0, 1, 6, 7):
    check(
        quaternion_distance(
            runtime["_target_basis_quaternion"](sensor_id),
            Quaternion((1.0, 0.0, 0.0, 0.0)),
        )
        < TOLERANCE_RAD,
        "sensor %d cancels parent global rotation" % sensor_id,
    )
check(runtime["_apply_pose"]() is True, "global pose applied")
check(
    quaternion_distance(rig.pose.bones["spine"].rotation_quaternion, spine_target)
    < TOLERANCE_RAD,
    "spine pose basis written",
)

# Isolated left-forearm motion must not leak into the upper arm or right side.
forearm_rotation = Quaternion((1.0, 0.0, 0.0), math.radians(40.0))
isolated_values = dict(neutral_values)
isolated_values[1] = quaternion_values(forearm_rotation)
runtime["_handle_datagram"](make_frame(3, isolated_values))
check(
    runtime["_update_sensor_deltas"](runtime["_snapshot_samples"]()) is True,
    "isolated forearm frame accepted",
)
left_forearm_target = runtime["_target_basis_quaternion"](1)
check(
    quaternion_distance(
        left_forearm_target, Quaternion((1.0, 0.0, 0.0, 0.0))
    )
    > math.radians(30.0),
    "left forearm receives isolated rotation",
)
for sensor_id in (0, 2, 6, 7):
    check(
        quaternion_distance(
            runtime["_target_basis_quaternion"](sensor_id),
            Quaternion((1.0, 0.0, 0.0, 0.0)),
        )
        < TOLERANCE_RAD,
        "sensor %d unaffected by left forearm" % sensor_id,
    )
runtime["_apply_pose"]()
check(
    quaternion_distance(
        rig.pose.bones["forearm.L"].rotation_quaternion, left_forearm_target
    )
    < TOLERANCE_RAD,
    "left forearm pose basis written",
)

invalid_before = runtime["_invalid_lines"]
frames_before = runtime["_sample_frames"]
runtime["_handle_datagram"](
    b"FRAME 4 400 0\nQ 99 1 0 0 0\nQ 0 0 0 0 0\nGARBAGE\n"
)
check(runtime["_invalid_lines"] == invalid_before + 3, "invalid lines counted")
check(runtime["_sample_frames"] == frames_before, "invalid frame not published")


class FakeSocket:
    def __init__(self):
        self.sent = []

    def send(self, payload):
        self.sent.append(payload)
        return len(payload)


fake_socket = FakeSocket()
runtime["_sock"] = fake_socket
runtime["set_stream_rate"](23)
check(runtime["STREAM_RATE_HZ"] == 23, "stream rate saved")
check(fake_socket.sent == [b"SET_RATE 23\n"], "stream rate command sent")
runtime["_sock"] = None

# Exercise the actual socket, reader thread, timer registration, command order,
# frame reception and shutdown against a local UDP peer.
server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server.bind(("127.0.0.1", 0))
server.settimeout(2.0)
try:
    runtime["DEVICE_IP"] = "127.0.0.1"
    runtime["DEVICE_PORT"] = server.getsockname()[1]
    runtime["LOCAL_BIND_IP"] = "127.0.0.1"
    runtime["LOCAL_BIND_PORT"] = 0
    runtime["STREAM_RATE_HZ"] = 10
    runtime["AUTO_CALIBRATE_NEUTRAL_ON_START"] = False
    runtime["start_udp_mocap"]()

    received_commands = []
    client_addresses = []
    for _index in range(4):
        payload, client_address = server.recvfrom(1024)
        received_commands.append(payload)
        client_addresses.append(client_address)
    check(
        received_commands
        == [b"HELLO\n", b"STATUS\n", b"SET_RATE 10\n", b"START\n"],
        "startup UDP command order",
    )
    check(len(set(client_addresses)) == 1, "one registered UDP client")
    check(bpy.app.timers.is_registered(runtime["_animation_step"]), "timer registered")

    server.sendto(make_frame(10, neutral_values), client_addresses[0])
    deadline = time.monotonic() + 2.0
    while runtime["_sample_frames"] < 1 and time.monotonic() < deadline:
        time.sleep(0.01)
    check(runtime["_sample_frames"] == 1, "reader thread received UDP frame")
    check(
        set(runtime["_snapshot_samples"]()) == set(SENSOR_IDS),
        "reader thread published full sensor frame",
    )

    runtime["stop_udp_mocap"](True)
    stop_payload, stop_client = server.recvfrom(1024)
    check(stop_payload == b"STOP\n", "STOP command sent")
    check(stop_client == client_addresses[0], "STOP uses registered UDP client")
    check(runtime["_running"] is False, "runtime stopped")
    check(runtime["_sock"] is None, "socket released")
    check(
        not bpy.app.timers.is_registered(runtime["_animation_step"]),
        "timer unregistered",
    )
    check(
        runtime["_reader_thread"] is None and runtime["_stop_event"] is None,
        "reader resources released",
    )
finally:
    if runtime["_running"] or runtime["_sock"] is not None:
        runtime["stop_udp_mocap"]()
    server.close()

runtime["reset_pose"]()

print("[test_udp_mocap] PASS: %d checks" % checks)
