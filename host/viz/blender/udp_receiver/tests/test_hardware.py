"""Exercise the Blender driver against a real ESP32 UDP controller.

Usage:
    blender --background Human_spine_UDP.blend --python-exit-code 1 \
        --python tests/test_hardware.py -- udp_mocap.py DEVICE_IP 4210 5
"""

import math
import os
import re
import sys
import time

import bpy


def arguments():
    if "--" not in sys.argv:
        raise SystemExit("expected driver.py device_ip device_port duration_s after --")
    values = sys.argv[sys.argv.index("--") + 1 :]
    if len(values) != 4:
        raise SystemExit("expected exactly four hardware-test arguments after --")
    return os.path.abspath(values[0]), values[1], int(values[2]), float(values[3])


driver_path, device_ip, device_port, duration_s = arguments()
if duration_s <= 0.0:
    raise ValueError("duration_s must be positive")

with open(driver_path, "r", encoding="utf-8") as source_file:
    driver_source = source_file.read()

test_source, replacements = re.subn(
    r"(?m)^AUTO_START = True$",
    "AUTO_START = False",
    driver_source,
)
if replacements != 1:
    raise RuntimeError("could not disable AUTO_START in the driver source")

runtime = {"__file__": driver_path, "__name__": "udp_mocap_hardware_test"}
exec(compile(test_source, driver_path, "exec"), runtime)
runtime["DEVICE_IP"] = device_ip
runtime["DEVICE_PORT"] = device_port
runtime["TAG_VIEWPORT_REDRAW"] = False
runtime["PRINT_DEVICE_MESSAGES"] = True

try:
    runtime["start_udp_mocap"]()
    deadline = time.monotonic() + duration_s
    while time.monotonic() < deadline:
        runtime["_animation_step"]()
        time.sleep(0.02)

    status = runtime["print_status"]()
    expected_ids = runtime["_EXPECTED_SENSOR_IDS"]
    counts = dict(runtime["_sensor_counts"])
    missing = [sensor_id for sensor_id in expected_ids if counts.get(sensor_id, 0) == 0]

    if runtime["_sample_frames"] < 3:
        raise AssertionError("fewer than three complete sensor frames received")
    if missing:
        raise AssertionError("no samples received for canonical IDs %s" % missing)
    if runtime["_neutral_pending"]:
        raise AssertionError("neutral pose was not captured")
    if len(runtime["_neutral_orientation"]) != len(expected_ids):
        raise AssertionError("neutral orientation is incomplete")
    if runtime["_apply_errors"] != 0:
        raise AssertionError("pose application errors: %s" % runtime["_apply_errors"])

    rig = bpy.data.objects[runtime["RIG_OBJECT"]]
    for bone_name in runtime["SENSOR_TO_BONE"].values():
        quaternion = rig.pose.bones[bone_name].rotation_quaternion
        values = (quaternion.w, quaternion.x, quaternion.y, quaternion.z)
        if not all(math.isfinite(value) for value in values):
            raise AssertionError("non-finite pose on bone %s" % bone_name)

    print(
        "[test_hardware] PASS ip=%s port=%s id_mode=%s sample_frames=%s q=%s"
        % (
            device_ip,
            device_port,
            runtime["_active_sensor_id_mode"],
            runtime["_sample_frames"],
            runtime["_quaternion_packets"],
        )
    )
    print("[test_hardware]", status)
finally:
    runtime["stop_udp_mocap"]()
