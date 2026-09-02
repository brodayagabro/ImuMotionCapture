"""Real-time UDP motion capture driver for Human_spine_UDP.blend.

Run this text from Blender's Scripting workspace. The ESP32 protocol is:
    FRAME <sequence> <millis> <count>
    Q <sensor_id> <w> <x> <y> <z>

Canonical physical mapping uses TCA9548A channels:
    0 -> left upper arm
    1 -> left forearm
    2 -> back / torso
    6 -> right forearm
    7 -> right upper arm

The driver auto-detects both firmware ID formats:
    tca_channel: Q IDs 0, 1, 2, 6, 7
    sequential:  Q IDs 1, 2, 3, 4, 5

Public helpers after Run Script:
    start_udp_mocap()
    stop_udp_mocap()
    calibrate_neutral_pose()
    hardware_calibrate()
    set_stream_rate(10)
    set_sensor_id_mode("auto")
    request_device_status()
    print_status()
    set_axis_map(0, "+X", "+Z", "-Y")
"""

import math
import queue
import socket
import threading
import time

import bpy
from mathutils import Matrix, Quaternion


# =============================================================================
# USER CONFIGURATION
# =============================================================================

DEVICE_IP = "192.168.1.117"
DEVICE_PORT = 4210
LOCAL_BIND_IP = "0.0.0.0"
LOCAL_BIND_PORT = 0
RIG_OBJECT = "Human_Rig"

# The IDs are TCA9548A channel numbers emitted by the updated firmware.
SENSOR_TO_BONE = {
    0: "shoulder.L",
    1: "forearm.L",
    2: "spine",
    6: "forearm.R",
    7: "shoulder.R",
}

# "auto" supports both the current sequential firmware on the controller and
# firmware that sends physical TCA channel numbers. A mode can be forced for
# diagnostics with set_sensor_id_mode("sequential") or "tca_channel".
SENSOR_ID_MODE = "auto"
RAW_SENSOR_ID_PROFILES = {
    "tca_channel": {
        0: 0,
        1: 1,
        2: 2,
        6: 6,
        7: 7,
    },
    "sequential": {
        1: 0,
        2: 1,
        3: 2,
        4: 6,
        5: 7,
    },
}

# A bone uses its segment orientation relative to this parent segment.
# None means an absolute root segment relative to the neutral pose.
PARENT_SENSOR = {
    0: 2,     # left upper arm relative to back
    1: 0,     # left forearm relative to left upper arm
    2: None,  # back is the root
    6: 7,     # right forearm relative to right upper arm
    7: 2,     # right upper arm relative to back
}

# Signed coordinate permutation for each sensor.
# Each tuple says: Blender X/Y/Z takes data from which signed sensor axis.
# Examples:
#   ("+X", "+Y", "+Z")  identity
#   ("+X", "+Z", "-Y")  swap Y/Z and invert the new Z
#   ("-Y", "+X", "+Z")  rotate axes in the XY plane
AXIS_MAPS = {
    0: ("+X", "+Y", "+Z"),
    1: ("+X", "+Y", "+Z"),
    2: ("+X", "+Y", "+Z"),
    6: ("+X", "+Y", "+Z"),
    7: ("+X", "+Y", "+Z"),
}

# Enable this if the library quaternion describes world-to-sensor rather than
# sensor-to-world orientation. It can also be overridden for one sensor.
INVERT_ALL_INPUT_QUATERNIONS = False
INVERT_SENSOR_QUATERNION = {
    0: False,
    1: False,
    2: False,
    6: False,
    7: False,
}

AUTO_START = True
AUTO_START_DEVICE_STREAM = True
AUTO_CALIBRATE_NEUTRAL_ON_START = True
STREAM_RATE_HZ = 10

# 0 disables smoothing; 1 applies the newest sample directly.
SMOOTH_ALPHA = 0.65
BLENDER_UPDATE_HZ = 60
SOCKET_TIMEOUT_S = 0.20
DEVICE_COMMAND_GAP_S = 0.08
SENSOR_STALE_S = 0.75
NEUTRAL_MAX_SAMPLE_AGE_S = 0.40
NEUTRAL_MAX_SAMPLE_SKEW_S = 0.25
STATUS_EVERY_S = 2.0
PRINT_DEVICE_MESSAGES = True
TAG_VIEWPORT_REDRAW = True
STATUS_TEXT_BLOCK = "UDP_MOCAP_STATUS"


# =============================================================================
# RUNTIME STATE
# =============================================================================

RUNTIME_NAMESPACE_KEY = "neuromorph_udp_mocap_runtime"
_previous_runtime = bpy.app.driver_namespace.get(RUNTIME_NAMESPACE_KEY)
if _previous_runtime and callable(_previous_runtime.get("stop")):
    try:
        _previous_runtime["stop"]()
    except Exception as exc:
        print("[udp_mocap] previous runtime stop failed:", exc)

_IDENTITY = Quaternion((1.0, 0.0, 0.0, 0.0))
_EXPECTED_SENSOR_IDS = tuple(sorted(SENSOR_TO_BONE))
_ACCEPTED_RAW_SENSOR_IDS = frozenset(
    raw_sensor_id
    for profile in RAW_SENSOR_ID_PROFILES.values()
    for raw_sensor_id in profile
)
_TIMER_INTERVAL_S = 1.0 / max(1, BLENDER_UPDATE_HZ)

_sock = None
_reader_thread = None
_stop_event = None
_send_lock = threading.Lock()
_sample_lock = threading.Lock()
_control_events = queue.SimpleQueue()

_running = False
_reader_state = "idle"
_last_device_message = ""
_last_frame_header = ""
_session_started_s = 0.0
_last_status_s = 0.0

_udp_packets = 0
_sample_frames = 0
_quaternion_packets = 0
_invalid_lines = 0
_apply_frames = 0
_apply_errors = 0
_last_apply_us = 0
_max_apply_us = 0

# sensor_id -> (w, x, y, z, monotonic_timestamp, sequence)
_latest_samples = {}
_sensor_counts = {}
_sensor_last_s = {}
_sample_sequence = 0

_rig = None
_bones = {}
_rest_quaternion = {}
_axis_matrix_cache = {}
_last_basis = {}
_sensor_delta = {}
_applied_sample_sequence = {}
_neutral_orientation = {}
_neutral_pending = False
_hardware_calibration_pending = False
_active_sensor_id_mode = None


# =============================================================================
# AXES AND QUATERNIONS
# =============================================================================

def _axis_matrix(axis_spec):
    """Return an orthogonal matrix for a signed axis permutation."""
    if len(axis_spec) != 3:
        raise ValueError("axis map must contain exactly three axes")

    rows = []
    used = set()
    axis_index = {"X": 0, "Y": 1, "Z": 2}

    for token in axis_spec:
        token = str(token).strip().upper()
        if len(token) != 2 or token[0] not in "+-" or token[1] not in axis_index:
            raise ValueError(
                "axis values must look like '+X', '-Y', '+Z'; got %r" % token
            )
        source_index = axis_index[token[1]]
        if source_index in used:
            raise ValueError("each source axis must occur exactly once: %r" % (axis_spec,))
        used.add(source_index)
        row = [0.0, 0.0, 0.0]
        row[source_index] = 1.0 if token[0] == "+" else -1.0
        rows.append(row)

    result = Matrix(rows)
    if abs(abs(result.determinant()) - 1.0) > 1e-5:
        raise ValueError("axis map must be an orthogonal signed permutation")
    return result


def _rebuild_axis_cache():
    _axis_matrix_cache.clear()
    for sensor_id in _EXPECTED_SENSOR_IDS:
        spec = AXIS_MAPS.get(sensor_id, ("+X", "+Y", "+Z"))
        _axis_matrix_cache[sensor_id] = _axis_matrix(spec)


def _normalized_quaternion(values):
    q = Quaternion((values[0], values[1], values[2], values[3]))
    norm_sq = q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z
    if norm_sq < 0.25 or norm_sq > 2.25 or not math.isfinite(norm_sq):
        raise ValueError("invalid quaternion norm")
    q.normalize()
    return q


def _mapped_sensor_quaternion(sensor_id, values):
    q = _normalized_quaternion(values)
    if INVERT_ALL_INPUT_QUATERNIONS != INVERT_SENSOR_QUATERNION.get(sensor_id, False):
        q.invert()

    basis = _axis_matrix_cache[sensor_id]
    # Matrix conjugation supports both proper and reflected signed permutations.
    mapped = (basis @ q.to_matrix() @ basis.transposed()).to_quaternion()
    mapped.normalize()
    return mapped


def _same_hemisphere(reference, candidate):
    if reference.dot(candidate) < 0.0:
        candidate = Quaternion((-candidate.w, -candidate.x, -candidate.y, -candidate.z))
    return candidate


def _smooth_quaternion(bone_name, target):
    target.normalize()
    previous = _last_basis.get(bone_name)
    if previous is None or SMOOTH_ALPHA >= 1.0:
        _last_basis[bone_name] = target.copy()
        return target

    if SMOOTH_ALPHA <= 0.0:
        return previous.copy()

    target = _same_hemisphere(previous, target)
    result = previous.slerp(target, SMOOTH_ALPHA)
    result.normalize()
    _last_basis[bone_name] = result.copy()
    return result


# =============================================================================
# RIG
# =============================================================================

def _get_rig():
    obj = bpy.data.objects.get(RIG_OBJECT)
    if obj is None or obj.type != "ARMATURE":
        raise RuntimeError("armature object %r was not found" % RIG_OBJECT)
    return obj


def _prepare_rig():
    global _rig

    _rig = _get_rig()
    _bones.clear()
    _rest_quaternion.clear()

    for sensor_id, bone_name in SENSOR_TO_BONE.items():
        bone = _rig.pose.bones.get(bone_name)
        if bone is None:
            raise RuntimeError("pose bone %r for sensor %s was not found" % (bone_name, sensor_id))
        bone.rotation_mode = "QUATERNION"
        _bones[bone_name] = bone
        rest_q = bone.bone.matrix_local.to_quaternion()
        rest_q.normalize()
        _rest_quaternion[bone_name] = rest_q

    _rebuild_axis_cache()


def reset_pose():
    """Reset only the five controlled bones to their rest pose."""
    if _rig is None:
        _prepare_rig()

    for bone in _bones.values():
        bone.rotation_mode = "QUATERNION"
        bone.rotation_quaternion = _IDENTITY.copy()

    _last_basis.clear()
    _sensor_delta.clear()
    _applied_sample_sequence.clear()
    _rig.update_tag()
    _tag_redraw()
    print("[udp_mocap] controlled bones reset")


def _target_basis_quaternion(sensor_id):
    child_delta = _sensor_delta.get(sensor_id)
    if child_delta is None:
        return None

    parent_sensor_id = PARENT_SENSOR.get(sensor_id)
    if parent_sensor_id is None:
        relative_delta = child_delta.copy()
    else:
        parent_delta = _sensor_delta.get(parent_sensor_id)
        if parent_delta is None:
            return None
        relative_delta = parent_delta.inverted() @ child_delta

    relative_delta.normalize()
    bone_name = SENSOR_TO_BONE[sensor_id]
    rest_q = _rest_quaternion[bone_name]

    # Blender pose basis is in the bone rest coordinate frame:
    # basis = R_rest^-1 * R_relative_world * R_rest.
    basis_q = rest_q.inverted() @ relative_delta @ rest_q
    basis_q.normalize()
    return basis_q


def _apply_pose():
    if _rig is None:
        return False

    applied = False
    # Parent segments first: torso, upper arms, forearms.
    apply_order = (2, 0, 7, 1, 6)
    for sensor_id in apply_order:
        basis_q = _target_basis_quaternion(sensor_id)
        if basis_q is None:
            continue
        bone_name = SENSOR_TO_BONE[sensor_id]
        basis_q = _smooth_quaternion(bone_name, basis_q)
        _bones[bone_name].rotation_quaternion = basis_q
        applied = True

    if applied:
        _rig.update_tag()
        _tag_redraw()
    return applied


def _tag_redraw():
    if not TAG_VIEWPORT_REDRAW:
        return
    screen = getattr(bpy.context, "screen", None)
    if screen is None:
        return
    for area in screen.areas:
        if area.type == "VIEW_3D":
            area.tag_redraw()


# =============================================================================
# UDP
# =============================================================================

def _set_reader_state(text):
    global _reader_state
    _reader_state = text


def _send_command(command, log=True):
    sock = _sock
    if sock is None:
        return False

    payload = (command.strip() + "\n").encode("ascii")
    try:
        with _send_lock:
            sock.send(payload)
    except OSError as exc:
        _set_reader_state("send error: %s" % exc)
        return False

    if log:
        print("[udp_mocap] TX", command.strip())
    return True


def _configured_sensor_id_mode():
    mode = str(SENSOR_ID_MODE).strip().lower()
    if mode != "auto" and mode not in RAW_SENSOR_ID_PROFILES:
        raise ValueError(
            "SENSOR_ID_MODE must be 'auto', 'sequential', or 'tca_channel'"
        )
    return mode


def _canonicalize_frame_samples(raw_samples):
    """Map one raw firmware frame to canonical TCA channel IDs."""
    global _active_sensor_id_mode

    if not raw_samples:
        return []

    configured_mode = _configured_sensor_id_mode()
    if _active_sensor_id_mode is None:
        if configured_mode != "auto":
            detected_mode = configured_mode
        else:
            raw_ids = {sensor_id for sensor_id, _values in raw_samples}
            candidates = [
                mode
                for mode, profile in RAW_SENSOR_ID_PROFILES.items()
                if raw_ids.issubset(profile)
            ]
            # IDs 1 and 2 alone are ambiguous. Wait for a frame containing one
            # of the distinguishing IDs instead of mapping it incorrectly.
            if len(candidates) != 1:
                return []
            detected_mode = candidates[0]

        _active_sensor_id_mode = detected_mode
        print("[udp_mocap] sensor ID mode:", detected_mode)

    profile = RAW_SENSOR_ID_PROFILES[_active_sensor_id_mode]
    return [
        (profile[raw_sensor_id], values)
        for raw_sensor_id, values in raw_samples
        if raw_sensor_id in profile
    ]


def _parse_quaternion_line(line):
    parts = line.split()
    if len(parts) != 6 or parts[0].upper() not in ("Q", "QUAT"):
        return None
    try:
        sensor_id = int(parts[1])
        values = tuple(float(value) for value in parts[2:6])
        _normalized_quaternion(values)
    except (ValueError, OverflowError):
        return None
    if sensor_id not in _ACCEPTED_RAW_SENSOR_IDS:
        return None
    return sensor_id, values


def _handle_datagram(payload):
    global _udp_packets, _sample_frames, _quaternion_packets, _invalid_lines
    global _last_frame_header, _sample_sequence

    timestamp_s = time.monotonic()
    text = payload.decode("utf-8", errors="replace").strip("\x00\r\n")
    raw_samples = []
    control_messages = []
    frame_header = None
    invalid_lines = 0

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parsed = _parse_quaternion_line(line)
        if parsed is not None:
            raw_samples.append(parsed)
            continue

        if line.upper().startswith("FRAME "):
            frame_header = line
            continue

        if line.startswith(("ACK", "STATUS", "SENSOR", "PONG", "ERR")):
            control_messages.append(line)
            continue

        invalid_lines += 1

    samples = _canonicalize_frame_samples(raw_samples)

    # Publish a whole sensor frame while holding the lock once. This prevents
    # the 60 Hz Blender timer from observing a mixture of two 10 Hz UDP frames.
    with _sample_lock:
        _udp_packets += 1
        _invalid_lines += invalid_lines
        if samples:
            _sample_sequence += 1
            generation = _sample_sequence
            _sample_frames += 1
            for sensor_id, values in samples:
                _latest_samples[sensor_id] = (
                    values[0],
                    values[1],
                    values[2],
                    values[3],
                    timestamp_s,
                    generation,
                )
                _sensor_counts[sensor_id] = _sensor_counts.get(sensor_id, 0) + 1
                _sensor_last_s[sensor_id] = timestamp_s
                _quaternion_packets += 1

    if frame_header is not None:
        _last_frame_header = frame_header
    for message in control_messages:
        _control_events.put(message)


def _udp_reader_loop(stop_event, sock):
    _set_reader_state("reader running")
    while not stop_event.is_set():
        try:
            payload = sock.recv(4096)
        except socket.timeout:
            continue
        except OSError as exc:
            if not stop_event.is_set():
                _set_reader_state("receive error: %s" % exc)
            break
        if payload:
            _handle_datagram(payload)
    _set_reader_state("reader stopped")


def _connect_udp():
    global _sock

    endpoint = socket.getaddrinfo(
        DEVICE_IP, DEVICE_PORT, socket.AF_INET, socket.SOCK_DGRAM
    )[0][4]
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind((LOCAL_BIND_IP, LOCAL_BIND_PORT))
        sock.connect(endpoint)
        sock.settimeout(SOCKET_TIMEOUT_S)
    except Exception:
        sock.close()
        raise
    _sock = sock

    local = sock.getsockname()
    _set_reader_state(
        "connected %s:%s from local port %s" % (endpoint[0], endpoint[1], local[1])
    )
    print("[udp_mocap]", _reader_state)


def _process_control_events():
    global _last_device_message, _hardware_calibration_pending

    while True:
        try:
            message = _control_events.get_nowait()
        except queue.Empty:
            break

        _last_device_message = message
        if PRINT_DEVICE_MESSAGES:
            print("[device]", message)

        if message.startswith(("ACK CALIB DONE", "ACK CALIB_GYRO DONE")) and _hardware_calibration_pending:
            _hardware_calibration_pending = False
            calibrate_neutral_pose()


# =============================================================================
# CALIBRATION AND MAIN TIMER
# =============================================================================

def _snapshot_samples():
    with _sample_lock:
        return dict(_latest_samples)


def calibrate_neutral_pose():
    """Capture one coherent five-sensor N-pose as soon as fresh data arrives."""
    global _neutral_pending

    _neutral_orientation.clear()
    _sensor_delta.clear()
    _applied_sample_sequence.clear()
    _last_basis.clear()
    _neutral_pending = True
    reset_pose()
    print(
        "[udp_mocap] neutral calibration requested: stand upright, "
        "arms down, palms toward the body, and keep still"
    )


def _try_capture_neutral(samples):
    global _neutral_pending

    if not _neutral_pending:
        return False
    if any(sensor_id not in samples for sensor_id in _EXPECTED_SENSOR_IDS):
        return False

    now = time.monotonic()
    timestamps = [samples[sensor_id][4] for sensor_id in _EXPECTED_SENSOR_IDS]
    if now - min(timestamps) > NEUTRAL_MAX_SAMPLE_AGE_S:
        return False
    if max(timestamps) - min(timestamps) > NEUTRAL_MAX_SAMPLE_SKEW_S:
        return False
    generations = {samples[sensor_id][5] for sensor_id in _EXPECTED_SENSOR_IDS}
    if len(generations) != 1:
        return False

    captured = {}
    try:
        for sensor_id in _EXPECTED_SENSOR_IDS:
            captured[sensor_id] = _mapped_sensor_quaternion(
                sensor_id, samples[sensor_id][:4]
            )
    except (ValueError, KeyError) as exc:
        print("[udp_mocap] neutral capture rejected:", exc)
        return False

    _neutral_orientation.update(captured)
    _sensor_delta.update(
        (sensor_id, _IDENTITY.copy()) for sensor_id in _EXPECTED_SENSOR_IDS
    )
    _applied_sample_sequence.update(
        (sensor_id, samples[sensor_id][5]) for sensor_id in _EXPECTED_SENSOR_IDS
    )
    _neutral_pending = False
    print("[udp_mocap] neutral pose captured for sensors", _EXPECTED_SENSOR_IDS)
    return True


def _update_sensor_deltas(samples):
    if _neutral_pending or len(_neutral_orientation) != len(_EXPECTED_SENSOR_IDS):
        return False

    updated = False
    for sensor_id, sample in samples.items():
        if sensor_id not in SENSOR_TO_BONE:
            continue
        sequence = sample[5]
        if _applied_sample_sequence.get(sensor_id) == sequence:
            continue

        try:
            current = _mapped_sensor_quaternion(sensor_id, sample[:4])
        except (ValueError, KeyError):
            continue

        zero = _neutral_orientation.get(sensor_id)
        if zero is None:
            continue

        # Global segment delta from the neutral pose. A fixed mounting rotation
        # cancels in current * zero^-1 as long as the sensor cannot slip.
        delta = current @ zero.inverted()
        delta.normalize()
        _sensor_delta[sensor_id] = delta
        _applied_sample_sequence[sensor_id] = sequence
        updated = True

    return updated


def _animation_step():
    global _last_status_s, _apply_frames, _apply_errors
    global _last_apply_us, _max_apply_us

    if not _running:
        return None

    started_ns = time.perf_counter_ns()
    try:
        _process_control_events()
        samples = _snapshot_samples()
        _try_capture_neutral(samples)
        if _update_sensor_deltas(samples):
            _apply_pose()
    except Exception as exc:
        _apply_errors += 1
        print("[udp_mocap] timer error:", exc)

    _last_apply_us = int((time.perf_counter_ns() - started_ns) // 1000)
    _max_apply_us = max(_max_apply_us, _last_apply_us)
    _apply_frames += 1

    now = time.monotonic()
    if now - _last_status_s >= STATUS_EVERY_S:
        _last_status_s = now
        print_status()

    return _TIMER_INTERVAL_S if _running else None


# =============================================================================
# PUBLIC CONTROL FUNCTIONS
# =============================================================================

def start_udp_mocap():
    """Connect to the ESP32, register this UDP client, and start Blender updates."""
    global _running, _reader_thread, _stop_event
    global _session_started_s, _last_status_s
    global _udp_packets, _sample_frames, _quaternion_packets, _invalid_lines
    global _apply_frames, _apply_errors, _last_apply_us, _max_apply_us
    global _sample_sequence
    global _last_device_message, _last_frame_header
    global _neutral_pending, _hardware_calibration_pending
    global _active_sensor_id_mode

    if _running:
        print("[udp_mocap] already running")
        return

    _prepare_rig()
    reset_pose()

    with _sample_lock:
        _latest_samples.clear()
        _sensor_counts.clear()
        _sensor_last_s.clear()
        _udp_packets = 0
        _sample_frames = 0
        _quaternion_packets = 0
        _invalid_lines = 0
        _sample_sequence = 0

    while True:
        try:
            _control_events.get_nowait()
        except queue.Empty:
            break

    _neutral_orientation.clear()
    _neutral_pending = False
    _hardware_calibration_pending = False
    _last_device_message = ""
    _last_frame_header = ""
    _active_sensor_id_mode = None

    _apply_frames = 0
    _apply_errors = 0
    _last_apply_us = 0
    _max_apply_us = 0
    _session_started_s = time.monotonic()
    _last_status_s = _session_started_s

    try:
        _connect_udp()
    except Exception:
        _running = False
        raise

    _stop_event = threading.Event()
    _reader_thread = threading.Thread(
        target=_udp_reader_loop,
        args=(_stop_event, _sock),
        name="blender-udp-mocap",
        daemon=True,
    )
    _running = True
    _reader_thread.start()

    startup_commands = ["HELLO", "STATUS"]
    if STREAM_RATE_HZ:
        startup_commands.append("SET_RATE %d" % int(STREAM_RATE_HZ))
    if AUTO_START_DEVICE_STREAM:
        startup_commands.append("START")

    for index, command in enumerate(startup_commands):
        _send_command(command)
        if DEVICE_COMMAND_GAP_S > 0.0 and index + 1 < len(startup_commands):
            time.sleep(DEVICE_COMMAND_GAP_S)

    if AUTO_CALIBRATE_NEUTRAL_ON_START:
        calibrate_neutral_pose()

    if not bpy.app.timers.is_registered(_animation_step):
        bpy.app.timers.register(_animation_step, first_interval=_TIMER_INTERVAL_S)

    print("[udp_mocap] started")
    for sensor_id in _EXPECTED_SENSOR_IDS:
        parent_id = PARENT_SENSOR[sensor_id]
        print(
            "  sensor %s -> %s (parent sensor %s)"
            % (sensor_id, SENSOR_TO_BONE[sensor_id], parent_id)
        )


def stop_udp_mocap(stop_device_stream=False):
    """Stop Blender reception. Pass True to also send STOP to the ESP32."""
    global _running, _sock, _reader_thread, _stop_event

    if not _running and _sock is None:
        return

    _running = False
    if bpy.app.timers.is_registered(_animation_step):
        bpy.app.timers.unregister(_animation_step)

    if stop_device_stream:
        _send_command("STOP")

    if _stop_event is not None:
        _stop_event.set()

    sock = _sock
    _sock = None
    if sock is not None:
        try:
            sock.close()
        except OSError:
            pass

    if (
        _reader_thread is not None
        and _reader_thread.is_alive()
        and _reader_thread is not threading.current_thread()
    ):
        _reader_thread.join(timeout=0.5)

    _reader_thread = None
    _stop_event = None
    _set_reader_state("stopped")
    print("[udp_mocap] stopped")


def hardware_calibrate():
    """Calibrate gyro offsets while sensors are worn, then capture an N-pose."""
    global _hardware_calibration_pending

    if _send_command("CALIB_GYRO"):
        _hardware_calibration_pending = True
        print("[udp_mocap] CALIB_GYRO sent: keep the whole body completely still")
    else:
        print("[udp_mocap] not connected")


def full_hardware_calibrate():
    """Calibrate accel+gyro; detached modules must lie flat with local Z up."""
    global _hardware_calibration_pending

    if _send_command("CALIB"):
        _hardware_calibration_pending = True
        print(
            "[udp_mocap] CALIB sent: all detached modules must be flat, "
            "local Z up, and completely still"
        )
    else:
        print("[udp_mocap] not connected")


def request_device_status():
    if not _send_command("STATUS"):
        print("[udp_mocap] not connected")


def set_stream_rate(rate_hz):
    global STREAM_RATE_HZ

    rate_hz = int(rate_hz)
    if not 1 <= rate_hz <= 100:
        raise ValueError("rate must be in range 1..100 Hz")
    STREAM_RATE_HZ = rate_hz
    if not _send_command("SET_RATE %d" % rate_hz):
        print("[udp_mocap] rate saved; it will be sent on the next start")


def set_sensor_id_mode(mode):
    """Force an ID profile or restore automatic detection."""
    global SENSOR_ID_MODE, _active_sensor_id_mode

    mode = str(mode).strip().lower()
    if mode != "auto" and mode not in RAW_SENSOR_ID_PROFILES:
        raise ValueError("mode must be 'auto', 'sequential', or 'tca_channel'")

    SENSOR_ID_MODE = mode
    _active_sensor_id_mode = None
    with _sample_lock:
        _latest_samples.clear()
        _sensor_counts.clear()
        _sensor_last_s.clear()
    print("[udp_mocap] sensor ID mode configured:", mode)
    if _running:
        calibrate_neutral_pose()


def set_axis_map(sensor_id, blender_x, blender_y, blender_z):
    """Change a signed axis permutation and request a new neutral pose."""
    sensor_id = int(sensor_id)
    if sensor_id not in SENSOR_TO_BONE:
        raise ValueError("unknown sensor id %s" % sensor_id)

    spec = (str(blender_x).upper(), str(blender_y).upper(), str(blender_z).upper())
    matrix = _axis_matrix(spec)
    AXIS_MAPS[sensor_id] = spec
    _axis_matrix_cache[sensor_id] = matrix
    print("[udp_mocap] sensor %s axes = %s" % (sensor_id, spec))
    calibrate_neutral_pose()


def print_status():
    now = time.monotonic()
    id_mode = _active_sensor_id_mode or (
        "detecting" if _configured_sensor_id_mode() == "auto" else SENSOR_ID_MODE
    )
    with _sample_lock:
        packets = _udp_packets
        sample_frames = _sample_frames
        quaternions = _quaternion_packets
        invalid = _invalid_lines
        counts = dict(_sensor_counts)
        last = dict(_sensor_last_s)

    sensor_parts = []
    for sensor_id in _EXPECTED_SENSOR_IDS:
        age = None if sensor_id not in last else now - last[sensor_id]
        state = "none" if age is None else ("stale" if age > SENSOR_STALE_S else "live")
        age_text = "-" if age is None else "%.2fs" % age
        sensor_parts.append(
            "S%s:%s/%s/%s" % (sensor_id, counts.get(sensor_id, 0), age_text, state)
        )

    status = (
        "[udp_mocap] %s id_mode=%s udp=%s sample_frames=%s q=%s "
        "invalid=%s timer_frames=%s "
        "apply_us=%s/%s errors=%s neutral_pending=%s reader=%s %s"
        % (
            "running" if _running else "stopped",
            id_mode,
            packets,
            sample_frames,
            quaternions,
            invalid,
            _apply_frames,
            _last_apply_us,
            _max_apply_us,
            _apply_errors,
            _neutral_pending,
            _reader_state,
            " ".join(sensor_parts),
        )
    )
    print(status)
    _write_status(status)
    return status


def _write_status(status):
    block = bpy.data.texts.get(STATUS_TEXT_BLOCK)
    if block is None:
        block = bpy.data.texts.new(STATUS_TEXT_BLOCK)
    block.clear()
    block.write(
        status
        + "\n\nLast device message:\n"
        + (_last_device_message or "-")
        + "\n\nLast frame:\n"
        + (_last_frame_header or "-")
        + "\n\nCommands:\n"
        + "  calibrate_neutral_pose()\n"
        + "  hardware_calibrate()\n"
        + "  full_hardware_calibrate()\n"
        + "  request_device_status()\n"
        + "  set_stream_rate(10)\n"
        + "  set_sensor_id_mode('auto')\n"
        + "  set_axis_map(0, '+X', '+Z', '-Y')\n"
        + "  stop_udp_mocap()\n"
    )


bpy.app.driver_namespace[RUNTIME_NAMESPACE_KEY] = {
    "stop": stop_udp_mocap,
    "status": print_status,
    "neutral": calibrate_neutral_pose,
    "id_mode": set_sensor_id_mode,
}

if AUTO_START:
    try:
        start_udp_mocap()
    except Exception as exc:
        try:
            stop_udp_mocap()
        except Exception:
            pass
        message = "[udp_mocap] auto start failed: %s" % exc
        print(message)
        _write_status(message)
