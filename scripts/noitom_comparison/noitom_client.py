"""Lazy ctypes binding to the official versioned Mocap API C procedure tables.

ABI follows demo/demo-py/MocapApi/mocap_api/mocap_api.py. No Qt imports here.
Only this module knows about SDK handles, enums, native functions or libraries.
"""
import ctypes as C
from dataclasses import replace
import math
import queue
import threading
import time
import numpy as np
from pyqt_mocap.mocap_core import SEGMENT_NAMES, quaternion_from_rotation_vector
from .config import BVH_ROTATIONS, ComparisonConfig
from .noitom_adapter import NoitomAdapter, sdk_position_to_project, sdk_rotation_to_project
from .synchronization import JointTransform, NoitomFrame

H, U, I, F = C.c_uint64, C.c_uint32, C.c_int, C.c_float
PH, PU, PI, PF = C.POINTER(H), C.POINTER(U), C.POINTER(I), C.POINTER(F)
PS = C.POINTER(C.c_char_p)


class _SystemError(C.Structure):
    _fields_ = [("error", I), ("info", H)]


class _EventData(C.Union):
    _fields_ = [("reserved", H * 6), ("avatar", H), ("system_error", _SystemError)]


class _Event(C.Structure):
    _fields_ = [("size", U), ("event_type", I), ("timestamp", C.c_double), ("data", _EventData)]


class NoitomError(RuntimeError):
    pass


def _check(code, operation, allowed=(0,)):
    if code not in allowed:
        raise NoitomError(f"{operation}: Mocap API error {code}")


# Explicit slot positions from the versioned C header, not C++ vtables.
_TABLES = {
    "application": ("IMCPApplication_002", 12, {
        "create": (0, (PH,)), "destroy": (1, (H,)), "settings": (2, (H, H)),
        "render": (3, (H, H)), "open": (4, (H,)),
        "close": (8, (H,)), "poll": (11, (C.POINTER(_Event), PU, H)),
    }),
    "settings": ("IMCPSettings_001", 9, {
        "create": (0, (PH,)), "destroy": (1, (H,)), "udp": (2, (C.c_uint16, H)),
        "tcp": (3, (C.c_char_p, C.c_uint16, H)), "rotation": (4, (I, H)),
        "transformation": (5, (I, H)), "format": (6, (I, H)),
    }),
    "render": ("IMCPRenderSettings_001", 13, {
        "preset": (1, (I, PH)),
        "create": (0, (PH,)), "up": (2, (I, I, H)), "front": (4, (I, I, H)),
        "coord": (6, (I, H)), "rotation": (8, (I, H)),
        "unit": (10, (I, H)), "destroy": (12, (H,)),
        "get_up": (3, (PI, PI, H)), "get_front": (5, (PI, PI, H)),
        "get_coord": (7, (PI, H)), "get_rotation": (9, (PI, H)), "get_unit": (11, (PI, H)),
    }),
    "avatar": ("IMCPAvatar_003", 10, {
        "index": (0, (PU, H)), "root": (1, (PH, H)),
        "name": (4, (PS, H)), "posture_index": (7, (PU, H)),
    }),
    "joint": ("IMCPJoint_003", 9, {
        "name": (0, (PS, H)), "rotation": (1, (PF, PF, PF, PF, H)),
        "position": (3, (PF, PF, PF, H)), "default_position": (4, (PF, PF, PF, H)),
        "children": (5, (PH, PU, H)), "tag": (8, (PI, H)),
    }),
}


class NoitomClient:
    def __init__(self, config: ComparisonConfig):
        self.config = config
        self._library = None
        self._functions = {}
        self._app, self._settings, self._render = H(), H(), H()
        self._opened = False
        self.version = None
        self._selected_avatar = config.avatar_index
        self._events = queue.Queue()
        self.delivery_mode = "not_connected"
        self.sdk_render_settings = {}
        self.stamp = lambda: (time.monotonic_ns(), None)
        self.release_claim = lambda claim, error=None: None

    def _call(self, table, method, *args, allowed=(0,)):
        code = self._functions[table][method](*args)
        _check(code, f"{table}.{method}", allowed)
        return code

    def connect(self):
        if self._app.value:
            return
        path = self.config.library_path()
        if not path.is_file():
            raise NoitomError(f"Mocap API library not found: {path}. Use --sdk-library.")
        try:
            self._library = C.CDLL(str(path))  # C header specifies __cdecl on Windows.
            get_interface = self._library.MCPGetGenericInterface
            get_interface.argtypes = (C.c_char_p, C.POINTER(C.c_void_p))
            get_interface.restype = I
            for key, (version, size, signatures) in _TABLES.items():
                pointer = C.c_void_p()
                _check(get_interface(f"PROC_TABLE:{version}".encode(), C.byref(pointer)), version)
                if not pointer.value:
                    raise NoitomError(f"SDK returned null interface: {version}")
                table = C.cast(pointer, C.POINTER(C.c_void_p * size)).contents
                # These getters only copy already-polled SDK state. Keep the GIL
                # across each tiny getter: releasing it hundreds of times per
                # frame lets GUI/disk work starve acquisition. Blocking lifecycle
                # and polling operations still release it via CFUNCTYPE.
                prototype = C.PYFUNCTYPE if key in {"joint", "avatar"} else C.CFUNCTYPE
                self._functions[key] = {
                    name: prototype(I, *args)(table[slot])
                    for name, (slot, args) in signatures.items()
                }
            version = self._library.MCPGetMocapApiVersionString
            version.argtypes, version.restype = (), C.c_char_p
            self.version = version().decode("utf-8", "replace")
            self._call("application", "create", C.byref(self._app))
            self._call("settings", "create", C.byref(self._settings))
            # Built-in presets belong to the SDK and must not be destroyed.
            self._call("render", "preset", 0, C.byref(self._render))
            if self.config.transport == "udp":
                self._call("settings", "udp", self.config.port, self._settings)
            else:
                self._call("settings", "tcp", self.config.server.encode(), self.config.port, self._settings)
            self._call("settings", "rotation", BVH_ROTATIONS[self.config.bvh_rotation], self._settings)
            self._call("settings", "transformation", 1, self._settings)
            data_format = {"string": 0, "legacy": 1, "binary": 2}[self.config.bvh_format]
            self._call("settings", "format", data_format, self._settings)
            expected = {"up": (2, 1), "front": (1, 1), "coord": (0,), "rotation": (1,), "unit": (0,)}
            for name, values in expected.items():
                actual = [I() for _ in values]
                self._call("render", "get_" + name, *(C.byref(value) for value in actual), self._render)
                self.sdk_render_settings[name] = [value.value for value in actual]
                if tuple(self.sdk_render_settings[name]) != values:
                    raise NoitomError(f"Unsupported Default render setting {name}: {self.sdk_render_settings[name]}")
            self._call("application", "settings", self._settings, self._app)
            self._call("application", "render", self._render, self._app)
            self.delivery_mode = "poll_latest"
        except Exception:
            self.disconnect()
            raise

    def start(self):
        if not self._app.value:
            raise NoitomError("Connect Noitom first")
        if not self._opened:
            self._call("application", "open", self._app)
            self._opened = True

    def stop(self):
        if self._opened:
            self._call("application", "close", self._app)
            self._opened = False

    def disconnect(self):
        errors = []
        try:
            self.stop()
        except Exception as error:
            errors.append(str(error))
        for table, handle in (("application", self._app), ("settings", self._settings)):
            if handle.value:
                try:
                    self._call(table, "destroy", handle)
                except Exception as error:
                    errors.append(str(error))
                handle.value = 0
        self._opened = False
        self._render.value = 0
        self._selected_avatar = self.config.avatar_index
        if errors:
            raise NoitomError("; ".join(errors))

    def poll_events(self):
        """Query count before fetching, exactly as the working official demo does."""
        if self._events.empty() and self._opened:
            self._poll_native()
        try:
            item = self._events.get_nowait()
        except queue.Empty:
            return []
        if isinstance(item, Exception):
            raise item
        return [item]

    def _poll_native(self):
        # In the live SDK the count query also advances pending network data;
        # polling a fixed-size array without this query can return no frames.
        for _ in range(4):
            count = U()
            code = self._functions["application"]["poll"](None, C.byref(count), self._app)
            _check(code, "PollApplicationNextEvent(count)", (0, 1, 12))
            if code == 12 or count.value == 0:
                return
            capacity = count.value
            if capacity > 4096:
                raise NoitomError("SDK event buffer exceeded 4096 entries")
            events = (_Event * capacity)()
            for event in events:
                event.size = C.sizeof(_Event)
            count = U(capacity)
            code = self._functions["application"]["poll"](events, C.byref(count), self._app)
            stamps = [self.stamp() for _ in range(min(count.value, capacity))] if code in (0, 1) else []
            if code == 2:
                continue
            _check(code, "PollApplicationNextEvent", (0, 1, 12))
            for event, (host_ns, claim) in zip(events, stamps):
                self._queue_snapshot(event, host_ns, claim)
            return
        raise NoitomError("SDK event buffer keeps growing")

    def _queue_snapshot(self, event, host_ns, claim):
        try:
            frame = self._snapshot_event(event, host_ns)
            if frame is not None:
                self._events.put(replace(frame, capture_context=claim))
                claim = None  # Ownership passes to the acquisition consumer.
        except Exception as error:
            self.release_claim(claim, str(error))
            claim = None
            self._events.put(error)
        finally:
            self.release_claim(claim)

    def _snapshot_event(self, event, host_ns):
        if event.event_type == 768:
            raise NoitomError(f"SDK event error {event.data.system_error.error}, info {event.data.system_error.info}")
        if event.event_type != 256:
            return None
        handle = event.data.avatar
        index = U()
        self._call("avatar", "index", C.byref(index), handle)
        if self._selected_avatar is None:
            self._selected_avatar = index.value
        if index.value != self._selected_avatar:
            return None
        name, root, frame_index = C.c_char_p(), H(), U()
        self._call("avatar", "name", C.byref(name), handle)
        self._call("avatar", "root", C.byref(root), handle)
        code = self._call("avatar", "posture_index", C.byref(frame_index), handle, allowed=(0, 6, 12))
        joints = {}
        self._read_joint(root.value, None, joints, set())
        return NoitomFrame(host_ns, event.timestamp, frame_index.value if code == 0 else None,
                           joints, index.value, (name.value or b"").decode("utf-8", "replace"))

    def _vector(self, method, handle):
        values = [F(), F(), F()]
        code = self._call("joint", method, *(C.byref(value) for value in values), handle, allowed=(0, 6, 11, 12))
        return None if code else np.array([value.value for value in values])

    def _read_joint(self, handle, parent, joints, visited):
        if handle in visited or len(visited) >= 512:
            raise NoitomError("Invalid SDK joint hierarchy")
        visited.add(handle)
        raw_name, tag = C.c_char_p(), I()
        self._call("joint", "name", C.byref(raw_name), handle)
        self._call("joint", "tag", C.byref(tag), handle)
        name = (raw_name.value or b"").decode("utf-8", "replace")
        if not name or name in joints:
            raise NoitomError(f"Duplicate/empty SDK joint name: {name}")
        values = [F(), F(), F(), F()]
        self._call("joint", "rotation", *(C.byref(value) for value in values), handle)
        raw_rotation = np.array([value.value for value in values])
        raw_position = self._vector("position", handle)
        raw_default = self._vector("default_position", handle)
        joints[name] = JointTransform(
            None if raw_position is None else sdk_position_to_project(raw_position),
            sdk_rotation_to_project(raw_rotation), parent,
            None if raw_default is None else sdk_position_to_project(raw_default), tag.value,
            raw_position, raw_rotation, raw_default)
        count = U()
        self._call("joint", "children", None, C.byref(count), handle, allowed=(0, 2, 14))
        if not count.value:
            return
        if count.value > 512:
            raise NoitomError("Invalid SDK child count")
        children = (H * count.value)()
        self._call("joint", "children", children, C.byref(count), handle)
        for child in children[:count.value]:
            self._read_joint(child, name, joints, visited)


class FakeNoitomClient:
    """Deterministic synthetic BVH-style hierarchy; never loads a native library."""
    version = "synthetic-1"
    delivery_mode = "synthetic"

    def __init__(self, config=None, rate_hz=60):
        self.rate_hz, self.connected, self.running = rate_hz, False, False
        self.index, self.next_ns = 0, 0
        self.stamp = lambda: (time.monotonic_ns(), None)

    def connect(self):
        self.connected = True

    def disconnect(self):
        self.stop()
        self.connected = False

    def start(self):
        if not self.connected:
            raise NoitomError("Connect fake Noitom first")
        self.running, self.next_ns = True, time.monotonic_ns()

    def stop(self):
        self.running = False

    @staticmethod
    def frame(index, timestamp_ns):
        phase = math.sin(index / 35)
        joints = {}

        def add(name, parent, tag, offset, ry=0):
            joints[name] = JointTransform(np.asarray(offset, float),
                quaternion_from_rotation_vector((0, ry, 0)), parent, np.asarray(offset, float), tag)

        add("Hips", None, 0, (0, 0, .92))
        add("Spine2", "Hips", 9, (0, 0, .4))
        add("Neck", "Spine2", 10, (0, 0, .2))
        for side, sign, tags in (("Left", -1, (37, 38, 39)), ("Right", 1, (14, 15, 16))):
            add(side + "Arm", "Spine2", tags[0], (sign * .29, 0, .12), sign * math.pi / 2 + .7 * phase)
            add(side + "ForeArm", side + "Arm", tags[1], (sign * .43, 0, 0), .35 * phase)
            add(side + "Hand", side + "ForeArm", tags[2], (sign * .39, 0, 0))
        return NoitomFrame(timestamp_ns, index / 60, index, joints, 0, "SYNTHETIC")

    def poll_events(self):
        now = time.monotonic_ns()
        if not self.running or now < self.next_ns:
            return []
        self.next_ns = now + int(1e9 / self.rate_hz)
        self.index += 1
        timestamp_ns, claim = self.stamp()
        return [replace(self.frame(self.index, timestamp_ns), capture_context=claim)]


class NoitomReceiverThread(threading.Thread):
    """Commands and SDK calls stay in one worker; GUI only reads events."""
    def __init__(self, client, recorder, enabled_segments=SEGMENT_NAMES):
        super().__init__(name="noitom-receiver", daemon=True)
        self.client, self.recorder = client, recorder
        self.client.stamp = recorder.stamp_packet
        self.client.release_claim = recorder.release_claim
        self.commands, self.events = queue.Queue(), queue.Queue()
        self.capture_lock = threading.RLock()
        self.adapter = NoitomAdapter()
        self.shutdown = threading.Event()
        self.streaming = False
        self._last_sdk_index = None
        self.heading_offset_rad = 0.0
        self.set_enabled_segments(enabled_segments)

    def set_enabled_segments(self, enabled):
        enabled = frozenset(enabled)
        if enabled.difference(SEGMENT_NAMES):
            raise ValueError("Unknown Noitom segment")
        # Atomic reference replacement; each frame uses one immutable selection.
        self.enabled_segments = enabled

    def command(self, name):
        self.commands.put(name)

    def _consume(self, frame):
        enabled = self.enabled_segments
        heading_offset = self.heading_offset_rad
        diagnostics = {}
        if frame.frame_index is not None and self._last_sdk_index is not None:
            step = (frame.frame_index - self._last_sdk_index) % (2 ** 32)
            if step != 1:
                diagnostics["noitom_frame_index_discontinuity"] = {
                    "previous": self._last_sdk_index, "current": frame.frame_index,
                    "delivery_mode": self.client.delivery_mode}
        self._last_sdk_index = frame.frame_index
        error = None
        try:
            orientations = self.adapter.orientations(frame, enabled, heading_offset_rad=heading_offset)
        except ValueError as exc:
            orientations, error = {}, str(exc)
        self.recorder.noitom_frame(frame, orientations, error, diagnostics, enabled_segments=enabled,
                                  heading_offset_rad=heading_offset)
        self.events.put(("frame", frame, (orientations, error)))

    def run(self):
        try:
            while not self.shutdown.is_set():
                try:
                    command = self.commands.get(timeout=0 if self.streaming else .05)
                except queue.Empty:
                    command = None
                try:
                    if command:
                        if command == "shutdown":
                            break
                        with self.capture_lock:
                            getattr(self.client, command)()
                            if command in {"stop", "disconnect"}:
                                self.streaming = False
                                self.recorder.event("noitom_" + command, incomplete=True)
                            elif command == "connect":
                                self.heading_offset_rad = 0.0
                                self.recorder.event("noitom_connect", {
                                    "sdk_version": self.client.version, "delivery_mode": self.client.delivery_mode})
                            elif command == "start":
                                self.streaming = True
                                self._last_sdk_index = None
                            state = {"connect": "CONNECTED", "start": "STREAMING",
                                     "stop": "CONNECTED", "disconnect": "DISCONNECTED"}[command]
                            self.events.put(("state", state, ""))
                    # Also drain snapshots queued immediately before Stop.
                    if self.streaming or isinstance(self.client, NoitomClient):
                        with self.capture_lock:
                            frames = self.client.poll_events()
                            for frame in frames:
                                self._consume(frame)
                        if not frames and self.streaming:
                            time.sleep(.001)
                except Exception as error:
                    self.streaming = False
                    self.recorder.event("noitom_error", {"message": str(error)}, incomplete=True)
                    self.events.put(("state", "ERROR", str(error)))
                    try:
                        self.client.disconnect()
                    except Exception as cleanup_error:
                        self.events.put(("state", "ERROR", str(cleanup_error)))
        finally:
            try:
                self.client.disconnect()
            except Exception as error:
                self.events.put(("state", "ERROR", str(error)))
            if isinstance(self.client, NoitomClient):
                while True:
                    try:
                        frames = self.client.poll_events()
                    except Exception as error:
                        self.events.put(("state", "ERROR", str(error)))
                        continue
                    if not frames:
                        break
                    for frame in frames:
                        self._consume(frame)
            self.shutdown.set()
