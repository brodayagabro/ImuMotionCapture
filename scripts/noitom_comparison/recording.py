"""Lossless asynchronous session writer with acquisition claims and shared T0."""
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime
from pathlib import Path
import copy
import csv
import json
import queue
import threading
import time
import uuid
import numpy as np


def plain(value):
    if value is None or type(value) in (str, int, float, bool):
        return value
    if is_dataclass(value):
        # The acquisition boundary already owns a deep copy. asdict would copy
        # every joint array again before immediately converting it to JSON.
        return {field.name: plain(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


class RecordingSession:
    def __init__(self, root, metadata, t0_ns):
        now = datetime.now().astimezone()
        self.session_id = now.strftime("%Y-%m-%d_%H-%M-%S_%f") + "_" + uuid.uuid4().hex[:6]
        self.path = Path(root).expanduser().resolve() / self.session_id
        self.t0_ns = t0_ns
        self.stop_ns = None
        self.pending = 0
        self.queue = queue.Queue()
        self.done = threading.Event()
        self.error = None
        self.metadata = {
            **plain(copy.deepcopy(metadata)), "schema_version": 1,
            "session_id": self.session_id, "started_at": now.isoformat(),
            "recording_clock": "time.monotonic_ns", "recording_t0_ns": t0_ns,
            "timestamp_semantics": "host acquisition, not hardware sampling time",
            "quaternion_order": "wxyz", "status": "recording",
        }
        self.thread = threading.Thread(target=self._write, name="experiment-writer", daemon=False)
        self.thread.start()

    def _save_metadata(self):
        temp = self.path / "metadata.json.tmp"
        temp.write_text(json.dumps(self.metadata, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        temp.replace(self.path / "metadata.json")

    def _write(self):
        handles = {}
        counts = {"own_frames": 0, "own_packets": 0, "noitom_frames": 0}
        reasons = []
        try:
            self.path.mkdir(parents=True, exist_ok=False)
            self._save_metadata()
            for name in counts:
                handles[name] = (self.path / f"{name}.jsonl").open("w", encoding="utf-8")
            handles["events"] = (self.path / "events.csv").open("w", encoding="utf-8", newline="")
            events = csv.writer(handles["events"])
            events.writerow(["host_timestamp_ns", "relative_timestamp_ns", "event", "details_json"])
            events.writerow([self.t0_ns, 0, "recording_started", "{}"])
            while True:
                kind, timestamp_ns, data = self.queue.get()
                if kind == "finish":
                    self.metadata.update(data)
                    events.writerow([timestamp_ns, timestamp_ns - self.t0_ns, "recording_stopped", "{}"])
                    break
                if kind == "event":
                    events.writerow([timestamp_ns, timestamp_ns - self.t0_ns,
                                     data["event"], json.dumps(data["details"], ensure_ascii=False)])
                    if data["incomplete"]:
                        reasons.append(data["event"])
                else:
                    # Noitom already supplies a JSON-ready independent snapshot.
                    row = {**(data if kind == "noitom_frames" else plain(data)), "host_timestamp_ns": timestamp_ns,
                           "relative_timestamp_ns": timestamp_ns - self.t0_ns}
                    handles[kind].write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    counts[kind] += 1
                # Bound crash exposure without synchronizing acquisition to disk.
                if sum(counts.values()) % 100 == 0:
                    for handle in handles.values():
                        handle.flush()
            if not counts["own_frames"]:
                reasons.append("no_own_frames")
            if not counts["noitom_frames"]:
                reasons.append("no_noitom_frames")
            self.metadata.update(counts=counts, incomplete_reasons=sorted(set(reasons)),
                                 status="incomplete" if reasons else "complete")
        except Exception as error:
            self.error = f"{type(error).__name__}: {error}"
            self.metadata.update(status="error", error=self.error, counts=counts)
        finally:
            for handle in handles.values():
                try:
                    handle.close()
                except OSError as error:
                    self.error = str(error)
                    self.metadata.update(status="error", error=self.error)
            try:
                if self.path.is_dir():
                    self._save_metadata()
            except OSError as error:
                self.error = str(error)
            self.done.set()


@dataclass(frozen=True)
class PacketClaim:
    session: RecordingSession
    timestamp_ns: int


class ExperimentRecorder:
    def __init__(self):
        self._lock = threading.Lock()
        self.active = None
        self.sessions = []

    def start(self, root, metadata, *, t0_ns=None):
        with self._lock:
            if self.active is not None:
                raise RuntimeError("Recording already active")
            session = RecordingSession(root, metadata, time.monotonic_ns() if t0_ns is None else t0_ns)
            self.active = session
            self.sessions.append(session)
            return session

    def stamp_packet(self):
        with self._lock:
            timestamp_ns = time.monotonic_ns()
            session = self.active
            if session is None or session.error:
                return timestamp_ns, None
            session.pending += 1
            return timestamp_ns, PacketClaim(session, timestamp_ns)

    def complete_packet(self, claim, packet, frame=None):
        if claim is None:
            return
        with self._lock:
            session = claim.session
            if not session.error:
                session.queue.put(("own_packets", claim.timestamp_ns, copy.deepcopy(packet)))
                if frame is not None:
                    session.queue.put(("own_frames", claim.timestamp_ns, copy.deepcopy(frame)))
            session.pending -= 1
            self._finish_if_ready(session)

    def noitom_frame(self, frame, orientations, mapping_error=None, diagnostics=None, *, enabled_segments=None,
                     heading_offset_rad=0.0):
        with self._lock:
            claim = frame.capture_context
            session = claim.session if claim is not None else self.active
            if session is None:
                return
            if not session.error and frame.host_timestamp_ns >= session.t0_ns:
                session.queue.put(("noitom_frames", frame.host_timestamp_ns, plain({
                    "sdk_timestamp": frame.sdk_timestamp, "sdk_frame_index": frame.frame_index,
                    "avatar_index": frame.avatar_index, "avatar_name": frame.avatar_name,
                    "joints": frame.joints,
                    "segment_orientations": orientations, "mapping_error": mapping_error,
                    "heading_offset_rad": heading_offset_rad,
                    "enabled_segments": sorted(orientations if enabled_segments is None else enabled_segments),
                    "diagnostics": diagnostics or {},
                })))
                if mapping_error:
                    session.queue.put(("event", frame.host_timestamp_ns, {
                        "event": "noitom_mapping_error", "details": {"message": mapping_error}, "incomplete": True}))
                for name, details in (diagnostics or {}).items():
                    session.queue.put(("event", frame.host_timestamp_ns, {
                        "event": name, "details": details, "incomplete": True}))
            if claim is not None:
                session.pending -= 1
                self._finish_if_ready(session)

    def release_claim(self, claim, error=None):
        if claim is None:
            return
        with self._lock:
            session = claim.session
            if error and not session.error:
                session.queue.put(("event", claim.timestamp_ns, {
                    "event": "noitom_snapshot_error", "details": {"message": error}, "incomplete": True}))
            session.pending -= 1
            self._finish_if_ready(session)

    def event(self, name, details=None, *, incomplete=False, timestamp_ns=None):
        with self._lock:
            if self.active is not None and not self.active.error:
                self.active.queue.put(("event", time.monotonic_ns() if timestamp_ns is None else timestamp_ns,
                    {"event": name, "details": plain(details or {}), "incomplete": incomplete}))

    def stop(self, *, stop_ns=None):
        with self._lock:
            session, self.active = self.active, None
            if session is not None:
                session.stop_ns = time.monotonic_ns() if stop_ns is None else stop_ns
                self._finish_if_ready(session)
            return session

    def recording_state(self):
        """Generic event receiver readiness; never implies that sensors are live."""
        with self._lock:
            session = self.active
            ready = session is not None and not session.error
            return {"ready": bool(ready),
                    "recording_session_id": session.session_id if ready else None}

    def external_event(self, payload, received_ns):
        """Atomically bind an external marker to the intended active recording."""
        with self._lock:
            session = self.active
            if session is None or session.error:
                return {"accepted": False, "reason": "recording_inactive"}
            if payload.get("recording_session_id") != session.session_id or received_ns < session.t0_ns:
                return {"accepted": False, "reason": "recording_session_changed"}
            session.queue.put(("event", received_ns, {
                "event": payload["type"], "details": copy.deepcopy(payload), "incomplete": False}))
            return {"accepted": True, "host_timestamp_ns": received_ns,
                    "recording_session_id": session.session_id,
                    "recording_path": str(session.path)}

    @staticmethod
    def _finish_if_ready(session):
        if session.stop_ns is not None and session.pending == 0:
            session.queue.put(("finish", session.stop_ns, {
                "recording_stop_ns": session.stop_ns,
                "duration_ns": session.stop_ns - session.t0_ns,
            }))
