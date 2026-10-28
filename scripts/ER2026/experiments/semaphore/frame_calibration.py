"""Capture a short timed window of complete measurements, with pose statistics."""
from collections import Counter, deque
from dataclasses import replace
import math

import numpy as np

from pyqt_mocap.calibration import PoseRecorder, MIN_SAMPLES_PER_SEGMENT, MAX_POSE_SPREAD_DEG, MIN_ALIGNMENT_MOTION_RAD
from pyqt_mocap.mocap_core import SEGMENT_NAMES
from pyqt_mocap.quaternion_utils import angular_distance, average_quaternions, rotate_vector, validate_input_quaternion
from pyqt_mocap.t_pose_calibration import ARM_DISTAL_SENSOR_AXES

POSE_CAPTURE_S = 1.5
POSE_PREPARATION_S = 3.
MAX_CAPTURE_GAP_S = .5


class FramePoseRecorder:
    def __init__(self, name, started_s, *, previous=None, duration_s=POSE_CAPTURE_S, timeout_s=120.):
        self.name, self.started_s, self.previous = name, started_s, previous
        if not math.isfinite(duration_s) or duration_s <= 0:
            raise ValueError('Длительность записи позы должна быть положительной')
        self.duration_s, self.timeout_s = duration_s, timeout_s
        self.completed = False
        self.recorder = PoseRecorder()
        self.window = deque(maxlen=MIN_SAMPLES_PER_SEGMENT)
        self.anchor = None
        self.seen = set()
        self.latest_key = None
        self.accepted = 0
        self.rejected = Counter()
        self.frames_seen = 0
        self.last_received_s = None

    def timeout(self, now):
        if self.completed:
            return
        if self.accepted and now - self.last_received_s > MAX_CAPTURE_GAP_S:
            raise ValueError('Прервался поток полных кадров всех пяти датчиков. Повторите этап.')
        if now - self.started_s >= self.timeout_s:
            raise ValueError(f"Тайм-аут: записано {self.elapsed_s:.2f}/{self.duration_s:g} с, {self.accepted} полных кадров. "
                             "Нужны все пять датчиков и неподвижная поза. Повторите этап.")

    @property
    def elapsed_s(self):
        if self.recorder.started_s is None or self.recorder.ended_s is None:
            return 0.
        return self.recorder.ended_s - self.recorder.started_s

    def diagnostics(self):
        end = self.recorder.ended_s
        start = self.recorder.started_s
        duration = end - start if start is not None and end is not None else 0.
        return {"required_duration_s": self.duration_s, "minimum_frames": MIN_SAMPLES_PER_SEGMENT,
                "accepted_frames": self.accepted, "rejected_frames": sum(self.rejected.values()),
                "rejection_reasons": dict(self.rejected), "received_frames": self.frames_seen,
                "waiting_started_s": self.started_s, "capture_started_s": start, "capture_ended_s": end,
                "capture_duration_s": duration,
                "accepted_frequency_hz": (self.accepted - 1) / duration if duration > 0 else None,
                "stable_window_frames": MIN_SAMPLES_PER_SEGMENT, "stability_limit_deg": MAX_POSE_SPREAD_DEG}

    def add_frame(self, samples, key, timestamp_s):
        if self.completed:
            return None
        self.timeout(timestamp_s)
        self.frames_seen += 1
        if timestamp_s < self.started_s:
            self.rejected['before_stage'] += 1
            return None
        if key is None or set(samples) != set(SEGMENT_NAMES):
            self.rejected['incomplete_or_invalid'] += 1
            self.window.clear()
            return None
        if key in self.seen or (self.latest_key is not None and
                               (key[0] <= self.latest_key[0] or key[1] < self.latest_key[1])) or (
                self.last_received_s is not None and timestamp_s <= self.last_received_s):
            self.rejected['duplicate_or_out_of_order'] += 1
            return None
        self.seen.add(key)
        self.latest_key = key
        try:
            samples = {s: validate_input_quaternion(samples[s]) for s in SEGMENT_NAMES}
        except (ValueError, TypeError):
            self.rejected['invalid_quaternion'] += 1
            return None
        self.last_received_s = timestamp_s
        self.window.append(samples)
        if self.anchor is None:
            if len(self.window) < self.window.maxlen:
                self.rejected['settling'] += 1
                return None
            means = {s: average_quaternions([row[s] for row in self.window]) for s in SEGMENT_NAMES}
            spread = max(np.percentile([math.degrees(angular_distance(row[s], means[s]))
                                       for row in self.window], 95) for s in SEGMENT_NAMES)
            moved = self.previous is None or all(math.acos(float(np.clip(
                rotate_vector(self.previous.average[s], axis) @ rotate_vector(means[s], axis), -1, 1)))
                >= MIN_ALIGNMENT_MOTION_RAD for s, axis in ARM_DISTAL_SENSOR_AXES.items())
            if spread > MAX_POSE_SPREAD_DEG or not moved:
                self.rejected['moving_or_previous_pose'] += 1
                return None
            self.anchor = means
        if max(math.degrees(angular_distance(samples[s], self.anchor[s])) for s in SEGMENT_NAMES) > MAX_POSE_SPREAD_DEG:
            # A contiguous stable pose is required: never combine two plateaus.
            self.rejected['unstable_or_discarded'] += self.accepted + 1
            self.accepted = 0
            self.anchor = None
            self.window.clear()
            self.recorder = PoseRecorder()
            return None
        self.recorder.add_snapshot({s: (q, timestamp_s, key) for s, q in samples.items()})
        self.accepted += 1
        # Finish on acquisition time, not packet count or GUI refresh rate.
        # The first sample at/after the boundary completes the window; tolerate
        # only floating-point roundoff, not a missing final measurement.
        if self.elapsed_s + 1e-9 >= self.duration_s:
            capture = self.recorder.finish(self.name)
            self.completed = True
            return replace(capture, frame_diagnostics=self.diagnostics())
        return None
