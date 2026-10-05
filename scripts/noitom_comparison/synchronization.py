"""Clock and immutable acquisition envelopes; no Qt or native SDK dependencies."""
from collections import deque
from dataclasses import dataclass, field
import time
import numpy as np


@dataclass(frozen=True)
class JointTransform:
    # Local transform in project basis/metres, followed by untouched SDK values.
    position: np.ndarray | None
    rotation: np.ndarray
    parent: str | None = None
    default_position: np.ndarray | None = None
    tag: int | None = None
    sdk_position: np.ndarray | None = None
    sdk_rotation_xyzw: np.ndarray | None = None
    sdk_default_position: np.ndarray | None = None


@dataclass(frozen=True)
class NoitomFrame:
    host_timestamp_ns: int
    sdk_timestamp: float | int | None
    frame_index: int | None
    joints: dict[str, JointTransform]
    avatar_index: int = 0
    avatar_name: str = ""
    capture_context: object = field(default=None, repr=False, compare=False)


@dataclass
class StreamStats:
    frames: int = 0
    last_frame_ns: int | None = None
    recent: deque = field(default_factory=lambda: deque(maxlen=240))

    def observe(self, timestamp_ns: int):
        self.frames += 1
        self.last_frame_ns = timestamp_ns
        self.recent.append(timestamp_ns)

    def age_ms(self, now_ns: int | None = None):
        if self.last_frame_ns is None:
            return None
        return max(0, (time.monotonic_ns() if now_ns is None else now_ns) - self.last_frame_ns) / 1e6

    def rate_hz(self):
        if len(self.recent) < 2 or self.recent[-1] <= self.recent[0]:
            return 0.0
        return (len(self.recent) - 1) * 1e9 / (self.recent[-1] - self.recent[0])
