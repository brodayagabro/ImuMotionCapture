"""BVH recording without Qt. Input coordinates: metres, X right/Y forward/Z up.

Output: configurable length units (default centimetres), Y up; ZXY degrees.
Irregular samples are resampled by holding the last received pose.
"""

from dataclasses import dataclass
from pathlib import Path
import math
import os
import tempfile

import numpy as np


TO_BVH = np.array(((1., 0., 0.), (0., 0., 1.), (0., -1., 0.)))


@dataclass(frozen=True)
class Joint:
    name: str
    parent: int | None
    offset: tuple[float, float, float]
    source: str | None = None
    end: tuple[float, float, float] = (0., 0., 0.)


def euler_zxy(matrix):
    """Decompose column-vector R = Rz @ Rx @ Ry, including gimbal lock."""
    x = math.asin(float(np.clip(matrix[2, 1], -1., 1.)))
    if abs(math.cos(x)) > 1e-7:
        z = math.atan2(-matrix[0, 1], matrix[1, 1])
        y = math.atan2(-matrix[2, 0], matrix[2, 2])
    else:
        z = math.atan2(matrix[1, 0], matrix[0, 0])
        y = 0.
    return np.degrees((z, x, y))


class BvhRecording:
    def __init__(self, joints, root_position, fps=30, units_per_meter=100.):
        if not math.isfinite(units_per_meter) or units_per_meter <= 0:
            raise ValueError("BVH units_per_meter must be positive and finite")
        self.units_per_meter = float(units_per_meter)
        if not math.isfinite(fps) or not 1 <= fps <= 100:
            raise ValueError("BVH FPS must be between 1 and 100")
        self.joints = tuple(joints)
        if not self.joints or self.joints[0].parent is not None:
            raise ValueError("A root joint is required")
        for index, joint in enumerate(self.joints):
            if index and (joint.parent is None or not 0 <= joint.parent < index):
                raise ValueError("Parents must precede children")
        self.root_position = np.asarray(root_position, dtype=float).copy()
        self.fps = float(fps)
        self.times = []
        self.frames = []

    def append(self, timestamp, orientations, *, root_position=None):
        if not math.isfinite(timestamp):
            raise ValueError("Invalid sample timestamp")
        if self.times and timestamp <= self.times[-1]:
            return
        # None means inherit the parent's world orientation (static child).
        world = []
        angles = []
        for joint in self.joints:
            parent = np.eye(3) if joint.parent is None else world[joint.parent]
            current = parent if joint.source is None else np.asarray(
                orientations[joint.source], dtype=float
            )
            if current.shape != (3, 3) or not np.isfinite(current).all():
                raise ValueError("Invalid joint rotation")
            world.append(current)
            angles.append(euler_zxy(TO_BVH @ (parent.T @ current) @ TO_BVH.T))
        angles = np.asarray(angles)
        if self.frames:
            previous = self.frames[-1][3:].reshape((-1, 3))
            angles += 360. * np.round((previous - angles) / 360.)
        position = self.root_position if root_position is None else np.asarray(root_position, dtype=float)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("Invalid root position")
        self.frames.append(np.concatenate((self.units_per_meter * TO_BVH @ position,
                                           angles.ravel())))
        self.times.append(float(timestamp))

    def append_batch(self, timestamps, orientations):
        """Vectorized world rotations on a precomputed grid, with a fixed root."""
        times = np.asarray(timestamps, dtype=float)
        if times.ndim != 1 or not len(times) or not np.isfinite(times).all():
            raise ValueError('Invalid sample timestamps')
        if np.any(np.diff(times) <= 0) or (self.times and times[0] <= self.times[-1]):
            raise ValueError('Sample timestamps must increase')
        world, angles = [], []
        identity = np.broadcast_to(np.eye(3), (len(times), 3, 3))
        for joint in self.joints:
            parent = identity if joint.parent is None else world[joint.parent]
            current = parent if joint.source is None else np.asarray(orientations[joint.source], float)
            if current.shape != identity.shape or not np.isfinite(current).all():
                raise ValueError('Invalid joint rotations')
            world.append(current)
            local = TO_BVH @ (np.swapaxes(parent, -1, -2) @ current) @ TO_BVH.T
            x = np.arcsin(np.clip(local[:, 2, 1], -1., 1.))
            regular = np.abs(np.cos(x)) > 1e-7
            z = np.where(regular, np.arctan2(-local[:, 0, 1], local[:, 1, 1]),
                         np.arctan2(local[:, 1, 0], local[:, 0, 0]))
            y = np.where(regular, np.arctan2(-local[:, 2, 0], local[:, 2, 2]), 0.)
            angles.append(np.degrees(np.column_stack((z, x, y))))
        angles = np.stack(angles, axis=1).reshape(len(times), -1)
        if self.frames:
            angles = np.vstack((self.frames[-1][3:], angles))
        angles = np.unwrap(angles, axis=0, period=360.)
        if self.frames:
            angles = angles[1:]
        positions = np.broadcast_to(self.units_per_meter * TO_BVH @ self.root_position, (len(times), 3))
        self.frames.extend(np.column_stack((positions, angles)))
        self.times.extend(times.tolist())

    def save(self, path, *, uniform_grid=False, progress=None):
        """Atomically write a recording; leave an existing file intact on failure."""
        if not self.frames:
            raise ValueError("Нет кадров для экспорта BVH")
        order = []
        lines = ["HIERARCHY"]

        def vector(value):
            return " ".join(f"{x:.8f}" for x in self.units_per_meter * TO_BVH @ value)

        def emit(index, depth):
            joint = self.joints[index]
            pad = "  " * depth
            order.append(index)
            lines.extend((f"{pad}{'ROOT' if index == 0 else 'JOINT'} {joint.name}",
                          pad + "{", pad + "  OFFSET " + vector(joint.offset)))
            channels = "6 Xposition Yposition Zposition" if index == 0 else "3"
            lines.append(pad + f"  CHANNELS {channels} Zrotation Xrotation Yrotation")
            children = [i for i, j in enumerate(self.joints) if j.parent == index]
            for child in children:
                emit(child, depth + 1)
            if not children:
                lines.extend((pad + "  End Site", pad + "  {",
                              pad + "    OFFSET " + vector(joint.end), pad + "  }"))
            lines.append(pad + "}")

        emit(0, 0)
        if uniform_grid and not np.allclose(np.diff(self.times), 1 / self.fps, rtol=0, atol=1e-8):
            raise ValueError("Samples must lie on the requested uniform grid")
        count = len(self.frames) if uniform_grid else int(math.floor((self.times[-1] - self.times[0]) * self.fps + 1e-6)) + 1
        lines.extend(("MOTION", f"Frames: {count}", f"Frame Time: {1/self.fps:.12f}"))
        path = Path(path)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="ascii", newline="\n",
                                             dir=path.parent, delete=False) as stream:
                temporary = stream.name
                stream.write("\n".join(lines) + "\n")
                source = 0
                for frame_index in range(count):
                    if uniform_grid:
                        source = frame_index
                    timestamp = self.times[0] + frame_index / self.fps
                    while not uniform_grid and source + 1 < len(self.times) and self.times[source + 1] <= timestamp + 1e-7:
                        source += 1
                    frame = self.frames[source]
                    values = list(frame[:3])
                    for index in order:
                        values.extend(frame[3 + index * 3:6 + index * 3])
                    stream.write(" ".join(f"{v:.8f}" for v in values) + "\n")
                    if progress is not None and (frame_index % 128 == 0 or frame_index + 1 == count):
                        progress((frame_index + 1) / count)
            os.replace(temporary, path)
        finally:
            if temporary is not None and os.path.exists(temporary):
                os.unlink(temporary)
        return count
