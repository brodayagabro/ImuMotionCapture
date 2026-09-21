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

    def append(self, timestamp, orientations):
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
        self.frames.append(np.concatenate((self.units_per_meter * TO_BVH @ self.root_position,
                                           angles.ravel())))
        self.times.append(float(timestamp))

    def save(self, path):
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
        count = int(math.floor((self.times[-1] - self.times[0]) * self.fps + 1e-6)) + 1
        lines.extend(("MOTION", f"Frames: {count}", f"Frame Time: {1/self.fps:.10f}"))
        path = Path(path)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="ascii", newline="\n",
                                             dir=path.parent, delete=False) as stream:
                temporary = stream.name
                stream.write("\n".join(lines) + "\n")
                source = 0
                for frame_index in range(count):
                    timestamp = self.times[0] + frame_index / self.fps
                    while source + 1 < len(self.times) and self.times[source + 1] <= timestamp + 1e-7:
                        source += 1
                    frame = self.frames[source]
                    values = list(frame[:3])
                    for index in order:
                        values.extend(frame[3 + index * 3:6 + index * 3])
                    stream.write(" ".join(f"{v:.8f}" for v in values) + "\n")
            os.replace(temporary, path)
        finally:
            if temporary is not None and os.path.exists(temporary):
                os.unlink(temporary)
        return count
