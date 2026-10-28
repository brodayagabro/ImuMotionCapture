"""Strict BVH reader and forward kinematics. BVH coordinates: X right, Y up.

The current comparison skeleton stores lengths in metres. No Euler subtraction
is used: rotations are composed in channel order (intrinsic rotations).
"""
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re

import numpy as np
from scipy.spatial.transform import Rotation

ARMS = ('LeftArm', 'LeftForeArm', 'RightArm', 'RightForeArm')
POINTS = ('LeftForeArm', 'LeftForeArm.end', 'RightForeArm', 'RightForeArm.end')


@dataclass
class Motion:
    path: Path
    names: tuple
    parents: tuple
    offsets: np.ndarray
    ends: dict
    dt: float
    rotations: np.ndarray
    positions: np.ndarray
    sha256: str

    @property
    def count(self):
        return len(self.positions)

    @property
    def duration(self):
        return (self.count - 1) * self.dt

    def index(self, name):
        return self.names.index(name)

    def body_rotations(self):
        spine = self.rotations[:, self.index('Spine')]
        return np.swapaxes(spine, -1, -2)[:, None] @ self.rotations

    def body_positions(self):
        origin = self.positions[:, self.index('Spine')]
        inverse = np.swapaxes(self.rotations[:, self.index('Spine')], -1, -2)
        return np.einsum('fij,fkj->fki', inverse, self.positions - origin[:, None])

    def points(self, aligned=True):
        result = []
        rotations = self.body_rotations() if aligned else self.rotations
        positions = self.body_positions() if aligned else self.positions
        for name in POINTS:
            joint = name.removesuffix('.end')
            i = self.index(joint)
            p = positions[:, i]
            if name.endswith('.end'):
                p = p + rotations[:, i] @ self.ends[i]
            result.append(p)
        return np.stack(result, axis=1)

    def directions(self):
        """Four bone unit vectors in the spine frame, shape (frames,4,3)."""
        rotations = self.body_rotations()
        result = []
        for name in ARMS:
            i = self.index(name)
            children = [j for j, parent in enumerate(self.parents) if parent == i]
            vector = self.offsets[children[0]] if children else self.ends[i]
            result.append(rotations[:, i] @ (vector / np.linalg.norm(vector)))
        return np.stack(result, axis=1)


def load_bvh(path):
    path = Path(path)
    raw = path.read_bytes()
    hierarchy, separator, data = raw.decode('ascii').partition('MOTION')
    if not separator:
        raise ValueError('BVH has no MOTION section')
    tokens = iter(re.findall(r'[^\s{}]+|[{}]', hierarchy))
    names, parents, offsets, channels, ends = [], [], [], [], {}

    def expect(value):
        actual = next(tokens)
        if actual != value:
            raise ValueError(f'BVH expected {value}, got {actual}')

    def vector():
        return [float(next(tokens)) for _ in range(3)]

    def joint(parent):
        i = len(names)
        name = next(tokens)
        if name in names:
            raise ValueError('Duplicate BVH joint')
        names.append(name)
        parents.append(parent)
        expect('{')
        expect('OFFSET')
        offsets.append(vector())
        expect('CHANNELS')
        n = int(next(tokens))
        ch = [next(tokens) for _ in range(n)]
        expected = (['Xposition', 'Yposition', 'Zposition'] if parent is None else [])
        if ch[:len(expected)] != expected or len(ch) != len(expected) + 3:
            raise ValueError('Unsupported BVH translation channels')
        rotation = ch[len(expected):]
        if sorted(rotation) != ['Xrotation', 'Yrotation', 'Zrotation']:
            raise ValueError('Unsupported BVH rotation channels')
        channels.append(ch)
        while True:
            token = next(tokens)
            if token == '}':
                break
            if token == 'JOINT':
                joint(i)
            elif token == 'End':
                expect('Site')
                expect('{')
                expect('OFFSET')
                ends[i] = np.array(vector())
                expect('}')
            else:
                raise ValueError(f'Unexpected BVH token: {token}')

    try:
        expect('HIERARCHY')
        expect('ROOT')
        joint(None)
        if list(tokens):
            raise ValueError('Trailing hierarchy tokens')
    except StopIteration as error:
        raise ValueError('Truncated BVH hierarchy') from error
    lines = data.strip().splitlines()
    count = int(lines[0].split(':')[1])
    dt = float(lines[1].split(':')[1])
    width = sum(map(len, channels))
    rows = [np.fromstring(line, sep=' ') for line in lines[2:]]
    if count < 2 or len(rows) != count or any(len(row) != width for row in rows):
        raise ValueError('BVH frame count/channel width mismatch')
    values = np.asarray(rows)
    if not np.isfinite(values).all() or not np.isfinite(dt) or dt <= 0:
        raise ValueError('Non-finite motion or invalid frame interval')
    if not np.isfinite(offsets).all() or any(not np.isfinite(v).all() for v in ends.values()):
        raise ValueError('Non-finite geometry')
    world = np.empty((count, len(names), 3, 3))
    positions = np.empty((count, len(names), 3))
    cursor = 0
    for i, (parent, ch) in enumerate(zip(parents, channels)):
        shift = 3 if parent is None else 0
        order = ''.join(c[0] for c in ch[shift:])
        local = Rotation.from_euler(order, values[:, cursor + shift:cursor + len(ch)], degrees=True).as_matrix()
        if parent is None:
            world[:, i] = local
            positions[:, i] = values[:, :3] + offsets[i]
        else:
            world[:, i] = world[:, parent] @ local
            positions[:, i] = positions[:, parent] + world[:, parent] @ np.array(offsets[i])
        cursor += len(ch)
    motion = Motion(path, tuple(names), tuple(parents), np.array(offsets), ends, dt, world, positions,
                    hashlib.sha256(raw).hexdigest())
    for name in ('Spine', *ARMS):
        if name not in names:
            raise ValueError(f'Missing required joint: {name}')
    for name in ('LeftForeArm', 'RightForeArm'):
        if motion.index(name) not in ends:
            raise ValueError(f'Missing wrist End Site: {name}')
    if len(names) != 14 or not .3 < np.linalg.norm(motion.ends[motion.index('LeftForeArm')]) < .6:
        raise ValueError('Expected comparison_14_joints_v1 in metres; legacy BVH requires explicit conversion')
    return motion


def require_pair(a, b):
    if a.names != b.names or a.parents != b.parents or not np.allclose(a.offsets, b.offsets, atol=1e-8):
        raise ValueError('Different skeleton geometry')
    if a.ends.keys() != b.ends.keys() or any(not np.allclose(a.ends[i], b.ends[i], atol=1e-8) for i in a.ends):
        raise ValueError('Different End Site geometry')
    if a.count != b.count or abs(a.dt - b.dt) > 1e-10:
        raise ValueError('Different time grids; automatic time warping is disabled')
