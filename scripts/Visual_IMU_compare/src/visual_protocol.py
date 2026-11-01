"""Strict decoder for the existing visual-mocap-bvh/1 wire protocol.

The video server's monotonic clock is retained only as source metadata. Pairing
with IMU uses the timestamp taken immediately after recv on this computer.
"""
from dataclasses import dataclass, field
import hashlib
import json
import math
import re
import numpy as np
from pyqt_mocap.mocap_core import matrix_to_quaternion
from storage.bvh import TO_BVH

PROTOCOL = 'visual-mocap-bvh/1'
MAX_MESSAGE_BYTES = 60000
AXES = 'X right, Y up, Z backward relative to calibration world'
BVH_TO_PROJECT = TO_BVH.T


@dataclass(frozen=True)
class JointTransform:
    position: np.ndarray
    rotation: np.ndarray
    parent: str | None
    default_position: np.ndarray
    tracked: bool


@dataclass(frozen=True)
class VisualFrame:
    host_timestamp_ns: int
    source_timestamp: float
    frame_index: int
    joints: dict[str, JointTransform]
    stream_id: str
    skeleton_id: str
    capture_context: object = field(default=None, repr=False, compare=False)


def finite_vector(values, size):
    if (not isinstance(values, (list, tuple)) or len(values) != size
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)):
        raise ValueError('Invalid finite numeric vector')
    return np.asarray(values, dtype=float)


def parse_hierarchy(text):
    if not isinstance(text, str) or len(text) > MAX_MESSAGE_BYTES:
        raise ValueError('Invalid BVH hierarchy')
    tokens = iter(text.replace('{', ' { ').replace('}', ' } ').split())
    joints, channels = [], []

    def expect(value):
        if next(tokens) != value:
            raise ValueError('Invalid BVH hierarchy: expected ' + value)

    def offset():
        expect('OFFSET')
        values = finite_vector([float(next(tokens)) for _ in range(3)], 3)
        if np.max(abs(values)) > 10000:
            raise ValueError('BVH offset out of range')
        return values

    def joint(parent=None, depth=0):
        name = next(tokens)
        if (not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,63}', name)
                or name in {j[0] for j in joints} or len(joints) >= 64 or depth > 32):
            raise ValueError('Invalid BVH joint name or hierarchy size')
        expect('{')
        position = offset()
        expect('CHANNELS')
        expected = (['Xposition', 'Yposition', 'Zposition'] if parent is None else [])
        expected += ['Zrotation', 'Xrotation', 'Yrotation']
        if int(next(tokens)) != len(expected) or [next(tokens) for _ in expected] != expected:
            raise ValueError('Expected root XYZ translation and ZXY rotations')
        joints.append((name, parent, position))
        channels.extend(name + '.' + channel for channel in expected)
        while True:
            item = next(tokens)
            if item == '}':
                break
            if item == 'JOINT':
                joint(name, depth + 1)
            elif item == 'End':
                expect('Site'); expect('{'); offset(); expect('}')
            else:
                raise ValueError('Unexpected BVH token: ' + item)

    try:
        expect('HIERARCHY'); expect('ROOT'); joint()
        if next(tokens, None) is not None:
            raise ValueError('Trailing BVH data')
    except (StopIteration, OverflowError) as error:
        raise ValueError('Incomplete BVH hierarchy') from error
    return joints, channels


def rotation_zxy(z, x, y):
    z, x, y = np.radians([z, x, y])
    cz, sz, cx, sx, cy, sy = np.cos(z), np.sin(z), np.cos(x), np.sin(x), np.cos(y), np.sin(y)
    return np.array([[cz, -sz, 0.], [sz, cz, 0.], [0., 0., 1.]]) @ np.array(
        [[1., 0., 0.], [0., cx, -sx], [0., sx, cx]]) @ np.array(
        [[cy, 0., sy], [0., 1., 0.], [-sy, 0., cy]])


class BvhDecoder:
    def __init__(self):
        self.metadata = None
        self.joints = []
        self.last_id = self.last_timestamp = None

    def decode(self, payload, received_ns):
        if len(payload) > MAX_MESSAGE_BYTES:
            raise ValueError('BVH message too large')
        def reject(value):
            raise ValueError('Non-finite JSON number: ' + value)
        message = json.loads(payload, parse_constant=reject)
        if not isinstance(message, dict) or message.get('protocol') != PROTOCOL:
            raise ValueError('Expected visual-mocap-bvh/1 JSON object')
        for key in ('stream_id', 'skeleton_id'):
            if not isinstance(message.get(key), str) or not 1 <= len(message[key]) <= 128:
                raise ValueError('Invalid ' + key)
        if message.get('type') == 'skeleton':
            joints, channels = parse_hierarchy(message.get('hierarchy'))
            if (message.get('units') not in {'m', 'cm'} or message.get('rotation_order') != 'ZXY'
                    or message.get('axes') != AXES or message.get('channel_names') != channels
                    or message.get('joint_names') != [j[0] for j in joints]
                    or message['skeleton_id'] != hashlib.sha256(message['hierarchy'].encode()).hexdigest()[:16]):
                raise ValueError('Inconsistent BVH skeleton metadata')
            identity = (message['stream_id'], message['skeleton_id'])
            previous = None if self.metadata is None else (self.metadata['stream_id'], self.metadata['skeleton_id'])
            if identity == previous and message != self.metadata:
                raise ValueError('Skeleton metadata changed without a new identity')
            if identity != previous:
                self.last_id = self.last_timestamp = None
            self.metadata, self.joints = message, joints
            return None
        if message.get('type') != 'frame':
            raise ValueError('Unknown BVH message type')
        if self.metadata is None or any(message[k] != self.metadata[k] for k in ('stream_id', 'skeleton_id')):
            return None  # UDP header may have been lost; it is repeated every 2 s.
        index, timestamp = message.get('frame_id'), message.get('timestamp_monotonic_s')
        if (type(index) is not int or index < 0 or type(timestamp) not in (int, float)
                or not math.isfinite(timestamp) or timestamp < 0):
            raise ValueError('Invalid frame ID or source timestamp')
        if self.last_id is not None and index <= self.last_id:
            return None  # Reordered/duplicate UDP samples must not refresh freshness.
        if self.last_timestamp is not None and timestamp <= self.last_timestamp:
            raise ValueError('Source time moved backwards; restart must use a new stream_id')
        values = finite_vector(message.get('channels'), len(self.metadata['channel_names']))
        tracked = message.get('tracked')
        if not isinstance(tracked, list) or len(tracked) != len(self.joints) or any(type(v) is not bool for v in tracked):
            raise ValueError('Invalid tracked-joint mask')
        scale = .01 if self.metadata['units'] == 'cm' else 1.
        joints = {}
        for i, (name, parent, offset) in enumerate(self.joints):
            local = rotation_zxy(*values[3+i*3:6+i*3])
            position = offset + values[:3] if parent is None else offset
            joints[name] = JointTransform(
                BVH_TO_PROJECT @ position * scale,
                matrix_to_quaternion(BVH_TO_PROJECT @ local @ BVH_TO_PROJECT.T), parent,
                BVH_TO_PROJECT @ offset * scale, tracked[i])
        self.last_id, self.last_timestamp = index, timestamp
        return VisualFrame(received_ns, timestamp, index, joints, message['stream_id'], message['skeleton_id'])
