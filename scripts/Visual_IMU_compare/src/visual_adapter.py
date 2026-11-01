"""Retarget the visual skeleton onto the same five segments as the IMU suit."""
import math
import numpy as np
from pyqt_mocap.mocap_core import compute_body_pose, matrix_to_quaternion, quaternion_to_matrix, normalize_quaternion
from .visual_protocol import BVH_TO_PROJECT

VISUAL_TO_PROJECT = {'Spine': 'spine', 'LeftArm': 'shoulder.L', 'LeftForeArm': 'forearm.L',
                     'RightArm': 'shoulder.R', 'RightForeArm': 'forearm.R'}
CHILDREN = {'Spine': 'Chest', 'LeftArm': 'LeftForeArm', 'LeftForeArm': 'LeftHand',
            'RightArm': 'RightForeArm', 'RightForeArm': 'RightHand'}
COORDINATE_DESCRIPTION = {
    'up': '+Z', 'front': '+Y', 'handedness': 'right', 'unit': 'metre',
    'quaternion_order': 'wxyz', 'bvh_to_project_basis': BVH_TO_PROJECT.tolist(),
    'root_translation_rendered': False,
    'orientation_method': 'visual torso basis and limb swing; axial twist is unmeasured',
}


def world_transforms(frame):
    worlds = {}
    for name, joint in frame.joints.items():
        rotation = quaternion_to_matrix(joint.rotation)
        position = joint.position
        if joint.parent is not None:
            parent_rotation, parent_position = worlds[joint.parent]
            position = parent_position + parent_rotation @ position
            rotation = parent_rotation @ rotation
        worlds[name] = rotation, position
    return worlds


def rotation_between(source, target):
    if min(np.linalg.norm(source), np.linalg.norm(target)) < 1e-8:
        raise ValueError('Zero-length visual bone')
    a, b = source / np.linalg.norm(source), target / np.linalg.norm(target)
    dot = float(np.clip(a @ b, -1, 1))
    if dot < -1 + 1e-8:
        axis = np.cross(a, np.eye(3)[np.argmin(abs(a))])
        return quaternion_to_matrix(np.r_[0., axis / np.linalg.norm(axis)])
    return quaternion_to_matrix(normalize_quaternion(np.r_[1 + dot, np.cross(a, b)]))


def torso_heading(frame):
    worlds = world_transforms(frame)
    try:
        direction = worlds['RightArm'][1] - worlds['LeftArm'][1]
    except KeyError as error:
        raise ValueError('Нет плеч для совмещения направления') from error
    if np.linalg.norm(direction[:2]) < 1e-6:
        raise ValueError('Для совмещения встаньте прямо')
    return math.atan2(direction[1], direction[0])


class VisualAdapter:
    def __init__(self):
        self.rest_segments = compute_body_pose({}).tracked_segments

    def orientations(self, frame, enabled_segments=None, *, heading_offset_rad=0.):
        if not math.isfinite(heading_offset_rad):
            raise ValueError('Invalid heading offset')
        enabled = set(VISUAL_TO_PROJECT.values()) if enabled_segments is None else set(enabled_segments)
        worlds = world_transforms(frame)
        c, s = math.cos(heading_offset_rad), math.sin(heading_offset_rad)
        heading = np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
        result = {}
        for name, segment in VISUAL_TO_PROJECT.items():
            if segment not in enabled:
                continue
            child_name = CHILDREN[name]
            if name not in frame.joints or child_name not in frame.joints:
                raise ValueError('Visual skeleton missing ' + name + '/' + child_name)
            # Held rotations for occluded landmarks are not valid measurements.
            ancestor = name
            while ancestor is not None:
                if not frame.joints[ancestor].tracked:
                    raise ValueError('Visual joint is untracked: ' + ancestor)
                ancestor = frame.joints[ancestor].parent
            child = frame.joints[child_name]
            if child.parent != name:
                raise ValueError('Unexpected visual bone hierarchy')
            start, end = self.rest_segments[segment]
            correction = rotation_between(end - start, child.default_position)
            result[segment] = matrix_to_quaternion(heading @ worlds[name][0] @ correction)
        return result
