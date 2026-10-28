"""Identical geometry and channels for the two five-segment measurements."""
import numpy as np

from storage.bvh import BvhRecording, Joint
from pyqt_mocap.mocap_core import compute_body_pose

SKELETON_ID = 'comparison_14_joints_v1'
HEIGHT_M = 1.70
SEGMENTS = ('spine', 'shoulder.L', 'forearm.L', 'shoulder.R', 'forearm.R')


def create_comparison_recording(fps):
    points = np.asarray(compute_body_pose({}).joints)
    floor = min(points[13, 2], points[14, 2])
    scale = HEIGHT_M / (points[2, 2] + .10 - floor)
    p = (points - (0., 0., floor)) * scale
    # The forearms end at the wrists: no hand/finger rotation channels.
    # Unmeasured head, pelvis and legs are identical static geometry in both.
    specs = (
        ('Hips', None, 0, None), ('Spine', 0, 0, 'spine'),
        ('Chest', 1, 1, None), ('Head', 2, 2, None),
        ('LeftArm', 1, 3, 'shoulder.L'), ('LeftForeArm', 4, 5, 'forearm.L'),
        ('RightArm', 1, 4, 'shoulder.R'), ('RightForeArm', 6, 6, 'forearm.R'),
        ('LeftUpLeg', 0, 9, None), ('LeftLeg', 8, 11, None), ('LeftFoot', 9, 13, None),
        ('RightUpLeg', 0, 10, None), ('RightLeg', 11, 12, None), ('RightFoot', 12, 14, None),
    )
    joints = []
    for name, parent, point, source in specs:
        offset = np.zeros(3) if parent is None else p[point] - p[specs[parent][2]]
        end = np.zeros(3)
        if name == 'Head': end = np.array((0., 0., .10 * scale))
        elif name == 'LeftForeArm': end = p[7] - p[5]
        elif name == 'RightForeArm': end = p[8] - p[6]
        elif name.endswith('Foot'): end = np.array((0., .13 * scale, 0.))
        joints.append(Joint(name, parent, tuple(offset), source, tuple(end)))
    return BvhRecording(joints, p[0], fps, units_per_meter=1.)
