"""All Noitom coordinate conversion and mannequin retargeting lives here."""
import math
import numpy as np
from pyqt_mocap.quaternion_utils import from_xyzw
from pyqt_mocap.mocap_core import (
    compute_body_pose, matrix_to_quaternion, normalize_quaternion, quaternion_to_matrix,
    validate_input_quaternion,
)
from .synchronization import NoitomFrame

# Official BVH default: anatomical left +X, up +Y, forward +Z, centimetres.
# Keep the SDK's working default render preset, then change basis in Python.
SDK_TO_PROJECT_BASIS = np.array([[-1., 0., 0.], [0., 0., 1.], [0., 1., 0.]])

# MocapCApi.h JointTag values; Shoulder tags represent clavicles, not upper arms.
NOITOM_TO_PROJECT = {
    "Spine2": "spine", "LeftArm": "shoulder.L", "LeftForeArm": "forearm.L",
    "RightArm": "shoulder.R", "RightForeArm": "forearm.R",
}
JOINT_TAGS = {"Spine2": 9, "LeftArm": 37, "LeftForeArm": 38,
              "RightArm": 14, "RightForeArm": 15}
CHILD_TAGS = {"Spine2": 10, "LeftArm": 38, "LeftForeArm": 39,
              "RightArm": 15, "RightForeArm": 16}
COORDINATE_DESCRIPTION = {
    "up": "+Z", "front": "+Y", "handedness": "right",
    "rotation_direction": "counterclockwise", "unit": "metre",
    "quaternion_order": "wxyz", "sdk_quaternion_order": "xyzw",
    "joint_transforms": "local to parent; composed to world before retargeting",
    "sdk_render_preset": "Default", "sdk_unit": "centimetre",
    "sdk_to_project_basis": SDK_TO_PROJECT_BASIS.tolist(),
    "sdk_position_scale": 0.01,
    "retargeting": "world_rotation @ rotation(project_rest_vector, default_child_offset or local_child_position)",
    "root_translation_rendered": False,
    "heading_alignment": "optional fixed world-Z rotation; offset stored per frame",
}


def convert_noitom_position(position, basis=None):
    basis = np.eye(3) if basis is None else np.asarray(basis, dtype=float)
    position = np.asarray(position, dtype=float)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError("Invalid Noitom position")
    return basis @ position


def convert_noitom_rotation(xyzw, basis=None):
    q = validate_input_quaternion(from_xyzw(xyzw))
    if basis is None:
        return q
    basis = np.asarray(basis, dtype=float)
    if basis.shape != (3, 3) or not np.allclose(basis @ basis.T, np.eye(3)):
        raise ValueError("Coordinate basis must be orthogonal")
    return matrix_to_quaternion(basis @ quaternion_to_matrix(q) @ basis.T)


def sdk_position_to_project(position):
    x, y, z = position
    if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
        raise ValueError("Invalid Noitom position")
    return np.array((-x * .01, z * .01, y * .01))


def sdk_rotation_to_project(xyzw):
    # B is a fixed proper rotation, so conjugation maps the quaternion's vector
    # part by B. Avoid revalidating B / forming matrices for all 59 joints at 100 Hz.
    q = from_xyzw(xyzw)
    return validate_input_quaternion((q[0], -q[1], q[3], q[2]))


def rotation_between(source, target):
    source, target = np.asarray(source, float), np.asarray(target, float)
    if min(np.linalg.norm(source), np.linalg.norm(target)) < 1e-8:
        raise ValueError("Missing or zero-length SDK rest offset")
    a, b = source / np.linalg.norm(source), target / np.linalg.norm(target)
    dot = float(np.clip(a @ b, -1, 1))
    if dot < -1 + 1e-8:
        axis = np.cross(a, np.eye(3)[np.argmin(abs(a))])
        axis /= np.linalg.norm(axis)
        return quaternion_to_matrix(np.r_[0., axis])
    return quaternion_to_matrix(normalize_quaternion(np.r_[1 + dot, np.cross(a, b)]))


def world_transforms(frame: NoitomFrame, names=None):
    result, visiting = {}, set()

    def visit(name):
        if name in result:
            return result[name]
        if name in visiting:
            raise ValueError("Cycle in SDK joint hierarchy")
        visiting.add(name)
        joint = frame.joints[name]
        rotation = quaternion_to_matrix(validate_input_quaternion(joint.rotation))
        position = None if joint.position is None else joint.position.copy()
        if joint.parent is not None:
            if joint.parent not in frame.joints:
                raise ValueError(f"Missing SDK parent: {joint.parent}")
            parent_rotation, parent_position = visit(joint.parent)
            position = (None if position is None or parent_position is None
                        else parent_position + parent_rotation @ position)
            rotation = parent_rotation @ rotation
        visiting.remove(name)
        result[name] = (rotation, position)
        return result[name]

    for name in frame.joints if names is None else names:
        visit(name)
    return result


def torso_heading(frame: NoitomFrame):
    """World heading from the shoulder line, independent of selected sensors."""
    by_tag = {joint.tag: name for name, joint in frame.joints.items() if joint.tag is not None}
    left = by_tag.get(JOINT_TAGS["LeftArm"], "LeftArm")
    right = by_tag.get(JOINT_TAGS["RightArm"], "RightArm")
    if left not in frame.joints or right not in frame.joints:
        raise ValueError("Нет плеч Noitom для совмещения направления")
    worlds = world_transforms(frame, (left, right))
    lp, rp = worlds[left][1], worlds[right][1]
    if lp is None or rp is None:
        raise ValueError("Нет положений плеч Noitom")
    direction = rp - lp
    if not np.all(np.isfinite(direction)) or np.linalg.norm(direction[:2]) < 1e-6:
        raise ValueError("Невозможно определить направление корпуса: встаньте прямо")
    return math.atan2(direction[1], direction[0])


class NoitomAdapter:
    def __init__(self):
        self.rest_segments = compute_body_pose({}).tracked_segments

    def orientations(self, frame: NoitomFrame, enabled_segments=None, *, heading_offset_rad=0.0):
        if not math.isfinite(heading_offset_rad):
            raise ValueError("Invalid heading offset")
        cosine, sine = math.cos(heading_offset_rad), math.sin(heading_offset_rad)
        heading = np.array(((cosine, -sine, 0.), (sine, cosine, 0.), (0., 0., 1.)))
        enabled = set(NOITOM_TO_PROJECT.values()) if enabled_segments is None else set(enabled_segments)
        mapping = {joint: segment for joint, segment in NOITOM_TO_PROJECT.items() if segment in enabled}
        by_tag = {joint.tag: name for name, joint in frame.joints.items() if joint.tag is not None}
        names = [by_tag.get(JOINT_TAGS[name], name) for name in mapping]
        for name in names:
            if name not in frame.joints:
                raise ValueError(f"Noitom skeleton missing {name}")
        worlds = world_transforms(frame, names)
        result = {}
        for sdk_name, segment in mapping.items():
            name = by_tag.get(JOINT_TAGS[sdk_name], sdk_name)
            if name not in frame.joints:
                raise ValueError(f"Noitom skeleton missing {sdk_name}")
            child_names = {"Spine2": ("Spine3", "Neck", "Neck1", "Head"),
                           "LeftArm": ("LeftForeArm",), "LeftForeArm": ("LeftHand",),
                           "RightArm": ("RightForeArm",), "RightForeArm": ("RightHand",)}[sdk_name]
            child_tags = (59, 10, 11, 12) if sdk_name == "Spine2" else (CHILD_TAGS[sdk_name],)
            candidates = [by_tag[tag] for tag in child_tags if tag in by_tag]
            candidates.extend(child_names)
            child = next((key for key in candidates if key in frame.joints
                          and frame.joints[key].parent == name), None)
            offset = None if child is None else frame.joints[child].default_position
            if offset is None and child is not None:
                # Binary BVH exposes the child translation in its parent's frame
                # even when GetJointDefaultLocalPosition is NotSupported. The
                # world bone direction is R_parent @ this local translation.
                offset = frame.joints[child].position
            if offset is None:
                raise ValueError(f"No child offset for {sdk_name}; BVH transformation is required")
            start, end = self.rest_segments[segment]
            correction = rotation_between(end - start, offset)
            result[segment] = matrix_to_quaternion(heading @ worlds[name][0] @ correction)
        return result
