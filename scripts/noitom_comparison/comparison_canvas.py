"""Two independent mannequins sharing one OpenGL view and one root."""
import numpy as np
from PyQt6.QtWidgets import QLabel
from pyqtgraph.opengl import GLScatterPlotItem
from OpenGL.GL import (
    GL_BLEND, GL_CULL_FACE, GL_DEPTH_TEST, GL_FUNC_ADD, GL_ONE,
    GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
)
from pyqt_mocap.human_canvas_gl import (
    OpenGLHumanCanvas, _bone_item, _face_colors, _static_bone_segments, bone_mesh,
)
from pyqt_mocap.mocap_core import compute_body_pose, DEFAULT_ENABLED_SEGMENTS, SEGMENT_NAMES

OWN_COLOR = (0.10, 0.56, 0.90, 1.0)
NOITOM_COLOR = (0.90, 0.15, 0.15, 0.85)
NOITOM_DISABLED_COLOR = (0.70, 0.58, 0.58, 0.45)
# Reference wireframe remains visible even where the two mannequins coincide.
REFERENCE_GL_OPTIONS = {GL_DEPTH_TEST: False, GL_BLEND: True, GL_CULL_FACE: False,
                        "glBlendEquationSeparate": (GL_FUNC_ADD, GL_FUNC_ADD),
                        "glBlendFuncSeparate": (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
                                                GL_ONE, GL_ONE_MINUS_SRC_ALPHA)}


class ComparisonOpenGLCanvas(OpenGLHumanCanvas):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.own_bones = self.tracked_bones
        pose = compute_body_pose({})
        self.noitom_bones = {}
        self.noitom_static_bones = []
        for name, (start, end) in pose.tracked_segments.items():
            bone = self._reference_bone(start, end)
            self.noitom_bones[name] = bone
        for start, end in _static_bone_segments(pose):
            self.noitom_static_bones.append(self._reference_bone(start, end))
        self.noitom_joints = GLScatterPlotItem(pos=np.asarray(pose.joints),
            color=NOITOM_COLOR, size=5., glOptions=REFERENCE_GL_OPTIONS)
        self.noitom_head = GLScatterPlotItem(pos=np.asarray([pose.head_center]),
            color=NOITOM_COLOR, size=12., glOptions=REFERENCE_GL_OPTIONS)
        self.noitom_joints.setDepthValue(10)
        self.noitom_head.setDepthValue(10)
        self.view.addItem(self.noitom_joints)
        self.view.addItem(self.noitom_head)
        self.layout().insertWidget(1, QLabel(
            '<b style="color:#198fe6">Own IMU — синий</b> · '
            '<b style="color:#e62626">Noitom — красный каркас</b> · общий таз и длины сегментов'))
        self.set_noitom_visible(False)

    def _reference_bone(self, start, end):
        bone = _bone_item(start, end, NOITOM_COLOR)
        bone.setGLOptions(REFERENCE_GL_OPTIONS)
        bone.setDepthValue(10)
        bone.opts.update(drawFaces=False, edgeColor=NOITOM_COLOR)
        self.view.addItem(bone)
        return bone

    def set_noitom_visible(self, visible):
        for item in [*self.noitom_bones.values(), *self.noitom_static_bones,
                     self.noitom_joints, self.noitom_head]:
            item.setVisible(visible)

    def update_pose(self, orientations, sensor_mapping, enabled_segments=DEFAULT_ENABLED_SEGMENTS):
        super().update_pose(orientations, sensor_mapping, enabled_segments)
        for name, bone in self.tracked_bones.items():
            color = OWN_COLOR if name in enabled_segments else (.65, .75, .83, 1.)
            bone.opts["color"] = color
            bone.opts["meshdata"].setFaceColors(_face_colors(color, 8))
            bone.meshDataChanged()

    update_own_pose = update_pose

    def update_noitom_pose(self, orientations, enabled_segments=SEGMENT_NAMES):
        enabled = frozenset(enabled_segments)
        pose = compute_body_pose({name: rotation for name, rotation in orientations.items() if name in enabled})
        for name, (start, end) in pose.tracked_segments.items():
            vertices, faces = bone_mesh(start, end, pose.axis_frames[name])
            self.noitom_bones[name].opts["edgeColor"] = NOITOM_COLOR if name in enabled else NOITOM_DISABLED_COLOR
            self.noitom_bones[name].setMeshData(vertexes=vertices, faces=faces)
        for bone, (start, end) in zip(self.noitom_static_bones, _static_bone_segments(pose), strict=True):
            vertices, faces = bone_mesh(start, end)
            bone.setMeshData(vertexes=vertices, faces=faces)
        self.noitom_joints.setData(pos=np.asarray(pose.joints, dtype=np.float32))
        self.noitom_head.setData(pos=np.asarray([pose.head_center], dtype=np.float32))
        self.set_noitom_visible(True)
