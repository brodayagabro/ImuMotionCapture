"""GPU-accelerated PyQtGraph/OpenGL skeleton renderer."""

from __future__ import annotations

from collections.abc import Collection, Mapping

import numpy as np
from PyQt6.QtGui import QColor, QFont, QVector3D
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget
from pyqtgraph.opengl import (
    GLGridItem,
    GLLinePlotItem,
    GLMeshItem,
    GLScatterPlotItem,
    GLTextItem,
    GLViewWidget,
)

from .mocap_core import (
    DEFAULT_ENABLED_SEGMENTS,
    DEFAULT_SENSOR_MAPPING,
    SEGMENT_NAMES,
    compute_body_pose,
)


TRACKED_COLORS = {
    "spine": (0.88, 0.55, 0.41, 1.0),
    "shoulder.L": (0.88, 0.55, 0.41, 1.0),
    "forearm.L": (0.94, 0.68, 0.53, 1.0),
    "shoulder.R": (0.88, 0.55, 0.41, 1.0),
    "forearm.R": (0.94, 0.68, 0.53, 1.0),
}
AXIS_COLORS = (
    (0.84, 0.17, 0.17, 1.0),
    (0.16, 0.62, 0.33, 1.0),
    (0.15, 0.46, 0.82, 1.0),
)


def _segment_positions(segments) -> np.ndarray:
    """Flatten line segment pairs into the layout used by GL_LINES."""
    positions = [point for start, end in segments for point in (start, end)]
    return np.asarray(positions, dtype=np.float32).reshape((-1, 3))


def bone_mesh(start, end, frame=None):
    """Blender-like octahedron: joint tips and a square near the bone head.

    The frame keeps the cross-section attached to the moving segment. All
    coordinates remain in the model's space; this changes rendering only.
    """
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    delta = end - start
    length = float(np.linalg.norm(delta))
    if not np.isfinite(length) or length < 1e-8:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint32)
    direction = delta / length
    basis = np.eye(3) if frame is None else np.asarray(frame, dtype=float)
    reference = basis[:, np.argmin(np.abs(direction @ basis))]
    u = np.cross(direction, reference)
    u /= np.linalg.norm(u)
    v = np.cross(direction, u)
    radius = min(length * .12, .045)
    center = start + delta * .20
    vertices = np.asarray((start, end, center + radius * u, center + radius * v,
                           center - radius * u, center - radius * v), dtype=np.float32)
    faces = []
    for index in range(4):
        a, b = 2 + index, 2 + (index + 1) % 4
        faces.extend(((0, b, a), (1, a, b)))
    return vertices, np.asarray(faces, dtype=np.uint32)


def _static_bone_segments(pose):
    """Split the model's shoulder bar into two outward-facing clavicles."""
    left, right = pose.static_segments[1]
    center = (left + right) * .5
    return (pose.static_segments[0], (center, left), (center, right),
            *pose.static_segments[2:])


def _face_colors(color, count):
    # Stable facet shading, independent of driver-specific scene lighting.
    colors = np.tile(color, (count, 1)).astype(np.float32)
    shades = np.resize(np.array((1., .90, .78, .70, .88, .80, .96, .85)), count)
    colors[:, :3] *= shades[:, None]
    return colors


def _bone_item(start, end, color):
    vertices, faces = bone_mesh(start, end)
    return GLMeshItem(vertexes=vertices, faces=faces, color=color,
                      faceColors=_face_colors(color, len(faces)),
                      smooth=False, computeNormals=False, drawEdges=True,
                      edgeColor=(.18, .24, .30, 1.), glOptions="opaque")


class OpenGLHumanCanvas(QWidget):
    """Interactive OpenGL skeleton view with the same API as HumanCanvas."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        heading = QHBoxLayout()
        title = QLabel("<b>Скелетная модель · OpenGL · Octahedral</b>")
        title.setAccessibleName("Скелетная модель OpenGL")
        legend = QLabel(
            '<span style="color:#d52b2b">X</span> · '
            '<span style="color:#2a9d55">Y</span> · '
            '<span style="color:#2676d2">Z</span> · '
            "мышь: вращение и масштаб"
        )
        heading.addWidget(title)
        heading.addStretch(1)
        heading.addWidget(legend)
        layout.addLayout(heading)

        self.view = GLViewWidget(rotationMethod="quaternion")
        self.view.setAccessibleName("Трёхмерная OpenGL-визуализация человека")
        self.view.setBackgroundColor(QColor("#f3f6fa"))
        self.view.opts["center"] = QVector3D(0.0, 0.0, 0.92)
        self.view.setCameraPosition(distance=3.3, elevation=11.0, azimuth=-90.0)
        layout.addWidget(self.view, 1)

        self.tracked_bones: dict[str, GLMeshItem] = {}
        self.static_bones: list[GLMeshItem] = []
        self.axis_lines: tuple[GLLinePlotItem, GLLinePlotItem, GLLinePlotItem]
        self.segment_labels: dict[str, GLTextItem] = {}
        self._create_scene()

    def _create_scene(self) -> None:
        pose = compute_body_pose({})

        grid = GLGridItem(
            size=QVector3D(1.4, 0.9, 1.0),
            color=QColor(174, 185, 199, 115),
            antialias=True,
        )
        grid.setSpacing(0.1, 0.1, 1.0)
        self.view.addItem(grid)

        for name in SEGMENT_NAMES:
            start, end = pose.tracked_segments[name]
            bone = _bone_item(start, end, TRACKED_COLORS[name])
            self.tracked_bones[name] = bone
            self.view.addItem(bone)

        for start, end in _static_bone_segments(pose):
            bone = _bone_item(start, end, (0.38, 0.46, 0.55, 1.0))
            self.static_bones.append(bone)
            self.view.addItem(bone)

        self.joints = GLScatterPlotItem(
            pos=np.asarray(pose.joints, dtype=np.float32),
            color=(0.15, 0.22, 0.30, 1.0),
            size=8.0,
            pxMode=True,
            glOptions="translucent",
        )
        self.view.addItem(self.joints)
        self.head = GLScatterPlotItem(
            pos=np.asarray((pose.head_center,), dtype=np.float32),
            color=(0.94, 0.96, 0.98, 1.0),
            size=25.0,
            pxMode=True,
            glOptions="translucent",
        )
        self.view.addItem(self.head)

        axis_items: list[GLLinePlotItem] = []
        for color in AXIS_COLORS:
            item = GLLinePlotItem(
                pos=np.empty((0, 3), dtype=np.float32),
                color=color,
                width=2.0,
                mode="lines",
                antialias=True,
                glOptions="translucent",
            )
            axis_items.append(item)
            self.view.addItem(item)
        self.axis_lines = (axis_items[0], axis_items[1], axis_items[2])

        label_font = QFont("Segoe UI", 9)
        for name in SEGMENT_NAMES:
            origin = pose.axis_origins[name]
            label = GLTextItem(
                pos=np.asarray(origin, dtype=float),
                color=QColor("#172535"),
                text=name,
                font=label_font,
                glOptions="translucent",
            )
            self.segment_labels[name] = label
            self.view.addItem(label)

        self.update_pose({}, DEFAULT_SENSOR_MAPPING)

    def update_pose(
        self,
        orientations: Mapping[str, object],
        sensor_mapping: Mapping[str, int],
        enabled_segments: Collection[str] = DEFAULT_ENABLED_SEGMENTS,
    ) -> None:
        pose = compute_body_pose(orientations)  # type: ignore[arg-type]
        enabled = set(enabled_segments)

        for name, (start, end) in pose.tracked_segments.items():
            color = list(TRACKED_COLORS[name])
            if name not in enabled:
                # Opaque muted bones keep depth testing reliable.
                color = [0.70, 0.74, 0.79, 1.0]
            vertices, faces = bone_mesh(start, end, pose.axis_frames[name])
            self.tracked_bones[name].setMeshData(
                vertexes=vertices, faces=faces,
                color=tuple(color),
                faceColors=_face_colors(color, len(faces)),
            )

        for bone, (start, end) in zip(self.static_bones, _static_bone_segments(pose), strict=True):
            vertices, faces = bone_mesh(start, end)
            bone.setMeshData(vertexes=vertices, faces=faces,
                             faceColors=_face_colors(bone.opts["color"], len(faces)))
        self.joints.setData(pos=np.asarray(pose.joints, dtype=np.float32))
        self.head.setData(pos=np.asarray((pose.head_center,), dtype=np.float32))

        axis_positions: list[list[np.ndarray]] = [[], [], []]
        axis_vertex_colors: list[list[tuple[float, float, float, float]]] = [
            [], [], []
        ]
        for name in SEGMENT_NAMES:
            active = name in enabled
            show_axes = active or name == "spine"
            label = self.segment_labels[name]
            if not show_axes:
                label.setData(text="")
                continue

            origin = pose.axis_origins[name]
            frame = pose.axis_frames[name]
            alpha = 1.0 if active else 0.55
            for axis_index in range(3):
                end = origin + frame[:, axis_index] * 0.17
                axis_positions[axis_index].extend((origin, end))
                base = AXIS_COLORS[axis_index]
                color = (base[0], base[1], base[2], alpha)
                axis_vertex_colors[axis_index].extend((color, color))
            label.setData(
                pos=np.asarray(origin + (0.025, 0.025, 0.035), dtype=float),
                text=f"{name}  [S{sensor_mapping[name]}]",
                color=QColor("#172535") if active else QColor("#6b7785"),
            )

        for axis_index, item in enumerate(self.axis_lines):
            item.setData(
                pos=np.asarray(axis_positions[axis_index], dtype=np.float32).reshape((-1, 3)),
                color=np.asarray(axis_vertex_colors[axis_index], dtype=np.float32).reshape((-1, 4)),
            )
        self.view.update()
