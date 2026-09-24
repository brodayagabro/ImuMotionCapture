"""GPU-accelerated PyQtGraph/OpenGL skeleton renderer."""

from __future__ import annotations

from collections.abc import Collection, Mapping

import numpy as np
from PyQt6.QtGui import QColor, QFont, QVector3D
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget
from pyqtgraph.opengl import (
    GLGridItem,
    GLLinePlotItem,
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
TRACKED_WIDTHS = {
    "spine": 5.0,
    "shoulder.L": 5.0,
    "forearm.L": 4.0,
    "shoulder.R": 5.0,
    "forearm.R": 4.0,
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


class OpenGLHumanCanvas(QWidget):
    """Interactive OpenGL skeleton view with the same API as HumanCanvas."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        heading = QHBoxLayout()
        title = QLabel("<b>Скелетная модель · OpenGL</b>")
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

        self.tracked_lines: dict[str, GLLinePlotItem] = {}
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
            line = GLLinePlotItem(
                pos=np.asarray((start, end), dtype=np.float32),
                color=TRACKED_COLORS[name],
                width=TRACKED_WIDTHS[name],
                mode="lines",
                antialias=True,
                glOptions="translucent",
            )
            self.tracked_lines[name] = line
            self.view.addItem(line)

        self.static_lines = GLLinePlotItem(
            pos=_segment_positions(pose.static_segments),
            color=(0.25, 0.33, 0.42, 1.0),
            width=3.0,
            mode="lines",
            antialias=True,
            glOptions="translucent",
        )
        self.view.addItem(self.static_lines)

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
                color[3] = 0.22
            self.tracked_lines[name].setData(
                pos=np.asarray((start, end), dtype=np.float32),
                color=tuple(color),
            )

        self.static_lines.setData(pos=_segment_positions(pose.static_segments))
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
