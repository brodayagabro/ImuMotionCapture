"""OpenGL variant of the PyQt motion-capture application."""

from __future__ import annotations

import os
import sys

from PyQt6.QtWidgets import QApplication

from .human_canvas_gl import OpenGLHumanCanvas

os.environ.setdefault("PYQT_MOCAP_RENDERER", "opengl")

from .mpu_udp_viewer import MotionCaptureWindow


class OpenGLMotionCaptureWindow(MotionCaptureWindow):
    """Main window using PyQtGraph instead of Matplotlib for the skeleton."""

    canvas_class = OpenGLHumanCanvas

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("MPU6050 UDP — PyQtGraph OpenGL motion capture")


def main() -> int:
    app = QApplication(sys.argv)
    app.setOrganizationName("Neuromorph")
    app.setApplicationName("MPU UDP Viewer OpenGL")
    window = OpenGLMotionCaptureWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
