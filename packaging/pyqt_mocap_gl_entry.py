"""PyInstaller entry point for the PyQtGraph/OpenGL motion-capture app."""

import os

os.environ["PYQT_MOCAP_RENDERER"] = "opengl"

from PyQt6 import QtCore, QtGui, QtOpenGL, QtOpenGLWidgets, QtWidgets  # noqa: F401
from pyqt_mocap.mpu_udp_viewer_gl import main


if __name__ == "__main__":
    raise SystemExit(main())
