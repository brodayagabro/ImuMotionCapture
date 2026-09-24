"""PyInstaller entry point for the host/viz/pyqt_mocap application."""

from PyQt6 import QtCore, QtGui, QtSvg, QtWidgets  # noqa: F401
from pyqt_mocap.mpu_udp_viewer import main


if __name__ == "__main__":
    raise SystemExit(main())
