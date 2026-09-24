"""Prepare the bundled Qt runtime before Matplotlib imports a binding."""

from __future__ import annotations

import os
from pathlib import Path
import sys


if hasattr(sys, "_MEIPASS"):
    qt_bin = Path(sys._MEIPASS) / "PyQt6" / "Qt6" / "bin"
    if qt_bin.is_dir():
        os.environ["PATH"] = str(qt_bin) + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            # Keep the handle alive; Windows removes this search path when the
            # returned object is garbage-collected.
            sys._pyqt_mocap_dll_directory = os.add_dll_directory(str(qt_bin))

os.environ["QT_API"] = "PyQt6"
