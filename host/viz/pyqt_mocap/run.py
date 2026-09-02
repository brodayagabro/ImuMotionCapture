#!/usr/bin/env python3
"""Direct launcher that also works as ``python pyqt_mocap/run.py``."""

from __future__ import annotations

from pathlib import Path
import sys


PACKAGE_PARENT = str(Path(__file__).resolve().parent.parent)
if PACKAGE_PARENT not in sys.path:
    sys.path.insert(0, PACKAGE_PARENT)

from pyqt_mocap.mpu_udp_viewer import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
