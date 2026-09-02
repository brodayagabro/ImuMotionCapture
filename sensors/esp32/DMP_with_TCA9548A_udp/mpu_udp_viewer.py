#!/usr/bin/env python3
"""Compatibility launcher for the standalone :mod:`pyqt_mocap` package."""

from __future__ import annotations

from pyqt_mocap.mpu_udp_viewer import (
    AXIS_OPTIONS,
    DEFAULT_DEVICE_IP,
    DEFAULT_DEVICE_PORT,
    DEFAULT_RENDER_FPS,
    DEFAULT_STREAM_RATE_HZ,
    ID_MODE_TITLES,
    SEGMENT_TITLES,
    HumanCanvas,
    MotionCaptureWindow,
    SettingsDialog,
    ViewerConfig,
    default_config,
    load_config,
    main,
    save_config,
)


if __name__ == "__main__":
    raise SystemExit(main())
