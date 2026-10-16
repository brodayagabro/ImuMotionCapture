"""Compatibility launcher for the OpenGL motion-capture application."""

from __future__ import annotations

from .mpu_udp_viewer import MotionCaptureWindow as OpenGLMotionCaptureWindow, main


if __name__ == "__main__":
    raise SystemExit(main())
