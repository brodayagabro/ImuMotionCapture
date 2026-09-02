#!/usr/bin/env python3
"""Offscreen end-to-end smoke test for the PyQt window and local UDP peer."""

from __future__ import annotations

import os
import socket
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from mpu_udp_viewer import MotionCaptureWindow


def pump(app: QApplication, duration_s: float) -> None:
    deadline = time.monotonic() + duration_s
    while time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.005)


def receive_command(
    app: QApplication, server: socket.socket
) -> tuple[bytes, tuple[str, int]]:
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        app.processEvents()
        try:
            return server.recvfrom(1024)
        except BlockingIOError:
            time.sleep(0.005)
    raise AssertionError("timeout waiting for GUI command")


def main() -> int:
    app = QApplication([])
    server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server.bind(("127.0.0.1", 0))
    server.setblocking(False)
    window = MotionCaptureWindow()
    window.config.device_ip = "127.0.0.1"
    window.config.device_port = server.getsockname()[1]
    window.config.stream_rate_hz = 23
    try:
        assert window.connect_device()
        hello, client = receive_command(app, server)
        assert hello == b"HELLO\n", hello
        status, status_client = receive_command(app, server)
        assert status == b"STATUS\n", status
        assert status_client == client

        window.start_stream()
        set_rate, rate_client = receive_command(app, server)
        start, start_client = receive_command(app, server)
        assert set_rate == b"SET_RATE 23\n", set_rate
        assert start == b"START\n", start
        assert rate_client == start_client == client

        lines = ["FRAME 1 100 5"]
        for sensor_id in (0, 1, 2, 6, 7):
            lines.append(f"Q {sensor_id} 1 0 0 0")
        server.sendto(("\n".join(lines) + "\n").encode("ascii"), client)
        pump(app, 0.25)
        assert window.model.sample_frames == 1
        assert not window.model.neutral_pending
        assert set(window.model.neutral_orientation) == set(window.config.sensor_mapping)
        window._refresh_plot()
        app.processEvents()
    finally:
        window.close()
        server.close()
        app.processEvents()
    print("GUI_UDP_SMOKE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
