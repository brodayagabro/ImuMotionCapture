"""Offscreen PyQt and loopback-UDP integration test."""

from __future__ import annotations

import socket
import time

import pytest

QtWidgets = pytest.importorskip("PyQt6.QtWidgets")
QApplication = QtWidgets.QApplication

from pyqt_mocap.mpu_udp_viewer import MotionCaptureWindow
from pyqt_mocap.mocap_core import SEGMENT_NAMES


pytestmark = pytest.mark.gui


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


def test_window_command_and_frame_round_trip() -> None:
    app = QApplication.instance() or QApplication([])
    server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server.bind(("127.0.0.1", 0))
    server.setblocking(False)
    window = MotionCaptureWindow()
    window.config.device_ip = "127.0.0.1"
    window.config.device_port = server.getsockname()[1]
    window.config.stream_rate_hz = 23
    try:
        assert not hasattr(window.human_canvas, "torso")
        assert window.human_canvas.axes.azim == 108
        assert (
            window.human_canvas.tracked_lines["spine"].get_color()
            == window.human_canvas.tracked_lines["shoulder.L"].get_color()
        )
        assert all(
            line.get_visible()
            for line in window.human_canvas.axis_lines["spine"]
        )
        assert window.human_canvas.segment_labels["spine"].get_visible()
        assert window.connect_device()
        hello, client = receive_command(app, server)
        status, status_client = receive_command(app, server)
        assert hello == b"HELLO\n"
        assert status == b"STATUS\n"
        assert status_client == client

        window.start_stream()
        set_rate, rate_client = receive_command(app, server)
        start, start_client = receive_command(app, server)
        assert set_rate == b"SET_RATE 23\n"
        assert start == b"START\n"
        assert rate_client == start_client == client

        lines = ["FRAME 1 100 5"]
        for sensor_id in (0, 1, 2, 6, 7):
            lines.append(f"Q {sensor_id} 1 0 0 0")
        server.sendto(("\n".join(lines) + "\n").encode("ascii"), client)
        pump(app, 0.25)
        assert window.model.sample_frames == 1
        assert not window.model.neutral_pending
        assert set(window.model.neutral_orientation) == set(SEGMENT_NAMES)
    finally:
        window.close()
        server.close()
        app.processEvents()
