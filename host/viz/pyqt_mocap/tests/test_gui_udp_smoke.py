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


def test_window_command_and_frame_round_trip(tmp_path, monkeypatch) -> None:
    from PyQt6.QtCore import QSettings
    from pyqt_mocap import mpu_udp_viewer

    # Applying calibration saves configuration: never use the user's store.
    settings = QSettings(str(tmp_path / "viewer.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(mpu_udp_viewer, "QSettings", lambda *args: settings)
    app = QApplication.instance() or QApplication([])
    server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server.bind(("127.0.0.1", 0))
    server.setblocking(False)
    window = MotionCaptureWindow()
    assert window.config.device_ip == "192.168.1.117"
    assert window.config.device_port == 4210
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
        assert not window.bvh_active
        assert window.record_bvh_button.text() == "Начать запись"
        window.record_bvh_button.click()
        assert window.bvh_active
        assert window.record_bvh_button.text() == "Остановить запись"
        for frame_id in (2, 3):
            lines[0] = f"FRAME {frame_id} {frame_id * 100} 5"
            server.sendto(("\n".join(lines) + "\n").encode("ascii"), client)
            pump(app, .08)
        window.record_bvh_button.click()
        assert not window.bvh_active
        assert window.record_bvh_button.text() == "Начать запись"
        assert len(window.bvh_recording.frames) == 2
        assert window.save_bvh_action.isEnabled()
        assert window.save_bvh_button.isEnabled()
        monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", lambda *a: ("", ""))
        assert not window.save_bvh_recording()
        assert window.bvh_dirty
        path = tmp_path / "gui_recording.bvh"
        monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", lambda *a: (str(path), ""))
        assert window.save_bvh_recording()
        assert not window.bvh_dirty
        assert "ROOT Hips" in path.read_text()
        assert "MOTION\nFrames:" in path.read_text()
        from pyqt_mocap.semaphore_calibration import calibrate_semaphore
        from pyqt_mocap.tests.test_semaphore import semaphore_captures
        assert not hasattr(window, "neutral_action")
        window.semaphore_action.trigger()
        assert window.guided_dialog.semaphore
        assert len(window.guided_dialog.pose_names) == 7
        result = calibrate_semaphore(semaphore_captures(),
                                     enabled_segments=window.config.enabled_segments)
        window.guided_dialog.result_ready.emit(result)
        assert window.last_calibration_result is result
        assert window.calibration_document["calibration"]["method"] == "semaphore_xz"
        assert not window.model.neutral_pending
        window.guided_dialog.reject()
    finally:
        window.bvh_dirty = False  # Do not show a modal prompt on assertion failure.
        window.close()
        server.close()
        app.processEvents()
