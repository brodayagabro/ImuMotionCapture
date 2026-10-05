import base64
from dataclasses import replace
import json
import socket
import time
import numpy as np
import pytest
from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QApplication
from pyqt_mocap import mpu_udp_viewer
from pyqt_mocap.mocap_core import DEFAULT_SENSOR_MAPPING, compute_body_pose, quaternion_from_rotation_vector
from scripts.noitom_comparison.comparison_canvas import (
    ComparisonOpenGLCanvas, OWN_COLOR, NOITOM_COLOR, NOITOM_DISABLED_COLOR,
)
from scripts.noitom_comparison.comparison_window import MocapComparisonWindow
from scripts.noitom_comparison.config import ComparisonConfig
from scripts.noitom_comparison.noitom_client import FakeNoitomClient, NoitomClient
from scripts.noitom_comparison.replay import ReplayWindow, session_frames

pytestmark = pytest.mark.gui


def pump_until(app, predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return
        time.sleep(.005)
    raise AssertionError("Timed out waiting for condition")


@pytest.mark.parametrize('fail_first', [False, True])
def test_calibration_button_runs_only_a_t_a_and_applies_profile(tmp_path, monkeypatch, fail_first):
    from types import SimpleNamespace
    from pyqt_mocap import guided_dialog
    from pyqt_mocap.t_pose_calibration import T_POSE_NAMES
    from pyqt_mocap.tests.test_t_pose_calibration import physical_captures
    from pyqt_mocap.mocap_core import matrix_to_quaternion, quaternion_to_matrix
    from pyqt_mocap.tests.test_calibration import rotation_x
    from scripts.diagnostics.orientation.calibration_analyzer import events_from_session
    from PyQt6.QtWidgets import QMessageBox

    monkeypatch.setattr(mpu_udp_viewer, "QSettings", lambda *a: QSettings(str(tmp_path / "viewer.ini"), QSettings.Format.IniFormat))
    app = QApplication.instance() or QApplication([])
    clock = [100.]
    monkeypatch.setattr(guided_dialog, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    captures = physical_captures()
    if fail_first:
        captures['return_a_pose'].average['forearm.R'] = matrix_to_quaternion(
            rotation_x(np.radians(40.3)) @ quaternion_to_matrix(captures['a_pose'].average['forearm.R']))
    monkeypatch.setattr(QMessageBox, 'warning', lambda *a: None)
    window = MocapComparisonWindow(ComparisonConfig(fake=True, recordings_dir=str(tmp_path / 'sessions')))
    server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server.bind(("127.0.0.1", 0))
    window.config.device_ip = "127.0.0.1"
    window.config.device_port = server.getsockname()[1]
    generation = [0]

    def snapshot():
        generation[0] += 1
        stage = window.guided_dialog.stage_index
        pose = captures[T_POSE_NAMES[stage]]
        return {s: (q, clock[0], generation[0]) for s, q in pose.average.items()}

    monkeypatch.setattr(window, "_calibration_snapshot", snapshot)
    try:
        assert window.connect_device()
        window.start_stream()
        assert window.calibration_button.text() == 'Калибровка'
        assert not window.guided_action.isVisible()
        assert not window.semaphore_action.isVisible()
        window.start_experiment_recording()
        session = window.session
        window.calibration_button.click()
        dialog = window.guided_dialog
        assert dialog.pose_names == T_POSE_NAMES
        assert dialog.windowTitle() == 'Калибровка A → T → A'
        assert dialog.title_label.text().startswith('1/3')
        dialog.capture_button.click()
        for stage in range(3):
            assert dialog.stage_index == stage
            assert dialog.title_label.text().startswith(f'{stage + 1}/3')
            clock[0] = dialog.phase_started_s + 5.
            dialog._tick()
            start = clock[0]
            for index in range(1, 51):
                clock[0] = start + index * .1
                dialog._tick()
        if fail_first:
            assert dialog.phase == 'idle' and window.calibration_document is None
            captures['return_a_pose'].average['forearm.R'] = captures['a_pose'].average['forearm.R'].copy()
            dialog.capture_button.click()
            clock[0] = dialog.phase_started_s + 5.
            dialog._tick()
            start = clock[0]
            for index in range(1, 51):
                clock[0] = start + index * .1
                dialog._tick()
        window.stop_experiment_recording()
        pump_until(app, session.done.is_set)
        assert session.error is None
        events = events_from_session(session.path)
        poses = [e['details'] for e in events if e['event'] == 'own_calibration_pose_captured']
        assert len(poses) == (4 if fail_first else 3)
        assert all('average_quaternions_wxyz' in e and 'sensor_mapping' in e for e in poses)
        failures = [e['details'] for e in events if e['event'] == 'own_calibration_fit_rejected']
        assert len(failures) == int(fail_first)
        if fail_first:
            assert failures[0]['closure_error_deg']['forearm.R'] == pytest.approx(40.3)
            assert set(failures[0]['poses']) == set(T_POSE_NAMES)
        assert dialog.phase == 'finished'
        assert not dialog.timer.isActive()
        assert tuple(dialog.captures) == T_POSE_NAMES
        assert window.last_calibration_result.method == 'a_t_a_sensor_y'
        assert window.calibration_document['calibration']['method'] == 'a_t_a_sensor_y'
        assert tuple(window.calibration_document['calibration']['poses']) == T_POSE_NAMES
        assert not window.model.neutral_pending
        assert window.last_calibration_result.max_drift_deg_s == pytest.approx(0.)
        assert not window.model.drift_compensation_enabled
        before_reference = {s:q.copy() for s,q in window.model.neutral_orientation.items()}
        window.stop_stream()
        window.start_stream()
        assert not window.model.neutral_pending
        for segment, reference in before_reference.items():
            np.testing.assert_allclose(window.model.neutral_orientation[segment], reference, atol=1e-12)
        from pyqt_mocap import window_calibration
        from pyqt_mocap.calibration import save_profile
        profile_path = tmp_path / 'calibration.json'
        save_profile(profile_path, window.calibration_document)
        monkeypatch.setattr(window_calibration.QFileDialog, 'getOpenFileName', lambda *a: (str(profile_path), ''))
        window.model.request_neutral()
        window.import_calibration_profile()
        assert not window.model.neutral_pending
        for segment, reference in before_reference.items():
            np.testing.assert_allclose(window.model.neutral_orientation[segment], reference, atol=1e-12)
        dialog.accept()
    finally:
        window.bvh_dirty = False
        window.close()
        pump_until(app, lambda: window._allow_close)
        server.close()


def test_two_independent_skeleton_mesh_sets_share_root():
    app = QApplication.instance() or QApplication([])
    canvas = ComparisonOpenGLCanvas()
    try:
        own = {"shoulder.L": quaternion_from_rotation_vector((0, 0.4, 0))}
        other = {"shoulder.L": quaternion_from_rotation_vector((0, -.8, 0))}
        items = tuple(canvas.view.items)
        canvas.update_own_pose(own, DEFAULT_SENSOR_MAPPING)
        canvas.update_noitom_pose(other)
        assert items == tuple(canvas.view.items)
        assert set(canvas.own_bones) == set(canvas.noitom_bones)
        for bones, pose in ((canvas.own_bones, compute_body_pose(own)),
                            (canvas.noitom_bones, compute_body_pose(other))):
            for name, bone in bones.items():
                np.testing.assert_allclose(bone.opts["meshdata"].vertexes()[:2], pose.tracked_segments[name], atol=1e-7)
        np.testing.assert_allclose(canvas.own_bones["spine"].opts["meshdata"].vertexes()[0],
                                   canvas.noitom_bones["spine"].opts["meshdata"].vertexes()[0])
        assert canvas.own_bones["shoulder.L"].opts["color"] == OWN_COLOR
        assert canvas.noitom_bones["shoulder.L"].opts["edgeColor"] == NOITOM_COLOR
        assert not canvas.noitom_bones["shoulder.L"].opts["drawFaces"]
    finally:
        canvas.close()


def test_udp_timestamp_pending_stop_repeated_sessions_and_source_isolation(tmp_path, monkeypatch):
    settings = QSettings(str(tmp_path / "viewer.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(mpu_udp_viewer, "QSettings", lambda *args: settings)
    app = QApplication.instance() or QApplication([])
    server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server.bind(("127.0.0.1", 0))
    server.settimeout(1)
    window = MocapComparisonWindow(ComparisonConfig(fake=True, recordings_dir=str(tmp_path / "sessions")))
    window.config.device_ip = "127.0.0.1"
    window.config.device_port = server.getsockname()[1]
    try:
        assert window.connect_device()
        hello, client_address = server.recvfrom(1024)
        assert hello == b"HELLO\n"
        window.start_stream()
        window.noitom_worker.command("connect")
        window.noitom_worker.command("start")
        pump_until(app, lambda: window.noitom_stats.frames >= 3)
        assert not window.noitom_error
        assert window.human_canvas.noitom_bones["spine"].visible()
        # Stop both GUI timers: acquisition/recording of Noitom continues, and own
        # packets must retain their received times while waiting in the Qt queue.
        window.event_timer.stop()
        window.render_timer.stop()
        window.start_experiment_recording()
        session = window.session
        payload = b"FRAME 17 123456 5\nQ 0 1 0 0 0\nQ 1 1 0 0 0\nQ 2 1 0 0 0\nQ 6 1 0 0 0\nQ 7 1 0 0 0\n"
        server.sendto(payload, client_address)
        deadline = time.monotonic() + 1
        while session.pending < 1 and time.monotonic() < deadline:
            time.sleep(.005)
        assert session.pending >= 1
        time.sleep(.08)
        window.stop_experiment_recording()
        assert not session.done.is_set()
        processing_ns = time.monotonic_ns()
        window._process_events()
        pump_until(app, session.done.is_set)
        frame = json.loads((session.path / "own_frames.jsonl").read_text().splitlines()[0])
        assert frame["frame_index"] == 17
        assert frame["host_timestamp_ns"] < processing_ns - 50_000_000
        assert frame["relative_timestamp_ns"] == frame["host_timestamp_ns"] - session.t0_ns
        assert window.model.latest_samples[0].received_s == frame["host_timestamp_ns"] / 1e9
        packet = json.loads((session.path / "own_packets.jsonl").read_text().splitlines()[0])
        assert base64.b64decode(packet["payload_base64"]) == payload
        assert frame["host_timestamp_ns"] == packet["host_timestamp_ns"]
        assert len((session.path / "noitom_frames.jsonl").read_text().splitlines()) >= 2
        assert session.metadata["status"] == "complete"
        times = [row[0] for row in session_frames(session.path)]
        assert times == sorted(times)

        window.event_timer.start(10)
        pump_until(app, lambda: window.experiment_button.isEnabled())
        window.start_experiment_recording()
        second = window.session
        window.noitom_worker.command("disconnect")
        pump_until(app, lambda: window.noitom_state == "DISCONNECTED")
        before = window.model.sample_frames
        server.sendto(payload.replace(b"17 ", b"18 "), client_address)
        pump_until(app, lambda: window.model.sample_frames > before)
        window.stop_experiment_recording()
        pump_until(app, second.done.is_set)
        assert "noitom_disconnect" in second.metadata["incomplete_reasons"]

        window.noitom_worker.command("connect")
        window.noitom_worker.command("start")
        pump_until(app, lambda: window.noitom_state == "STREAMING")
        window.disconnect_device()
        count = window.noitom_stats.frames
        pump_until(app, lambda: window.noitom_stats.frames > count + 2)
    finally:
        window.event_timer.start(10)
        window.bvh_dirty = False
        window.close()
        pump_until(app, lambda: window._allow_close)
        server.close()


def test_missing_sdk_error_does_not_close_window(tmp_path, monkeypatch):
    monkeypatch.setattr(mpu_udp_viewer, "QSettings", lambda *a: QSettings(str(tmp_path / "viewer.ini"), QSettings.Format.IniFormat))
    app = QApplication.instance() or QApplication([])
    window = MocapComparisonWindow(ComparisonConfig(sdk_library=str(tmp_path / "missing.dll")))
    try:
        window.noitom_worker.command("connect")
        pump_until(app, lambda: window.noitom_state == "ERROR")
        assert "sdk-library" in window.noitom_error
        assert window.noitom_worker.is_alive()
        assert not window._closing_requested
    finally:
        window.close()
        pump_until(app, lambda: window._allow_close)


def test_noitom_selection_during_capture_persists_records_and_replays(tmp_path, monkeypatch):
    settings = QSettings(str(tmp_path / "viewer.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(mpu_udp_viewer, "QSettings", lambda *args: settings)
    app = QApplication.instance() or QApplication([])

    class BentSpineClient(FakeNoitomClient):
        @staticmethod
        def frame(index, timestamp_ns):
            frame = FakeNoitomClient.frame(index, timestamp_ns)
            return replace(frame, joints={**frame.joints, "Spine2": replace(frame.joints["Spine2"],
                rotation=quaternion_from_rotation_vector((.5, .3, 0)))})

    window = MocapComparisonWindow(ComparisonConfig(fake=True, recordings_dir=str(tmp_path / "sessions")),
                                   client=BentSpineClient())
    reopened = replay = None
    try:
        assert window.noitom_enabled_segments == set(DEFAULT_SENSOR_MAPPING) - {"spine"}
        own_config = window.config.copy()
        window.noitom_worker.command("connect")
        window.noitom_worker.command("start")
        pump_until(app, lambda: window.noitom_stats.frames >= 3)
        window.open_settings()
        dialog = window.settings_dialog
        assert not dialog.noitom_checks["spine"].isChecked()
        dialog.noitom_checks["spine"].setChecked(True)
        # Noitom-only settings must not stop the separate own-system BVH capture.
        window.bvh_active = True
        monkeypatch.setattr(window, "stop_bvh_recording", lambda: pytest.fail("Own BVH was stopped"))
        assert dialog._apply()
        assert window.bvh_active and window.config == own_config
        window.bvh_active = False
        pump_until(app, lambda: "spine" in window.noitom_pose)
        window._refresh_plot()
        neutral_spine = compute_body_pose({}).tracked_segments["spine"]
        assert not np.allclose(window.human_canvas.noitom_bones["spine"].opts["meshdata"].vertexes()[:2], neutral_spine)

        window.start_experiment_recording()
        session = window.session
        initial = window.noitom_stats.frames
        pump_until(app, lambda: window.noitom_stats.frames > initial + 2)
        dialog.noitom_checks["spine"].setChecked(False)
        assert dialog._apply()
        np.testing.assert_allclose(window.human_canvas.noitom_bones["spine"].opts["meshdata"].vertexes()[:2], neutral_spine, atol=1e-7)
        pump_until(app, lambda: "spine" not in window.noitom_pose)
        assert len(window.noitom_pose) == 4
        assert window.human_canvas.noitom_bones["spine"].opts["edgeColor"] == NOITOM_DISABLED_COLOR

        for checkbox in dialog.noitom_checks.values():
            checkbox.setChecked(False)
        assert dialog._apply()
        pump_until(app, lambda: window.noitom_pose == {})
        window._refresh_plot()
        assert not window.noitom_error
        for name, bone in window.human_canvas.noitom_bones.items():
            np.testing.assert_allclose(bone.opts["meshdata"].vertexes()[:2], compute_body_pose({}).tracked_segments[name], atol=1e-7)
        window.stop_experiment_recording()
        pump_until(app, session.done.is_set)
        saved = [json.loads(line) for line in (session.path / "noitom_frames.jsonl").read_text().splitlines()]
        assert {len(row["enabled_segments"]) for row in saved} == {0, 4, 5}
        assert all(set(row["enabled_segments"]) == set(row["segment_orientations"]) for row in saved)
        assert all("Spine2" in row["joints"] for row in saved)
        assert len(session.metadata["noitom"]["enabled_segments"]) == 5
        assert "noitom_enabled_segments_changed" in (session.path / "events.csv").read_text()
        assert "noitom_mapping_error" not in session.metadata["incomplete_reasons"]
        # Cancel does not apply a checkbox edit, including to the saved defaults.
        dialog.noitom_checks["spine"].setChecked(True)
        dialog.reject()
        assert window.noitom_enabled_segments == frozenset()
        window.close()
        pump_until(app, lambda: window._allow_close)

        reopened = MocapComparisonWindow(ComparisonConfig(fake=True))
        assert reopened.noitom_enabled_segments == frozenset()
        assert reopened.config == own_config
        replay = ReplayWindow(session.path)
        replay.t0 -= 10_000_000_000
        replay.tick()
        assert replay.next_frame is None
        for name, bone in replay.canvas.noitom_bones.items():
            np.testing.assert_allclose(bone.opts["meshdata"].vertexes()[:2], compute_body_pose({}).tracked_segments[name], atol=1e-7)
            assert bone.opts["edgeColor"] == NOITOM_DISABLED_COLOR
    finally:
        window.bvh_active = False
        for candidate in (window, reopened):
            if candidate is not None:
                candidate.close()
                pump_until(app, lambda: candidate._allow_close)
        if replay is not None:
            replay.close()


def test_heading_control_records_fixed_transform_and_replay_matches(tmp_path, monkeypatch):
    settings = QSettings(str(tmp_path / "viewer.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(mpu_udp_viewer, "QSettings", lambda *args: settings)
    app = QApplication.instance() or QApplication([])
    root_rotation = quaternion_from_rotation_vector((0, 0, -.9))

    class TurnedClient(FakeNoitomClient):
        @staticmethod
        def frame(index, timestamp_ns):
            frame = FakeNoitomClient.frame(index, timestamp_ns)
            return replace(frame, joints={**frame.joints,
                "Hips": replace(frame.joints["Hips"], rotation=root_rotation)})

    window = MocapComparisonWindow(ComparisonConfig(fake=True, recordings_dir=str(tmp_path / "sessions")),
                                   client=TurnedClient())
    replay = None
    try:
        assert not window.align_heading_button.isEnabled()
        window.noitom_worker.command("connect")
        window.noitom_worker.command("start")
        pump_until(app, lambda: window.noitom_stats.frames >= 3)
        assert not window.align_heading_button.isEnabled()  # own data/calibration absent
        window.streaming_requested = True
        window.model.set_guided_calibration(window.config.axis_maps,
            {name: (0., 0., 0.) for name in DEFAULT_SENSOR_MAPPING},
            {name: (1., 0., 0., 0.) for name in DEFAULT_SENSOR_MAPPING}, time.monotonic())
        window.own_stats.observe(time.monotonic_ns())
        window._update_comparison_status()
        assert window.align_heading_button.isEnabled()
        window.start_experiment_recording()
        session = window.session
        count = window.noitom_stats.frames
        pump_until(app, lambda: window.noitom_stats.frames > count + 2)
        window.align_heading_button.click()
        assert window.noitom_worker.heading_offset_rad == pytest.approx(.9)
        assert window._metadata()["noitom"]["heading_offset_rad"] == pytest.approx(.9)
        count = window.noitom_stats.frames
        pump_until(app, lambda: window.noitom_stats.frames > count + 2)
        window.stop_experiment_recording()
        pump_until(app, session.done.is_set)
        saved = [json.loads(line) for line in (session.path / "noitom_frames.jsonl").read_text().splitlines()]
        assert {round(row["heading_offset_rad"], 6) for row in saved} == {0., .9}
        assert "noitom_heading_aligned" in (session.path / "events.csv").read_text()
        np.testing.assert_allclose(saved[-1]["joints"]["Hips"]["rotation"], root_rotation)
        replay = ReplayWindow(session.path)
        replay.t0 -= 10_000_000_000
        replay.tick()
        expected = compute_body_pose(saved[-1]["segment_orientations"])
        for name, bone in replay.canvas.noitom_bones.items():
            np.testing.assert_allclose(bone.opts["meshdata"].vertexes()[:2], expected.tracked_segments[name], atol=1e-7)
        window.own_stats.last_frame_ns = time.monotonic_ns() - 2 * window.options.stale_timeout_ms * 1_000_000
        window._update_comparison_status()
        assert not window.align_heading_button.isEnabled()
        window.noitom_worker.command("disconnect")
        pump_until(app, lambda: window.noitom_state == "DISCONNECTED")
        window.noitom_worker.command("connect")
        pump_until(app, lambda: window.noitom_state == "CONNECTED")
        assert window.noitom_worker.heading_offset_rad == 0.
    finally:
        window.streaming_requested = False
        window.close()
        pump_until(app, lambda: window._allow_close)
        if replay is not None:
            replay.close()


def test_bundled_native_sdk_lifecycle_without_suit():
    if not ComparisonConfig().library_path().is_file():
        pytest.skip("Optional local Mocap API library absent")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    client = NoitomClient(ComparisonConfig(port=port))
    try:
        client.connect()
        assert client.version
        for _ in range(2):
            client.start()
            assert client.poll_events() == []
            client.stop()
    finally:
        client.disconnect()
