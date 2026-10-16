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
from scripts.ER2026.noitom_comparison.comparison_canvas import (
    ComparisonOpenGLCanvas, OWN_COLOR, NOITOM_COLOR, NOITOM_DISABLED_COLOR,
)
from scripts.ER2026.noitom_comparison.comparison_window import MocapComparisonWindow
from scripts.ER2026.noitom_comparison.config import ComparisonConfig
from scripts.ER2026.noitom_comparison.noitom_client import FakeNoitomClient, NoitomClient
from scripts.ER2026.noitom_comparison.replay import ReplayWindow, session_frames

pytestmark = pytest.mark.gui


def pump_until(app, predicate, timeout=3):
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        app.processEvents()
        if predicate():
            return
        time.sleep(.005)
    raise AssertionError("Timed out waiting for condition")


def test_experiment_settings_do_not_overwrite_main_viewer(tmp_path, monkeypatch):
    def settings(organization, application):
        return QSettings(str(tmp_path / (application + '.ini')), QSettings.Format.IniFormat)

    monkeypatch.setattr(mpu_udp_viewer, 'QSettings', settings)
    app = QApplication.instance() or QApplication([])
    main_settings = settings('Neuromorph', mpu_udp_viewer.MotionCaptureWindow.settings_application_name)
    config = mpu_udp_viewer.load_config(main_settings)
    config.device_ip = '192.0.2.10'
    mpu_udp_viewer.save_config(main_settings, config)
    window = MocapComparisonWindow(ComparisonConfig(fake=True, recordings_dir=str(tmp_path / 'records')))
    try:
        assert window.config.device_ip != '192.0.2.10'
        window.config.device_ip = '192.0.2.20'
        mpu_udp_viewer.save_config(window.settings_store, window.config)
        assert mpu_udp_viewer.load_config(main_settings).device_ip == '192.0.2.10'
        assert window.settings_store.fileName() != main_settings.fileName()
    finally:
        window.close()
        pump_until(app, lambda: window._allow_close)


@pytest.mark.parametrize('fail_first', [False, True])
def test_calibration_button_runs_five_measured_poses_and_applies_profile(tmp_path, monkeypatch, fail_first):
    from types import SimpleNamespace
    from scripts.ER2026.experiments.semaphore import guided_dialog
    from scripts.ER2026.experiments.semaphore.calibration_protocol import POSE_SEQUENCE as T_POSE_NAMES
    from scripts.ER2026.experiments.semaphore.tests.test_series_calibration import physical_series_captures as physical_captures
    from pyqt_mocap.mocap_core import matrix_to_quaternion, quaternion_to_matrix
    from pyqt_mocap.tests.test_calibration import rotation_x
    from scripts.ER2026.diagnostics.orientation.calibration_analyzer import events_from_session
    from PyQt6.QtWidgets import QMessageBox

    monkeypatch.setattr(mpu_udp_viewer, "QSettings", lambda *a: QSettings(str(tmp_path / "viewer.ini"), QSettings.Format.IniFormat))
    app = QApplication.instance() or QApplication([])
    clock = [100.]
    monkeypatch.setattr(guided_dialog, "time", SimpleNamespace(perf_counter=lambda: clock[0]))
    captures = physical_captures()
    if fail_first:
        captures['neutral_a_pose_final'].average['forearm.R'] = matrix_to_quaternion(
            rotation_x(np.radians(40.3)) @ quaternion_to_matrix(captures['neutral_a_pose'].average['forearm.R']))
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
        window.calibration_button.click()
        dialog = window.guided_dialog
        assert dialog.pose_names == T_POSE_NAMES
        assert dialog.windowTitle() == 'Калибровка: A-поза → А → Т → У → A-поза'
        assert dialog.title_label.text().startswith('1/5')
        diagnostic_events = []
        dialog.diagnostic_event.connect(lambda name, detail: diagnostic_events.append({'event': 'own_calibration_' + name, 'details': detail}))
        dialog.capture_button.click()
        for stage in range(5):
            assert dialog.stage_index == stage
            assert dialog.title_label.text().startswith(f'{stage + 1}/5')
            assert dialog.phase == 'preparing' and dialog.preparation_duration_s == 3.
            start = clock[0]
            for index in range(1, 91):
                clock[0] = start + 3. + index * .02
                dialog.feed_measurement(captures[T_POSE_NAMES[dialog.stage_index]].average,
                                        (index, index*20), clock[0])
        if fail_first:
            assert dialog.phase == 'idle' and window.calibration_document is None
            captures['neutral_a_pose_final'].average['forearm.R'] = captures['neutral_a_pose'].average['forearm.R'].copy()
            dialog.capture_button.click()
            start = clock[0]
            for index in range(1, 91):
                clock[0] = start + 3. + index * .02
                dialog.feed_measurement(captures[T_POSE_NAMES[dialog.stage_index]].average,
                                        (index, index*20), clock[0])
        events = diagnostic_events
        poses = [e['details'] for e in events if e['event'] == 'own_calibration_pose_captured']
        assert len(poses) == (6 if fail_first else 5)
        assert all('average_quaternions_wxyz' in e for e in poses)
        assert all(e['frame_diagnostics']['accepted_frames'] == 76 for e in poses)
        assert all(e['frame_diagnostics']['capture_duration_s'] == pytest.approx(1.5) for e in poses)
        assert all(e['frame_diagnostics']['preparation_duration_s'] == 3. for e in poses)
        failures = [e['details'] for e in events if e['event'] == 'own_calibration_fit_rejected']
        assert len(failures) == int(fail_first)
        if fail_first:
            assert failures[0]['closure_error_deg']['forearm.R'] == pytest.approx(40.3)
            assert set(failures[0]['poses']) == set(T_POSE_NAMES)
        assert dialog.phase == 'finished'
        assert not dialog.timer.isActive()
        assert tuple(dialog.captures) == T_POSE_NAMES
        assert window.last_calibration_result.method == 'neutral_a_semaphore_a_t_u_neutral_a_sensor_y'
        assert window.calibration_document['calibration']['method'] == 'neutral_a_semaphore_a_t_u_neutral_a_sensor_y'
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


def test_noitom_display_selection_persists_without_changing_full_source(tmp_path, monkeypatch):
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
        assert 'Spine2' in window.latest_noitom_frame.joints
        # Cancel does not apply a checkbox edit, including to the saved defaults.
        dialog.noitom_checks["spine"].setChecked(True)
        dialog.reject()
        assert window.noitom_enabled_segments == frozenset()
        window.close()
        pump_until(app, lambda: window._allow_close)

        reopened = MocapComparisonWindow(ComparisonConfig(fake=True))
        assert reopened.noitom_enabled_segments == frozenset()
        assert reopened.config == own_config
    finally:
        window.bvh_active = False
        for candidate in (window, reopened):
            if candidate is not None:
                candidate.close()
                pump_until(app, lambda: candidate._allow_close)
        if replay is not None:
            replay.close()


def test_heading_control_aligns_fixed_transform_between_recordings(tmp_path, monkeypatch):
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
            {name: (1., 0., 0., 0.) for name in DEFAULT_SENSOR_MAPPING}, time.perf_counter())
        window.own_stats.observe(time.perf_counter_ns())
        window._update_comparison_status()
        assert window.align_heading_button.isEnabled()
        count = window.noitom_stats.frames
        pump_until(app, lambda: window.noitom_stats.frames > count + 2)
        window.align_heading_button.click()
        assert window.noitom_worker.heading_offset_rad == pytest.approx(.9)
        assert window._metadata()["noitom"]["heading_offset_rad"] == pytest.approx(.9)
        count = window.noitom_stats.frames
        pump_until(app, lambda: window.noitom_stats.frames > count + 2)
        window.own_stats.last_frame_ns = time.perf_counter_ns() - 2 * window.options.stale_timeout_ms * 1_000_000
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


def test_imu_calibration_button_uses_shared_calib_command_and_ack(tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox

    monkeypatch.setattr(mpu_udp_viewer, 'QSettings', lambda *args: QSettings(str(tmp_path/'viewer.ini'), QSettings.Format.IniFormat))
    app = QApplication.instance() or QApplication([])
    window = MocapComparisonWindow(ComparisonConfig(fake=True, recordings_dir=str(tmp_path/'sessions')))
    server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server.bind(('127.0.0.1', 0))
    server.settimeout(.1)
    window.config.device_ip, window.config.device_port = server.getsockname()
    answer = [QMessageBox.StandardButton.No]
    confirmations = []

    def confirm(*args):
        confirmations.append(args)
        assert window._control_status()['calibration_running']
        return answer[0]

    monkeypatch.setattr(QMessageBox, 'warning', confirm)
    try:
        assert window.imu_calibration_button.text() == 'IMU-калибровка'
        assert not window.imu_calibration_button.isEnabled()
        assert window.connect_device()
        hello, address = server.recvfrom(1024)
        assert hello == b'HELLO\n' and window.imu_calibration_button.isEnabled()
        old_profile = {'calibration': {'calibration_id': 'old-profile'}}
        window.calibration_document = old_profile
        window.imu_calibration_button.click()
        assert not window.hardware_calibration_pending and window.calibration_document is old_profile
        with pytest.raises(socket.timeout):
            server.recvfrom(1024)

        window.experiment_recording_active = True
        window._update_imu_calibration_button()
        assert not window.imu_calibration_button.isEnabled()
        window.open_imu_calibration_dialog()
        assert len(confirmations) == 1
        window.experiment_recording_active = False
        window._update_imu_calibration_button()
        assert window.imu_calibration_button.isEnabled()

        answer[0] = QMessageBox.StandardButton.Yes
        window.imu_calibration_button.click()
        command, _ = server.recvfrom(1024)
        while command == b'STATUS\n':  # deferred connection status request
            command, _ = server.recvfrom(1024)
        assert command == b'CALIB\n'
        assert window.hardware_calibration_pending and not window.imu_calibration_button.isEnabled()
        assert window.calibration_document is old_profile
        window.open_guided_calibration()
        assert window.guided_dialog is None
        with pytest.raises(ValueError, match='Завершите'):
            window._handle_control({'command': 'start_recording'})
        server.sendto(b'ACK CALIB DONE\n', address)
        pump_until(app, lambda: not window.hardware_calibration_pending)
        assert window.calibration_document is None and window.model.neutral_pending
        assert window.imu_calibration_button.isEnabled()
        assert confirmations[-1][2] == 'Все датчики сняты, лежат неподвижно локальной Z вверх?'
        window.disconnect_device()
        assert not window.imu_calibration_button.isEnabled()
    finally:
        window.bvh_dirty = False
        window.close()
        pump_until(app, lambda: window._allow_close)
        server.close()
