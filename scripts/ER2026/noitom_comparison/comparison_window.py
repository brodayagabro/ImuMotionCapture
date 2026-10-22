"""Experimental controls layered on the existing OpenGL viewer."""
from dataclasses import asdict
import math
import queue
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QPushButton, QProgressBar
from pyqt_mocap.mpu_udp_viewer_gl import OpenGLMotionCaptureWindow
from pyqt_mocap.mocap_core import SEGMENT_NAMES, quaternion_to_matrix
from scripts.ER2026.experiments.semaphore.frame_calibration import POSE_CAPTURE_S, POSE_PREPARATION_S
from .comparison_canvas import ComparisonOpenGLCanvas
from .config import ComparisonConfig
from .event_server import EventServer
from .noitom_adapter import COORDINATE_DESCRIPTION, NOITOM_TO_PROJECT, torso_heading
from .noitom_client import FakeNoitomClient, NoitomClient, NoitomReceiverThread
from .recording import plain
from .paired_recording import PairedRecorder
from pyqt_mocap.mocap_core import mapped_sensor_quaternion
from .settings_dialog import ComparisonSettingsDialog, load_noitom_segments, save_noitom_segments
from .synchronization import StreamStats
from .control_bridge import ControlBridge


class MocapComparisonWindow(OpenGLMotionCaptureWindow):
    canvas_class = ComparisonOpenGLCanvas
    settings_application_name = "MPU UDP Viewer ER2026"

    def __init__(self, options=None, *, client=None):
        self.options = options or ComparisonConfig()
        self.recorder = PairedRecorder(max_gap_ms=self.options.stale_timeout_ms)
        self.presenter = None
        self.experiment_recording_active = False
        self.recording_t0_ns = None
        self.session = None
        self.own_stats, self.noitom_stats = StreamStats(), StreamStats()
        self.noitom_state, self.noitom_error = "DISCONNECTED", ""
        self.noitom_pose, self.noitom_dirty = {}, False
        self.latest_noitom_frame = None
        self._own_started_ns, self._noitom_started_ns = None, None
        self._stale_sources = set()
        self._reported_sessions = set()
        self._closing_requested = self._allow_close = False
        self.control_bridge = ControlBridge()
        self.noitom_calibration = None
        self.series_context = None
        self._imu_calibration_dialog_open = False
        self.udp_monitor_enabled = False
        super().__init__()
        self.udp_monitor_check = QCheckBox('Показывать пакеты')
        self.udp_monitor_check.setToolTip('Включает обновление монитора. Приём данных, калибровка и запись работают при любом положении галочки.')
        self.udp_monitor_check.toggled.connect(self._set_udp_monitor_enabled)
        self.monitor_header.insertWidget(1, self.udp_monitor_check)
        self._set_udp_monitor_enabled(False)
        self.calibration_button.setText("Калибровка")
        self.calibration_button.setToolTip(f"A-поза → буква А → буква Т → буква У → A-поза · запись {POSE_CAPTURE_S:g} с · подготовка {POSE_PREPARATION_S:g} с на позу")
        self.calibration_action.setText("Калибровка…")
        self.imu_calibration_button = QPushButton('IMU-калибровка')
        self.imu_calibration_button.setToolTip(
            'Полная калибровка датчиков (CALIB), как в основном визуализаторе. '
            'Датчики должны быть сняты и лежать неподвижно локальной Z вверх.')
        self.imu_calibration_button.clicked.connect(self.open_imu_calibration_dialog)
        controls = self.calibration_button.parentWidget().layout()
        controls.insertWidget(controls.indexOf(self.calibration_button) + 1, self.imu_calibration_button)
        self._update_imu_calibration_button()
        self.guided_action.setVisible(False)
        self.semaphore_action.setVisible(False)
        self.noitom_enabled_segments = load_noitom_segments(self.settings_store)
        factory = FakeNoitomClient if self.options.fake else NoitomClient
        self.noitom_worker = NoitomReceiverThread(client or factory(self.options), self.recorder,
                                                self.noitom_enabled_segments)
        self.noitom_worker.start()
        self.setWindowTitle("IMU + Noitom — синхронный эксперимент" + (" · SYNTHETIC Noitom" if self.options.fake else ""))
        self._build_comparison_controls()
        self.event_server = None
        if self.options.event_port:
            try:
                self.event_server = EventServer(self.recorder, self.options.event_port, control=self.control_bridge)
                event_status = f"Управление демонстратором: 127.0.0.1:{self.event_server.port}"
            except OSError as error:
                event_status = f"Приём внешних событий недоступен: {error}"
            label = QLabel(event_status)
            label.setWordWrap(True)
            self.centralWidget().layout().insertWidget(4, label)

    def _set_udp_monitor_enabled(self, enabled):
        self.udp_monitor_enabled = enabled
        self.monitor.setEnabled(enabled)
        self.monitor.setPlaceholderText('Ожидание UDP-пакетов…' if enabled else
                                       'Монитор выключен. Включите «Показывать пакеты» для просмотра.')

    def _append_monitor(self, text):
        # Skip the text widget entirely; acquisition and recorder hooks run separately.
        if self.udp_monitor_enabled:
            super()._append_monitor(text)

    def open_calibration_dialog(self):
        self.open_guided_calibration()

    def _update_imu_calibration_button(self):
        if not hasattr(self, 'imu_calibration_button'):
            return
        busy = (self.experiment_recording_active or self.series_context or self.hardware_calibration_pending
                or self._imu_calibration_dialog_open
                or (self.session and not self.session.done.is_set())
                or (self.guided_dialog is not None and self.guided_dialog.phase != 'finished'))
        self.imu_calibration_button.setEnabled(self.sock is not None and not busy)

    def _set_connected_controls(self, connected):
        super()._set_connected_controls(connected)
        self._update_imu_calibration_button()

    def open_imu_calibration_dialog(self):
        self._update_imu_calibration_button()
        if not self.imu_calibration_button.isEnabled():
            self.statusBar().showMessage('Для IMU-калибровки подключите ESP32 и завершите запись или текущую калибровку', 8000)
            return
        self._imu_calibration_dialog_open = True
        try:
            super().open_imu_calibration_dialog()
        finally:
            self._imu_calibration_dialog_open = False
            self._update_imu_calibration_button()

    def import_calibration_profile(self):
        if self.series_context:
            self.statusBar().showMessage('Завершите запись перед заменой профиля', 8000)
            return
        super().import_calibration_profile()

    def open_guided_calibration(self, checked=False, *, semaphore=False):
        # This experiment uses one workflow for every calibration entry point.
        if self.series_context or self.hardware_calibration_pending:
            self.statusBar().showMessage('Завершите запись или IMU-калибровку перед калибровкой по позам', 8000)
            return
        previous = self.guided_dialog
        if previous is not None and previous.phase == 'finished':
            previous.accept()
            # Delete-on-close is deferred; dispose the reference before opening.
            previous.destroyed.disconnect(self._guided_dialog_closed)
            self.guided_dialog = None
        super().open_guided_calibration(checked)
        if self.guided_dialog is not None and self.guided_dialog is not previous:
            self.guided_dialog.diagnostic_event.connect(
                lambda name, detail: self.recorder.event("own_calibration_" + name, {
                    **detail, "sensor_mapping": dict(self.config.sensor_mapping),
                    "sensor_id_mode": self.model.active_sensor_id_mode,
                }))

    def _create_guided_calibration_dialog(self, *args, **kwargs):
        from scripts.ER2026.experiments.semaphore.guided_dialog import GuidedCalibrationDialog
        return GuidedCalibrationDialog(*args, sample_series=True, **kwargs)

    def _control_status(self):
        calibration = (self.calibration_document or {}).get('calibration', {})
        return {**self.recorder.recording_state(),
                'recording_sources': self._recording_sources(),
                'last_recording_sources': list(self.session.sources) if self.session else [],
                'recording_path': str(self.session.path) if self.session and self.experiment_recording_active else None,
                'series_context': self.series_context,
                'records_root': str(Path(self.options.recordings_dir).resolve()),
                'own_fresh': self.own_stats.age_ms() is not None and self.own_stats.age_ms() < self.options.stale_timeout_ms,
                'noitom_fresh': self.noitom_stats.age_ms() is not None and self.noitom_stats.age_ms() < self.options.stale_timeout_ms,
                'last_export': {k: self.session.metadata.get(k) for k in ('frames', 'fps', 'frame_time')} if self.session else {},
                'export_progress': self.session.export_progress if self.session else None,
                'own_calibration_id': calibration.get('calibration_id'),
                'own_calibration_method': calibration.get('method'),
                'noitom_calibration': self.noitom_calibration,
                'calibration_running': self.hardware_calibration_pending or self._imu_calibration_dialog_open or (
                    self.guided_dialog is not None and self.guided_dialog.phase != 'finished'),
                'own_streaming': self.streaming_requested, 'noitom_streaming': self.noitom_worker.streaming,
                'saving': bool(self.session and not self.experiment_recording_active and not self.session.done.is_set()),
                'last_recording_id': self.session.session_id if self.session else None,
                'recording_initialized': bool(self.session and self.session.ready.is_set() and not self.session.error),
                'last_recording_status': self.session.metadata.get('status') if self.session else None,
                'last_recording_path': str(self.session.path) if self.session else None,
                'last_recording_error': self.session.error if self.session else None}

    def _handle_control(self, payload):
        command = payload.get('command')
        if command == 'status':
            self.recorder.heartbeat(payload.get('recording_session_id'))
            return self._control_status()
        if command == 'calibrate':
            if self.series_context or not self.streaming_requested:
                raise ValueError('Завершите запись и включите поток IMU перед калибровкой')
            self.open_guided_calibration()
        elif command == 'imu_calibrate':
            self._update_imu_calibration_button()
            if not self.imu_calibration_button.isEnabled():
                raise ValueError('Подключите ESP32 и завершите запись или калибровку')
            # Acknowledge before opening the operator's modal confirmation.
            QTimer.singleShot(0, self.open_imu_calibration_dialog)
        elif command == 'confirm_noitom_calibration':
            if self.series_context or not self.noitom_worker.streaming:
                raise ValueError('Подтвердите калибровку Noitom между записями при активном потоке')
            self.noitom_calibration = {'calibration_id': uuid.uuid4().hex,
                'confirmed_at': datetime.now(timezone.utc).isoformat(), 'confirmed_ns': time.perf_counter_ns(),
                'source': 'operator_axis_studio_confirmation'}
        elif command == 'start_recording':
            status = self._control_status()
            if self.experiment_recording_active or status['saving'] or status['calibration_running']:
                raise ValueError('Завершите текущую запись или калибровку')
            sources = self._recording_sources()
            if not sources:
                raise ValueError('Выберите хотя бы один костюм для записи в окне захвата')
            if 'custom' in sources:
                if not self.streaming_requested or not status['own_fresh']:
                    raise ValueError('Нужны свежие кадры custom (ESP32)')
                if status['own_calibration_method'] != 'neutral_a_semaphore_a_t_u_neutral_a_sensor_y':
                    raise ValueError('Выполните пятиэтапную калибровку custom (ESP32)')
            if 'noitom' in sources:
                if not self.noitom_worker.streaming or not status['noitom_fresh']:
                    raise ValueError('Нужны свежие кадры Noitom')
                if not self.noitom_calibration:
                    raise ValueError('Подтвердите калибровку Noitom')
            self.series_context = {'subject_id': payload.get('subject_id'),
                                   'word_set_id': payload.get('word_set_id'),
                                   'duration_s': payload.get('duration_s', 300.),
                                   'export_fps': payload.get('export_fps', 50.),
                                   'recording_sources': sources}
            try:
                self.start_experiment_recording()
            except Exception:
                self.series_context = None
                raise
        elif command == 'stop_recording':
            if not self.session or payload.get('recording_session_id') != self.session.session_id:
                raise ValueError('ID записи не совпадает; чужая запись не остановлена')
            self.stop_experiment_recording()
            self.series_context = None
        else:
            raise ValueError('Неизвестная команда управления')
        return {**self._control_status(), 'series_context': self.series_context}

    def _build_comparison_controls(self):
        self.own_record_check = QCheckBox('Custom · в датасет')
        self.own_record_check.setChecked(True)
        self.own_record_check.setToolTip('Записывать custom.bvh с ESP32. Снимите галочку для записи только Noitom.')
        own_bar = self.connect_button.parentWidget().layout()
        own_bar.insertWidget(0, self.own_record_check)
        bar = QHBoxLayout()
        bar.addWidget(QLabel("<b>Noitom</b>" + (" · SYNTHETIC" if self.options.fake else "")))
        self.noitom_record_check = QCheckBox('В датасет')
        self.noitom_record_check.setChecked(True)
        self.noitom_record_check.setToolTip('Записывать noitom.bvh. Снимите галочку для записи только custom.')
        bar.addWidget(self.noitom_record_check)
        self.noitom_buttons = {}
        for command, title in (("connect", "Connect"), ("disconnect", "Disconnect"),
                               ("start", "Start"), ("stop", "Stop")):
            button = QPushButton(title)
            button.clicked.connect(lambda checked=False, command=command: self.noitom_worker.command(command))
            bar.addWidget(button)
            self.noitom_buttons[command] = button
        self.noitom_status_label = QLabel("DISCONNECTED")
        bar.addWidget(self.noitom_status_label, 1)
        self.align_heading_button = QPushButton("Совместить направление")
        self.align_heading_button.setToolTip(
            "После калибровки встаньте прямо, лицом в её исходном направлении. "
            "Совмещается только общий поворот Noitom вокруг вертикали; углы суставов сохраняются.")
        self.align_heading_button.clicked.connect(self.align_noitom_heading)
        bar.addWidget(self.align_heading_button)
        self.experiment_button = QPushButton("Эксперимент…")
        self.experiment_button.clicked.connect(self.open_experiment_presenter)
        bar.addWidget(self.experiment_button)
        self.centralWidget().layout().insertLayout(1, bar)
        self.comparison_status_label = QLabel()
        self.centralWidget().layout().insertWidget(2, self.comparison_status_label)
        self.session_label = QLabel("Записи: " + self.options.recordings_dir)
        self.session_label.setWordWrap(True)
        self.centralWidget().layout().insertWidget(3, self.session_label)
        self.export_progress_bar = QProgressBar()
        self.export_progress_bar.setRange(0, 100)
        self.export_progress_bar.setValue(0)
        self.export_progress_bar.hide()
        self.centralWidget().layout().insertWidget(4, self.export_progress_bar)
        # Dataset capture is controlled by the experiment presenter.
        self.record_bvh_button.hide()
        self.save_bvh_button.hide()
        self.bvh_label.hide()
        for action in (self.record_bvh_action, self.stop_bvh_action, self.save_bvh_action):
            action.setVisible(False)
        self._update_comparison_status()

    def _recording_sources(self):
        return [name for name, check in (('custom', self.own_record_check),
                                        ('noitom', self.noitom_record_check)) if check.isChecked()]

    def _stamp_packet(self):
        return self.recorder.stamp_packet()

    def _heading_alignment_available(self):
        now = time.perf_counter_ns()
        limit = self.options.stale_timeout_ms * 1_000_000
        return (not self.experiment_recording_active and self.latest_noitom_frame is not None and self.noitom_worker.streaming
                and self.streaming_requested and not self.model.neutral_pending
                and set(self.model.neutral_orientation) == set(SEGMENT_NAMES)
                and self.own_stats.last_frame_ns is not None
                and 0 <= now - self.latest_noitom_frame.host_timestamp_ns < limit
                and 0 <= now - self.own_stats.last_frame_ns < limit)

    def align_noitom_heading(self):
        if not self._heading_alignment_available():
            self.statusBar().showMessage("Для совмещения нужны свежие данные обеих систем и калибровка IMU", 8000)
            return
        frame = self.latest_noitom_frame
        # Own calibration defines forward. If spine is disabled its mannequin
        # stays at zero heading, irrespective of Noitom's SDK world heading.
        right = quaternion_to_matrix(self.model.orientations()["spine"]) @ [1., 0., 0.]
        try:
            if math.hypot(right[0], right[1]) < 1e-6:
                raise ValueError("Для совмещения встаньте прямо")
            own_heading = math.atan2(right[1], right[0])
            offset = math.remainder(own_heading - torso_heading(frame), 2 * math.pi)
        except ValueError as error:
            self.statusBar().showMessage(str(error), 8000)
            return
        # Atomic scalar replacement: the worker snapshots it once per frame.
        self.noitom_worker.heading_offset_rad = offset
        self.recorder.event("noitom_heading_aligned", {
            "heading_offset_rad": offset, "reference_frame_ns": frame.host_timestamp_ns,
            "own_heading_rad": own_heading,
        })
        self.statusBar().showMessage(
            f"Направления совмещены: поворот Noitom {math.degrees(offset):+.1f}°", 8000)

    def open_settings(self):
        if self.experiment_recording_active:
            self.statusBar().showMessage('Изменяйте настройки между записями', 8000)
            return
        if self.settings_dialog is not None:
            self.settings_dialog.raise_()
            self.settings_dialog.activateWindow()
            return
        dialog = ComparisonSettingsDialog(self.config, self.noitom_enabled_segments, self)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.config_applied.connect(self.apply_configuration)
        dialog.noitom_segments_applied.connect(self.apply_noitom_segments)
        dialog.destroyed.connect(self._settings_closed)
        self.settings_dialog = dialog
        dialog.show()

    def apply_configuration(self, config):
        if self.experiment_recording_active:
            self.statusBar().showMessage('Изменяйте настройки между записями', 8000)
            return
        super().apply_configuration(config)

    def capture_neutral_pose(self):
        if self.experiment_recording_active:
            self.statusBar().showMessage('Калибровка доступна между записями', 8000)
            return
        super().capture_neutral_pose()

    def apply_noitom_segments(self, enabled):
        enabled = frozenset(enabled)
        if enabled == self.noitom_enabled_segments:
            return
        self.noitom_worker.set_enabled_segments(enabled)
        self.noitom_enabled_segments = enabled
        save_noitom_segments(self.settings_store, enabled)
        self.recorder.event("noitom_enabled_segments_changed", {"enabled_segments": sorted(enabled)})
        self.noitom_dirty = self.noitom_stats.frames > 0
        self._refresh_plot()

    def _calibration_state(self):
        return plain({"config": asdict(self.config), "neutral_orientation": self.model.neutral_orientation,
                      "axis_alignment_quaternion": self.model.axis_alignment_quaternion,
                      "drift_rate_rad_s": self.model.drift_rate_rad_s,
                      "drift_reference_s": self.model.drift_reference_s,
                      "drift_compensation_enabled": self.model.drift_compensation_enabled,
                      "segment_calibrations": {s: asdict(c) for s, c in self.model.segment_calibrations.items()},
                      "smooth_alpha": self.model.smooth_alpha,
                      "profile": self.calibration_document})

    def _on_datagram_processed(self, raw, address, wall_timestamp, received_ns, result, context):
        if result.calibration_samples:
            self.own_stats.observe(received_ns)
        if context is not None:
            frame = {'measurement_complete': bool(result.calibration_samples and result.measurement_key
                and not self.model.neutral_pending and len(self.model.segment_calibrations) == 5),
                'measurement_key': result.measurement_key}
            try:
                if frame['measurement_complete']:
                    # Export C*q*K for every measured segment, independent of display smoothing/visibility.
                    frame['bvh_orientations'] = {
                        name: self.model.segment_calibrations[name].apply(mapped_sensor_quaternion(
                            result.calibration_samples[name], self.model.axis_maps[name]))
                        for name in self.model.sensor_mapping}
                self.recorder.complete_packet(context, {}, frame)
            except (ValueError, KeyError) as error:
                self.recorder.release_claim(context, error=str(error))
        if self.guided_dialog is not None and self.guided_dialog.sample_series:
            self.guided_dialog.feed_measurement(result.calibration_samples, result.measurement_key, received_ns / 1e9)

    def start_stream(self):
        super().start_stream()
        if self.streaming_requested:
            self._own_started_ns = time.perf_counter_ns()
            self.own_stats = StreamStats()

    def send_command(self, command, warn=True):
        sent = super().send_command(command, warn)
        if sent:
            verb = command.strip().split()[0].upper()
            self.recorder.event("own_command", {"command": command}, incomplete=verb == "STOP")
        return sent

    def disconnect_device(self, _checked=False, log=True):
        if self.sock is not None:
            self.recorder.event("own_disconnect", incomplete=True)
        super().disconnect_device(_checked, log)

    def _metadata(self):
        return {"series": self.series_context, "own_system": {"device": "ESP32_MPU6050", **self._calibration_state()},
                "noitom": {"sdk": "MocapApi", "sdk_version": self.noitom_worker.client.version,
                           "enabled_segments": sorted(self.noitom_enabled_segments),
                           "heading_offset_rad": self.noitom_worker.heading_offset_rad,
                           "delivery_mode": self.noitom_worker.client.delivery_mode,
                           "sdk_render_settings": getattr(self.noitom_worker.client, "sdk_render_settings", {}),
                           "configuration": self.options.metadata(), "mapping": NOITOM_TO_PROJECT,
                           "coordinate_conversion": COORDINATE_DESCRIPTION},
                "synthetic_noitom": self.options.fake}

    def open_experiment_presenter(self):
        if self.presenter is None:
            from scripts.ER2026.experiments.semaphore.config import load_config
            from scripts.ER2026.experiments.semaphore.presenter_window import PresenterWindow
            self.presenter = PresenterWindow(load_config(), backend=self._handle_control)
        self.presenter.show()
        self.presenter.raise_()
        self.presenter.activateWindow()

    def toggle_experiment_recording(self):
        self.open_experiment_presenter()

    def start_experiment_recording(self):
        if self.experiment_recording_active or (self.session and not self.session.done.is_set()):
            return
        # Recorder claims establish the boundary without waiting for a native
        # connect/close/poll operation in the Noitom worker.
        if not self.series_context:
            self.open_experiment_presenter()
            return
        sources = self.series_context.get('recording_sources', ['custom', 'noitom'])
        self.session = self.recorder.start(self.options.recordings_dir,
            {**self._metadata(), **self.series_context,
             'max_fps': self.config.stream_rate_hz if 'custom' in sources else 100.,
             'require_heartbeat': True})
        self.recording_t0_ns = self.session.t0_ns
        self.export_progress_bar.setValue(0)
        self.export_progress_bar.hide()
        self.experiment_recording_active = True
        self.experiment_button.setText("Открыть эксперимент…")
        self.session_label.setText("Запись: " + str(self.session.path))
        self._stale_sources.clear()
        if 'custom' in self.session.sources and not self.streaming_requested:
            self.recorder.event("own_not_streaming_at_start", incomplete=True)
        if 'noitom' in self.session.sources and not self.noitom_worker.streaming:
            self.recorder.event("noitom_not_streaming_at_start", incomplete=True)
        self._update_comparison_status()

    def stop_experiment_recording(self):
        if not self.experiment_recording_active:
            return
        self.recorder.stop()
        self.experiment_recording_active = False
        self.series_context = None
        self.experiment_button.setEnabled(False)
        self.experiment_button.setText("Сохранение…")

    def _process_events(self):
        super()._process_events()
        if not hasattr(self, "noitom_worker"):
            return
        self.control_bridge.process(self._handle_control)
        for _ in range(500):
            try:
                kind, value, detail = self.noitom_worker.events.get_nowait()
            except queue.Empty:
                break
            if kind == "state":
                self.noitom_state, self.noitom_error = value, detail
                if value == "CONNECTED" and self.noitom_worker.client.delivery_mode == "poll_latest":
                    self._append_monitor(f"Noitom SDK {self.noitom_worker.client.version}: "
                        f"{self.options.transport.upper()} :{self.options.port}, BVH {self.options.bvh_rotation}, "
                        f"polling как demo-py. DLL: {self.options.library_path()}")
                if value == "STREAMING":
                    self._noitom_started_ns = time.perf_counter_ns()
                    self.noitom_stats = StreamStats()
                if value in {"DISCONNECTED", "ERROR"}:
                    self.latest_noitom_frame = None
                    self.human_canvas.set_noitom_visible(False)
                if detail:
                    self._append_monitor("Noitom: " + detail)
            else:
                self.noitom_stats.observe(value.host_timestamp_ns)
                pose, error = detail
                if error is None:
                    self.latest_noitom_frame = value
                    self.noitom_pose, self.noitom_dirty = pose, True
                if error != (self.noitom_error or None):
                    if error:
                        self._append_monitor("Noitom mapping: " + error)
                    self.noitom_error = error or ""
        if self.experiment_recording_active and self.recorder.active is None:
            self.experiment_recording_active = False
            self.series_context = None
        if self.session and self.session.ready.is_set():
            self.recording_t0_ns = self.session.t0_ns
        if self.session and self.session.metadata.get('status') in {'saving', 'complete', 'error'}:
            progress = self.session.export_progress
            self.export_progress_bar.show()
            self.export_progress_bar.setValue(progress['percent'])
            self.export_progress_bar.setFormat('Сохранение BVH: %p%')
            if self.session.metadata.get('status') == 'saving':
                self.session_label.setText(progress['phase'])
        self._update_comparison_status()
        if self.session and self.session.error and self.experiment_recording_active:
            self.stop_experiment_recording()
        for session in self.recorder.sessions:
            if session.done.is_set() and session.session_id not in self._reported_sessions:
                self._reported_sessions.add(session.session_id)
                self.session_label.setText(f"{session.metadata['status']}: {session.path}" +
                                           (f" · {session.error}" if session.error else ""))
                self.experiment_button.setText("Эксперимент…")
                self.experiment_button.setEnabled(not self._closing_requested)
        if self._closing_requested and not self.noitom_worker.is_alive() and all(
                session.done.is_set() for session in self.recorder.sessions):
            self._allow_close = True
            self.close()

    def _update_comparison_status(self):
        self._update_imu_calibration_button()
        busy = (self.experiment_recording_active or bool(self.series_context)
                or bool(self.session and not self.session.done.is_set()) or self._closing_requested)
        for check in (self.own_record_check, self.noitom_record_check):
            check.setEnabled(not busy)
        now = time.perf_counter_ns()
        summaries = []
        for source, stats, running, started in (
            ("OWN", self.own_stats, self.streaming_requested, self._own_started_ns),
            ("NOITOM", self.noitom_stats, self.noitom_worker.streaming, self._noitom_started_ns),
        ):
            age = stats.age_ms(now)
            reference = stats.last_frame_ns if stats.last_frame_ns is not None else started
            stale = running and reference is not None and now - reference > self.options.stale_timeout_ms * 1_000_000
            if stale and source not in self._stale_sources:
                self.recorder.event(source.lower() + "_stale", incomplete=True)
                self._stale_sources.add(source)
            elif not stale and source in self._stale_sources:
                self.recorder.event(source.lower() + "_resumed")
                self._stale_sources.remove(source)
            state = "STALE" if stale else ("STREAMING" if running else "STOPPED")
            summaries.append(f"{source} {state} · Frames {stats.frames} · {stats.rate_hz() if not stale else 0:.1f} Hz · Age {age:.0f} ms"
                             if age is not None else f"{source} {state} · Frames {stats.frames} · ожидаются данные")
        if self.own_stats.last_frame_ns is not None and self.noitom_stats.last_frame_ns is not None:
            delta = (self.own_stats.last_frame_ns - self.noitom_stats.last_frame_ns) / 1e6
            summaries.append(f"latest-frame Δ {delta:+.1f} ms")
        if self.experiment_recording_active:
            seconds = (now - self.recording_t0_ns) / 1e9
            summaries.append(f"REC {int(seconds // 3600):02}:{int(seconds // 60) % 60:02}:{seconds % 60:06.3f}")
        self.comparison_status_label.setText("   |   ".join(summaries))
        self.comparison_status_label.setWordWrap(True)
        state = "STALE" if "NOITOM" in self._stale_sources else self.noitom_state
        endpoint = f"{self.options.transport.upper()} :{self.options.port}"
        self.noitom_status_label.setText(endpoint + " · " + state + (" · " + self.noitom_error if self.noitom_error else ""))
        if state == "STALE" and self.noitom_stats.frames == 0:
            self.noitom_status_label.setText(f"{endpoint} · STALE: нет кадров — проверьте порт вещания и версию DLL")
        for name, button in self.noitom_buttons.items():
            enabled = {"connect": self.noitom_state in {"DISCONNECTED", "ERROR"},
                       "disconnect": self.noitom_state not in {"DISCONNECTED"},
                       "start": self.noitom_state == "CONNECTED", "stop": self.noitom_worker.streaming}[name]
            button.setEnabled(enabled and not self._closing_requested)
        self.align_heading_button.setEnabled(self._heading_alignment_available() and not self._closing_requested)

    def _refresh_plot(self):
        super()._refresh_plot()
        if self.noitom_dirty:
            if self.noitom_state in {"CONNECTED", "STREAMING"}:
                self.human_canvas.update_noitom_pose(self.noitom_pose, self.noitom_enabled_segments)
            self.noitom_dirty = False

    def closeEvent(self, event):
        if self._allow_close:
            if self.event_server is not None:
                self.event_server.close()
            self.event_timer.stop()
            self.render_timer.stop()
            event.accept()
            return
        if not self._closing_requested:
            super().closeEvent(event)
            if not event.isAccepted():
                return
            self.stop_experiment_recording()
            if self.presenter is not None:
                self.presenter.close()
            self._closing_requested = True
            self.noitom_worker.command("shutdown")
            self.experiment_button.setEnabled(False)
            self.session_label.setText("Завершение приёма и сохранения…")
        # Keep the event loop alive to process pending own packet claims.
        event.ignore()
