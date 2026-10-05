"""Experimental controls layered on the existing OpenGL viewer."""
import base64
from dataclasses import asdict
import json
import math
import queue
import time
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton
from pyqt_mocap.mpu_udp_viewer_gl import OpenGLMotionCaptureWindow
from pyqt_mocap.mocap_core import SEGMENT_NAMES, quaternion_to_matrix
from .comparison_canvas import ComparisonOpenGLCanvas
from .config import ComparisonConfig
from .event_server import EventServer
from .noitom_adapter import COORDINATE_DESCRIPTION, NOITOM_TO_PROJECT, torso_heading
from .noitom_client import FakeNoitomClient, NoitomClient, NoitomReceiverThread
from .recording import ExperimentRecorder, plain
from .settings_dialog import ComparisonSettingsDialog, load_noitom_segments, save_noitom_segments
from .synchronization import StreamStats


class MocapComparisonWindow(OpenGLMotionCaptureWindow):
    canvas_class = ComparisonOpenGLCanvas

    def __init__(self, options=None, *, client=None):
        self.options = options or ComparisonConfig()
        self.recorder = ExperimentRecorder()
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
        self._calibration_key = None
        super().__init__()
        self.calibration_button.setText("Калибровка")
        self.calibration_button.setToolTip("A-поза → T-поза → A-поза")
        self.calibration_action.setText("Калибровка…")
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
                self.event_server = EventServer(self.recorder, self.options.event_port)
                event_status = f"Внешние события: 127.0.0.1:{self.event_server.port} · запись меток при REC"
            except OSError as error:
                event_status = f"Приём внешних событий недоступен: {error}"
            label = QLabel(event_status)
            label.setWordWrap(True)
            self.centralWidget().layout().insertWidget(4, label)

    def open_calibration_dialog(self):
        self.open_guided_calibration()

    def open_guided_calibration(self, checked=False, *, semaphore=False):
        # This experiment uses one workflow for every calibration entry point.
        previous = self.guided_dialog
        super().open_guided_calibration(checked, t_pose_only=True)
        if self.guided_dialog is not None and self.guided_dialog is not previous:
            self.guided_dialog.diagnostic_event.connect(
                lambda name, detail: self.recorder.event("own_calibration_" + name, {
                    **detail, "sensor_mapping": dict(self.config.sensor_mapping),
                    "sensor_id_mode": self.model.active_sensor_id_mode,
                }))

    def _build_comparison_controls(self):
        bar = QHBoxLayout()
        bar.addWidget(QLabel("<b>Noitom</b>" + (" · SYNTHETIC" if self.options.fake else "")))
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
        self.experiment_button = QPushButton("REC · эксперимент")
        self.experiment_button.clicked.connect(self.toggle_experiment_recording)
        bar.addWidget(self.experiment_button)
        self.centralWidget().layout().insertLayout(1, bar)
        self.comparison_status_label = QLabel()
        self.centralWidget().layout().insertWidget(2, self.comparison_status_label)
        self.session_label = QLabel("Записи: " + self.options.recordings_dir)
        self.session_label.setWordWrap(True)
        self.centralWidget().layout().insertWidget(3, self.session_label)
        self.record_bvh_button.setText("Начать запись BVH")
        self._update_comparison_status()

    def _stamp_packet(self):
        return self.recorder.stamp_packet()

    def _heading_alignment_available(self):
        now = time.monotonic_ns()
        limit = self.options.stale_timeout_ms * 1_000_000
        return (self.latest_noitom_frame is not None and self.noitom_worker.streaming
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
        if result.published:
            self.own_stats.observe(received_ns)
        packet = {"wall_timestamp": wall_timestamp, "address": address,
                  "payload_base64": base64.b64encode(raw).decode("ascii"),
                  "published": result.published, "invalid_lines": result.invalid_lines}
        frame = None
        if result.published:
            header = (result.frame_header or "").split()
            try:
                index = int(header[1]) if len(header) > 1 else None
            except ValueError:
                index = None
            frame = {
                "frame_index": index, "frame_header": result.frame_header,
                "sample_generation": self.model.sample_generation,
                "raw_sensor_quaternions": {str(sensor): sample.quaternion
                    for sensor, sample in self.model.latest_samples.items()
                    if sample.generation == self.model.sample_generation},
                "segment_orientations": self.model.orientations(),
                "neutral_pending": self.model.neutral_pending,
                "enabled_segments": sorted(self.config.enabled_segments),
                "sensor_id_mode": self.model.active_sensor_id_mode,
                "sensor_mapping": dict(self.model.sensor_mapping),
                "calibration_method": (self.calibration_document or {}).get("calibration", {}).get("method"),
            }
        self.recorder.complete_packet(context, packet, frame)
        if self.experiment_recording_active and received_ns >= self.recording_t0_ns:
            state = self._calibration_state()
            key = json.dumps(state, sort_keys=True)
            if key != self._calibration_key:
                self.recorder.event("own_calibration_state", state, timestamp_ns=received_ns)
                self._calibration_key = key

    def start_stream(self):
        super().start_stream()
        if self.streaming_requested:
            self._own_started_ns = time.monotonic_ns()
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
        return {"own_system": {"device": "ESP32_MPU6050", **self._calibration_state()},
                "noitom": {"sdk": "MocapApi", "sdk_version": self.noitom_worker.client.version,
                           "enabled_segments": sorted(self.noitom_enabled_segments),
                           "heading_offset_rad": self.noitom_worker.heading_offset_rad,
                           "delivery_mode": self.noitom_worker.client.delivery_mode,
                           "sdk_render_settings": getattr(self.noitom_worker.client, "sdk_render_settings", {}),
                           "configuration": self.options.metadata(), "mapping": NOITOM_TO_PROJECT,
                           "coordinate_conversion": COORDINATE_DESCRIPTION},
                "synthetic_noitom": self.options.fake}

    def toggle_experiment_recording(self):
        if self.experiment_recording_active:
            self.stop_experiment_recording()
        else:
            self.start_experiment_recording()

    def start_experiment_recording(self):
        if self.experiment_recording_active or (self.session and not self.session.done.is_set()):
            return
        # Recorder claims establish the boundary without waiting for a native
        # connect/close/poll operation in the Noitom worker.
        self.session = self.recorder.start(self.options.recordings_dir, self._metadata())
        self.recording_t0_ns = self.session.t0_ns
        self.experiment_recording_active = True
        self._calibration_key = None
        self.experiment_button.setText("STOP REC")
        self.session_label.setText("Запись: " + str(self.session.path))
        self._stale_sources.clear()
        if not self.streaming_requested:
            self.recorder.event("own_not_streaming_at_start", incomplete=True)
        if not self.noitom_worker.streaming:
            self.recorder.event("noitom_not_streaming_at_start", incomplete=True)

    def stop_experiment_recording(self):
        if not self.experiment_recording_active:
            return
        self.recorder.stop()
        self.experiment_recording_active = False
        self.experiment_button.setEnabled(False)
        self.experiment_button.setText("Сохранение…")

    def _process_events(self):
        super()._process_events()
        if not hasattr(self, "noitom_worker"):
            return
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
                    self._noitom_started_ns = time.monotonic_ns()
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
        self._update_comparison_status()
        if self.session and self.session.error and self.experiment_recording_active:
            self.stop_experiment_recording()
        for session in self.recorder.sessions:
            if session.done.is_set() and session.session_id not in self._reported_sessions:
                self._reported_sessions.add(session.session_id)
                self.session_label.setText(f"{session.metadata['status']}: {session.path}" +
                                           (f" · {session.error}" if session.error else ""))
                self.experiment_button.setText("REC · эксперимент")
                self.experiment_button.setEnabled(not self._closing_requested)
        if self._closing_requested and not self.noitom_worker.is_alive() and all(
                session.done.is_set() for session in self.recorder.sessions):
            self._allow_close = True
            self.close()

    def _update_comparison_status(self):
        now = time.monotonic_ns()
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
            self._closing_requested = True
            self.noitom_worker.command("shutdown")
            self.experiment_button.setEnabled(False)
            self.session_label.setText("Завершение приёма и сохранения…")
        # Keep the event loop alive to process pending own packet claims.
        event.ignore()
