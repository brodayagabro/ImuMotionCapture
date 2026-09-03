"""Calibration/profile actions mixed into the main PyQt window."""

from __future__ import annotations

from datetime import datetime

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFileDialog, QMessageBox

from .calibration import (
    CalibrationResult,
    load_profile,
    profile_document,
    save_profile,
)
from .guided_dialog import GuidedCalibrationDialog
from .mocap_core import (
    AXIS_MAPPING_REVISION,
    DEFAULT_AXIS_MAPS,
    DEFAULT_SENSOR_MAPPING,
    IDENTITY_QUATERNION,
    LEGACY_SPINE_AXIS_MAP,
    LEGACY_SENSOR_MAPPING,
    SEGMENT_NAMES,
    SENSOR_MAPPING_REVISION,
)

HIGH_POSE_ERROR_WARNING_DEG = 45.0


class CalibrationWindowMixin:
    """Methods that operate on attributes supplied by MotionCaptureWindow."""

    def _calibration_snapshot(self) -> dict[str, tuple[object, float, int]]:
        snapshot: dict[str, tuple[object, float, int]] = {}
        for segment in self.config.enabled_segments:
            sensor_id = self.config.sensor_mapping[segment]
            sample = self.model.latest_samples.get(sensor_id)
            if sample is not None:
                snapshot[segment] = (
                    sample.quaternion.copy(),
                    sample.received_s,
                    sample.generation,
                )
        if snapshot:
            reference = max(snapshot.values(), key=lambda item: item[2])
            for segment in SEGMENT_NAMES:
                if segment not in self.config.enabled_segments:
                    snapshot[segment] = (
                        IDENTITY_QUATERNION.copy(),
                        reference[1],
                        reference[2],
                    )
        return snapshot

    def open_guided_calibration(self) -> None:
        if self.sock is None or not self.streaming_requested:
            QMessageBox.warning(
                self,
                "Нет активного потока",
                "Сначала подключитесь и нажмите START. Затем откройте "
                "комплексную калибровку, не останавливая поток.",
            )
            return
        if self.guided_dialog is not None:
            self.guided_dialog.raise_()
            self.guided_dialog.activateWindow()
            return
        dialog = GuidedCalibrationDialog(self._calibration_snapshot, self)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.result_ready.connect(self._apply_guided_calibration)
        dialog.destroyed.connect(self._guided_dialog_closed)
        self.guided_dialog = dialog
        dialog.show()

    def _guided_dialog_closed(self) -> None:
        self.guided_dialog = None

    def _application_profile_config(self) -> dict[str, object]:
        return {
            "device_ip": self.config.device_ip,
            "device_port": self.config.device_port,
            "stream_rate_hz": self.config.stream_rate_hz,
            "render_fps": self.config.render_fps,
            "sensor_id_mode": self.config.sensor_id_mode,
            "sensor_mapping_revision": SENSOR_MAPPING_REVISION,
            "axis_mapping_revision": AXIS_MAPPING_REVISION,
            "enabled_segments": sorted(self.config.enabled_segments),
            "sensor_mapping": dict(self.config.sensor_mapping),
        }

    def _apply_guided_calibration(self, result: CalibrationResult) -> None:
        new_config = self.config.copy()
        new_config.axis_maps = dict(result.axis_maps)
        self.apply_configuration(new_config)
        neutral_raw = {
            name: result.captures["n_pose"].average[name]
            for name in SEGMENT_NAMES
        }
        self.model.set_guided_calibration(
            result.axis_maps,
            result.drift_rates_rad_s,
            neutral_raw,
            result.reference_s,
            result.axis_alignment_quaternions,
        )
        self.last_calibration_result = result
        self.calibration_document = profile_document(
            self._application_profile_config(), result
        )
        self.needs_redraw = True
        worst_segment = max(result.scores_deg, key=result.scores_deg.__getitem__)
        maximum_error = max(result.scores_deg.values())
        self._append_monitor(
            f"[{self._clock()}] GUIDED CALIBRATION applied; "
            f"max_error={maximum_error:.2f} deg; "
            f"worst_segment={worst_segment}; "
            f"max_axis_alignment={result.max_axis_alignment_deg:.2f} deg; "
            f"max_drift={result.max_drift_deg_s:.4f} deg/s"
        )
        self.statusBar().showMessage(
            f"Комплексная калибровка применена: ошибка {maximum_error:.1f}°, "
            f"согласование осей {result.max_axis_alignment_deg:.1f}°, "
            f"дрейф {result.max_drift_deg_s:.3f}°/с",
            12000,
        )
        if maximum_error > HIGH_POSE_ERROR_WARNING_DEG:
            QMessageBox.warning(
                self,
                "Высокая ошибка поз",
                f"Ошибка направления для {worst_segment} достигла "
                f"{maximum_error:.1f}° (порог "
                f"{HIGH_POSE_ERROR_WARNING_DEG:.0f}°). "
                "Рекомендуется повторить мастер, точнее удерживая T-позу "
                "и прямые руки вперёд.",
            )

    def export_calibration_profile(self) -> None:
        if self.calibration_document is None:
            QMessageBox.information(
                self,
                "Нет профиля",
                "Сначала выполните комплексную калибровку или импортируйте профиль.",
            )
            return
        self.calibration_document["application"] = self._application_profile_config()
        suggested = "mocap_calibration_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".json"
        path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Сохранить профиль калибровки",
            suggested,
            "JSON-профиль (*.json)",
        )
        if not path:
            return
        if not path.lower().endswith(".json"):
            path += ".json"
        try:
            save_profile(path, self.calibration_document)
        except (OSError, TypeError, ValueError) as error:
            QMessageBox.critical(self, "Ошибка сохранения", str(error))
            return
        self.statusBar().showMessage(f"Профиль сохранён: {path}", 8000)

    def import_calibration_profile(self) -> None:
        path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Импортировать профиль калибровки",
            "",
            "JSON-профиль (*.json);;Все файлы (*)",
        )
        if not path:
            return
        try:
            document = load_profile(path)
            application = document["application"]
            calibration = document["calibration"]
            if not isinstance(application, dict) or not isinstance(calibration, dict):
                raise ValueError("некорректная структура профиля")
            new_config = self.config.copy()
            new_config.device_ip = str(
                application.get("device_ip", new_config.device_ip)
            )
            new_config.device_port = int(
                application.get("device_port", new_config.device_port)
            )
            new_config.stream_rate_hz = int(
                application.get("stream_rate_hz", new_config.stream_rate_hz)
            )
            new_config.render_fps = int(
                application.get("render_fps", new_config.render_fps)
            )
            new_config.sensor_id_mode = str(
                application.get("sensor_id_mode", new_config.sensor_id_mode)
            )
            raw_enabled = application.get("enabled_segments")
            if raw_enabled is not None:
                if not isinstance(raw_enabled, list):
                    raise ValueError("некорректный список активных датчиков")
                new_config.enabled_segments = {
                    str(name) for name in raw_enabled
                }
            mapping_migrated = False
            mapping = application.get("sensor_mapping")
            if isinstance(mapping, dict):
                new_config.sensor_mapping = {
                    name: int(mapping[name]) for name in SEGMENT_NAMES
                }
            mapping_revision = int(application.get("sensor_mapping_revision", 1))
            axis_mapping_revision = int(application.get("axis_mapping_revision", 1))
            if (
                mapping_revision < SENSOR_MAPPING_REVISION
                and new_config.sensor_mapping == LEGACY_SENSOR_MAPPING
            ):
                new_config.sensor_mapping = dict(DEFAULT_SENSOR_MAPPING)
                mapping_migrated = True
            raw_axis_maps = calibration["axis_maps"]
            raw_alignment = calibration.get(
                "axis_alignment_quaternions_wxyz"
            )
            raw_drift_rates = calibration["drift_rates_rad_s"]
            if not isinstance(raw_axis_maps, dict) or not isinstance(
                raw_drift_rates, dict
            ):
                raise ValueError("в профиле отсутствуют оси или дрейф")
            if raw_alignment is None:
                raw_alignment = {}
            if not isinstance(raw_alignment, dict):
                raise ValueError("некорректное согласование осей в профиле")
            new_config.axis_maps = {
                name: tuple(str(value) for value in raw_axis_maps[name])
                for name in SEGMENT_NAMES
            }
            axis_mapping_migrated = (
                axis_mapping_revision < AXIS_MAPPING_REVISION
                and new_config.axis_maps["spine"] == LEGACY_SPINE_AXIS_MAP
            )
            alignment_quaternions = {
                name: tuple(
                    float(value)
                    for value in raw_alignment.get(name, (1, 0, 0, 0))
                )
                for name in SEGMENT_NAMES
            }
            drift_rates = {
                name: tuple(float(value) for value in raw_drift_rates[name])
                for name in SEGMENT_NAMES
            }
            if mapping_migrated:
                new_config.axis_maps = dict(DEFAULT_AXIS_MAPS)
                alignment_quaternions = {
                    name: (1.0, 0.0, 0.0, 0.0) for name in SEGMENT_NAMES
                }
                drift_rates = {name: (0.0, 0.0, 0.0) for name in SEGMENT_NAMES}
            elif axis_mapping_migrated:
                new_config.axis_maps["spine"] = DEFAULT_AXIS_MAPS["spine"]
                alignment_quaternions["spine"] = (1.0, 0.0, 0.0, 0.0)
                drift_rates["spine"] = (0.0, 0.0, 0.0)
            if not new_config.device_ip.strip():
                raise ValueError("IP-адрес или имя устройства не может быть пустым")
            if not 1 <= new_config.device_port <= 65535:
                raise ValueError("UDP-порт должен быть в диапазоне 1…65535")
            if not 1 <= new_config.stream_rate_hz <= 100:
                raise ValueError("частота потока должна быть в диапазоне 1…100 Гц")
            if not 5 <= new_config.render_fps <= 60:
                raise ValueError("FPS интерфейса должен быть в диапазоне 5…60")
            if new_config.sensor_id_mode not in {"auto", "tca_channel", "sequential", "raw"}:
                raise ValueError("неизвестный режим ID датчиков")
            sensor_ids = tuple(new_config.sensor_mapping.values())
            if not new_config.enabled_segments:
                raise ValueError("должен быть включён хотя бы один датчик")
            unknown_segments = new_config.enabled_segments.difference(SEGMENT_NAMES)
            if unknown_segments:
                raise ValueError(
                    "неизвестные активные сегменты: "
                    + ", ".join(sorted(unknown_segments))
                )
            if any(sensor_id < 0 or sensor_id > 255 for sensor_id in sensor_ids):
                raise ValueError("ID датчиков должны быть в диапазоне 0…255")
            if len(set(sensor_ids)) != len(sensor_ids):
                raise ValueError("каждому сегменту нужен отдельный ID датчика")
            self.apply_configuration(new_config)
            self.model.set_axis_alignment(alignment_quaternions)
            self.model.set_drift_compensation(drift_rates)
            self.model.request_neutral()
        except (KeyError, OSError, TypeError, ValueError) as error:
            QMessageBox.critical(self, "Ошибка импорта", str(error))
            return
        profile_migrated = mapping_migrated or axis_mapping_migrated
        self.calibration_document = None if profile_migrated else document
        self.last_calibration_result = None
        self.needs_redraw = True
        if mapping_migrated:
            status_message = (
                "Руки в старом профиле переставлены. Выполните новую "
                "комплексную калибровку N → T → руки вперёд."
            )
            monitor_state = "arm mapping migrated; guided calibration required"
        elif axis_mapping_migrated:
            status_message = (
                "Направление осей spine в старом профиле исправлено. "
                "Если датчик корпуса используется, повторите калибровку."
            )
            monitor_state = "spine axis mapping migrated; calibration required"
        else:
            status_message = (
                "Профиль импортирован. Примите N-позу: она нужна заново после "
                "каждого запуска контроллера."
            )
            monitor_state = "fresh N-pose required"
        self._append_monitor(
            f"[{self._clock()}] PROFILE imported from {path}; {monitor_state}"
        )
        self.statusBar().showMessage(
            status_message,
            12000,
        )
