#!/usr/bin/env python3
"""PyQt6 application for ESP32/MPU6050 UDP motion capture."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import queue
import socket
import sys
import threading
import time
from typing import Collection, Final, Mapping

import matplotlib

matplotlib.use("QtAgg")

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from PyQt6.QtCore import QSettings, QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QAction, QCloseEvent, QKeySequence
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

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
    DEFAULT_ENABLED_SEGMENTS,
    DEFAULT_SENSOR_MAPPING,
    LEGACY_SPINE_AXIS_MAP,
    LEGACY_SENSOR_MAPPING,
    SEGMENT_NAMES,
    SENSOR_MAPPING_REVISION,
    MotionCaptureModel,
    axis_map_matrix,
    compute_body_pose,
)


from .window_calibration import CalibrationWindowMixin
DEFAULT_DEVICE_IP: Final = "192.168.1.117"
DEFAULT_DEVICE_PORT: Final = 4210
DEFAULT_STREAM_RATE_HZ: Final = 10
DEFAULT_RENDER_FPS: Final = 30
AXIS_OPTIONS: Final = ("+X", "-X", "+Y", "-Y", "+Z", "-Z")

SEGMENT_TITLES: Final = {
    "spine": "Корпус",
    "shoulder.L": "Левое плечо",
    "forearm.L": "Левое предплечье",
    "shoulder.R": "Правое плечо",
    "forearm.R": "Правое предплечье",
}
ID_MODE_TITLES: Final = {
    "auto": "Авто: TCA или последовательные 1…5",
    "tca_channel": "Физические каналы TCA: 0, 1, 2, 6, 7",
    "sequential": "Последовательные ID: 1…5",
    "raw": "Без преобразования ID",
}


@dataclass
class ViewerConfig:
    device_ip: str
    device_port: int
    stream_rate_hz: int
    render_fps: int
    sensor_id_mode: str
    enabled_segments: set[str]
    sensor_mapping: dict[str, int]
    axis_maps: dict[str, tuple[str, str, str]]

    def copy(self) -> "ViewerConfig":
        return ViewerConfig(
            self.device_ip,
            self.device_port,
            self.stream_rate_hz,
            self.render_fps,
            self.sensor_id_mode,
            set(self.enabled_segments),
            dict(self.sensor_mapping),
            dict(self.axis_maps),
        )


def default_config() -> ViewerConfig:
    return ViewerConfig(
        DEFAULT_DEVICE_IP,
        DEFAULT_DEVICE_PORT,
        DEFAULT_STREAM_RATE_HZ,
        DEFAULT_RENDER_FPS,
        "auto",
        set(DEFAULT_ENABLED_SEGMENTS),
        dict(DEFAULT_SENSOR_MAPPING),
        dict(DEFAULT_AXIS_MAPS),
    )


def load_config(settings: QSettings) -> ViewerConfig:
    config = default_config()
    try:
        config.device_ip = str(settings.value("network/device_ip", config.device_ip))
        config.device_port = int(settings.value("network/device_port", config.device_port))
        config.stream_rate_hz = int(
            settings.value("display/stream_rate_hz", config.stream_rate_hz)
        )
        config.render_fps = int(settings.value("display/render_fps", config.render_fps))
        config.sensor_id_mode = str(
            settings.value("sensors/id_mode", config.sensor_id_mode)
        )
        mapping_revision = int(settings.value("sensors/mapping_revision", 1))
        axis_mapping_revision = int(settings.value("axes/mapping_revision", 1))
        for name in SEGMENT_NAMES:
            enabled = settings.value(
                f"sensors/enabled/{name}",
                name in config.enabled_segments,
                type=bool,
            )
            if enabled:
                config.enabled_segments.add(name)
            else:
                config.enabled_segments.discard(name)
            config.sensor_mapping[name] = int(
                settings.value(f"sensors/{name}", config.sensor_mapping[name])
            )
            raw_axes = str(
                settings.value(f"axes/{name}", ",".join(config.axis_maps[name]))
            )
            config.axis_maps[name] = tuple(
                value.strip().upper() for value in raw_axes.split(",")
            )  # type: ignore[assignment]
        if (
            mapping_revision < SENSOR_MAPPING_REVISION
            and config.sensor_mapping == LEGACY_SENSOR_MAPPING
        ):
            config.sensor_mapping = dict(DEFAULT_SENSOR_MAPPING)
            config.axis_maps = dict(DEFAULT_AXIS_MAPS)
        elif (
            axis_mapping_revision < AXIS_MAPPING_REVISION
            and config.axis_maps["spine"] == LEGACY_SPINE_AXIS_MAP
        ):
            config.axis_maps["spine"] = DEFAULT_AXIS_MAPS["spine"]
        if not config.device_ip.strip():
            raise ValueError
        if not 1 <= config.device_port <= 65535:
            raise ValueError
        config.stream_rate_hz = min(100, max(1, config.stream_rate_hz))
        config.render_fps = min(60, max(5, config.render_fps))
        model = MotionCaptureModel(
            config.sensor_mapping, config.axis_maps, config.sensor_id_mode
        )
        model.set_enabled_segments(config.enabled_segments)
    except (TypeError, ValueError):
        return default_config()
    return config


def save_config(settings: QSettings, config: ViewerConfig) -> None:
    settings.setValue("network/device_ip", config.device_ip)
    settings.setValue("network/device_port", config.device_port)
    settings.setValue("display/stream_rate_hz", config.stream_rate_hz)
    settings.setValue("display/render_fps", config.render_fps)
    settings.setValue("sensors/id_mode", config.sensor_id_mode)
    settings.setValue("sensors/mapping_revision", SENSOR_MAPPING_REVISION)
    settings.setValue("axes/mapping_revision", AXIS_MAPPING_REVISION)
    for name in SEGMENT_NAMES:
        settings.setValue(
            f"sensors/enabled/{name}", name in config.enabled_segments
        )
        settings.setValue(f"sensors/{name}", config.sensor_mapping[name])
        settings.setValue(f"axes/{name}", ",".join(config.axis_maps[name]))
    settings.sync()


class SettingsDialog(QDialog):
    """Separate window for network, mapping, axes, and frame rates."""

    config_applied = pyqtSignal(object)

    def __init__(self, config: ViewerConfig, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.config = config.copy()
        self.enabled_checks: dict[str, QCheckBox] = {}
        self.mapping_spins: dict[str, QSpinBox] = {}
        self.axis_combos: dict[str, tuple[QComboBox, QComboBox, QComboBox]] = {}
        self.setWindowTitle("Настройки MPU UDP Viewer")
        self.setMinimumSize(720, 500)

        root = QVBoxLayout(self)
        tabs = QTabWidget()
        tabs.addTab(self._network_tab(), "Сеть")
        tabs.addTab(self._sensors_tab(), "Датчики")
        tabs.addTab(self._axes_tab(), "Оси")
        tabs.addTab(self._frame_rate_tab(), "Частота кадров")
        root.addWidget(tabs)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.Apply
        )
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self._apply)
        root.addWidget(buttons)

    def _network_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        self.ip_edit = QLineEdit(self.config.device_ip)
        self.ip_edit.setPlaceholderText("192.168.1.117")
        self.port_spin = QSpinBox()
        self.port_spin.setRange(1, 65535)
        self.port_spin.setValue(self.config.device_port)
        form.addRow("IP-адрес или имя ESP32:", self.ip_edit)
        form.addRow("UDP-порт:", self.port_spin)
        note = QLabel(
            "Новый адрес применяется сразу. Открытое соединение будет "
            "автоматически пересоздано."
        )
        note.setWordWrap(True)
        form.addRow(note)
        return tab

    def _sensors_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Формат ID в пакетах:"))
        self.mode_combo = QComboBox()
        for mode, title in ID_MODE_TITLES.items():
            self.mode_combo.addItem(title, mode)
        self.mode_combo.setCurrentIndex(
            max(0, self.mode_combo.findData(self.config.sensor_id_mode))
        )
        mode_row.addWidget(self.mode_combo, 1)
        layout.addLayout(mode_row)

        table = QTableWidget(len(SEGMENT_NAMES), 4)
        table.setHorizontalHeaderLabels(
            ("Сегмент тела", "Имя из Blender", "Смотреть", "Канонический ID")
        )
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        for row, name in enumerate(SEGMENT_NAMES):
            table.setItem(row, 0, QTableWidgetItem(SEGMENT_TITLES[name]))
            table.setItem(row, 1, QTableWidgetItem(name))
            enabled = QCheckBox()
            enabled.setChecked(name in self.config.enabled_segments)
            table.setCellWidget(row, 2, enabled)
            self.enabled_checks[name] = enabled
            spin = QSpinBox()
            spin.setRange(0, 255)
            spin.setValue(self.config.sensor_mapping[name])
            table.setCellWidget(row, 3, spin)
            self.mapping_spins[name] = spin
        table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.ResizeToContents
        )
        layout.addWidget(table)
        table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.ResizeMode.ResizeToContents
        )
        note = QLabel(
            "Авто-режим переводит прошивочные ID 1…5 в канонические "
            "0, 1, 2, 6, 7. Снятый флажок исключает датчик из визуализации "
            "и калибровки."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        return tab

    def _axes_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        table = QTableWidget(len(SEGMENT_NAMES), 4)
        table.setHorizontalHeaderLabels(
            ("Имя из Blender", "X получает", "Y получает", "Z получает")
        )
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        for row, name in enumerate(SEGMENT_NAMES):
            table.setItem(row, 0, QTableWidgetItem(name))
            combos: list[QComboBox] = []
            for column, selected in enumerate(self.config.axis_maps[name], start=1):
                combo = QComboBox()
                combo.addItems(AXIS_OPTIONS)
                combo.setCurrentText(selected)
                table.setCellWidget(row, column, combo)
                combos.append(combo)
            self.axis_combos[name] = (combos[0], combos[1], combos[2])
        table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        for column in range(1, 4):
            table.horizontalHeader().setSectionResizeMode(
                column, QHeaderView.ResizeMode.ResizeToContents
            )
        layout.addWidget(table)
        note = QLabel(
            "Настройка совпадает с AXIS_MAPS Blender-драйвера. Каждая исходная "
            "ось X/Y/Z должна использоваться один раз. После изменения нужна N-поза."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        return tab

    def _frame_rate_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        self.stream_rate_spin = QSpinBox()
        self.stream_rate_spin.setRange(1, 100)
        self.stream_rate_spin.setSuffix(" кадр/с")
        self.stream_rate_spin.setValue(self.config.stream_rate_hz)
        self.render_fps_spin = QSpinBox()
        self.render_fps_spin.setRange(5, 60)
        self.render_fps_spin.setSuffix(" FPS")
        self.render_fps_spin.setValue(self.config.render_fps)
        form.addRow("Поток ESP32 (SET_RATE):", self.stream_rate_spin)
        form.addRow("Перерисовка человека:", self.render_fps_spin)
        note = QLabel(
            "Обе частоты меняются без перезапуска. Частота ESP32 относится к "
            "измерениям, FPS — только к нагрузке графического интерфейса."
        )
        note.setWordWrap(True)
        form.addRow(note)
        return tab

    def _collect(self) -> ViewerConfig | None:
        host = self.ip_edit.text().strip()
        if not host:
            QMessageBox.warning(self, "Адрес", "Введите IP-адрес или имя ESP32.")
            return None
        enabled_segments = {
            name for name, checkbox in self.enabled_checks.items()
            if checkbox.isChecked()
        }
        if not enabled_segments:
            QMessageBox.warning(self, "Датчики", "Включите хотя бы один датчик.")
            return None
        mapping = {name: spin.value() for name, spin in self.mapping_spins.items()}
        if len(set(mapping.values())) != len(mapping):
            QMessageBox.warning(
                self, "Датчики", "Каждому сегменту нужен отдельный ID датчика."
            )
            return None
        axes: dict[str, tuple[str, str, str]] = {}
        try:
            for name, combos in self.axis_combos.items():
                values = tuple(combo.currentText() for combo in combos)
                axis_map_matrix(values)
                axes[name] = values  # type: ignore[assignment]
        except ValueError as error:
            QMessageBox.warning(self, "Оси", str(error))
            return None
        return ViewerConfig(
            host,
            self.port_spin.value(),
            self.stream_rate_spin.value(),
            self.render_fps_spin.value(),
            str(self.mode_combo.currentData()),
            enabled_segments,
            mapping,
            axes,
        )

    def _apply(self) -> bool:
        config = self._collect()
        if config is None:
            return False
        self.config = config.copy()
        self.config_applied.emit(config)
        return True

    def _accept(self) -> None:
        if self._apply():
            self.accept()


class HumanCanvas(QWidget):
    """Matplotlib 3D skeleton with tracked segment names and local axes."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.figure = Figure(figsize=(8.5, 7.0), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.axes = self.figure.add_subplot(111, projection="3d")
        layout.addWidget(self.canvas, 1)
        layout.addWidget(NavigationToolbar2QT(self.canvas, self))
        self.tracked_lines: dict[str, Line2D] = {}
        self.static_lines: list[Line2D] = []
        self.axis_lines: dict[str, tuple[Line2D, Line2D, Line2D]] = {}
        self.segment_labels: dict[str, object] = {}
        self._configure_plot()
        self._create_artists()

    def _configure_plot(self) -> None:
        self.figure.patch.set_facecolor("#f3f6fa")
        self.axes.set_facecolor("#f3f6fa")
        self.axes.set_title(
            "Скелетная модель и ориентация сегментов",
            pad=16,
            fontsize=13,
            fontweight="bold",
        )
        self.axes.set(
            xlim=(-1.05, 1.05),
            ylim=(-0.78, 0.78),
            zlim=(0.0, 1.95),
            xlabel="X — вправо",
            ylabel="Y — вперёд",
            zlabel="Z — вверх",
        )
        self.axes.set_box_aspect((2.1, 1.55, 1.95))
        # Front view: the subject's anatomical left is on the viewer's right.
        self.axes.view_init(elev=13, azim=108)
        self.axes.grid(True, alpha=0.24)
        self.axes.plot(
            (-0.65, 0.65, 0.65, -0.65, -0.65),
            (-0.45, -0.45, 0.35, 0.35, -0.45),
            (0.0, 0.0, 0.0, 0.0, 0.0),
            color="#aeb9c7",
            linewidth=1.0,
            alpha=0.65,
        )
        self.axes.legend(
            handles=(
                Line2D((0,), (0,), color="#d52b2b", lw=2.5, label="X"),
                Line2D((0,), (0,), color="#2a9d55", lw=2.5, label="Y"),
                Line2D((0,), (0,), color="#2676d2", lw=2.5, label="Z"),
            ),
            title="Локальные оси",
            loc="upper left",
        )

    def _create_artists(self) -> None:
        pose = compute_body_pose({})
        colors = {
            "spine": "#e08c68",
            "shoulder.L": "#e08c68",
            "forearm.L": "#efad86",
            "shoulder.R": "#e08c68",
            "forearm.R": "#efad86",
        }
        widths = {
            "spine": 5.0,
            "shoulder.L": 5.0,
            "forearm.L": 4.0,
            "shoulder.R": 5.0,
            "forearm.R": 4.0,
        }
        for name in SEGMENT_NAMES:
            start, end = pose.tracked_segments[name]
            self.tracked_lines[name] = self.axes.plot(
                (start[0], end[0]),
                (start[1], end[1]),
                (start[2], end[2]),
                color=colors[name],
                linewidth=widths[name],
                solid_capstyle="round",
                zorder=5,
            )[0]
        for start, end in pose.static_segments:
            self.static_lines.append(
                self.axes.plot(
                    (start[0], end[0]),
                    (start[1], end[1]),
                    (start[2], end[2]),
                    color="#40536a",
                    linewidth=3.4,
                    solid_capstyle="round",
                    zorder=3,
                )[0]
            )
        self.joints = self.axes.scatter(
            [point[0] for point in pose.joints],
            [point[1] for point in pose.joints],
            [point[2] for point in pose.joints],
            s=28,
            color="#26384d",
            depthshade=True,
            zorder=6,
        )
        self.head = self.axes.scatter(
            (pose.head_center[0],),
            (pose.head_center[1],),
            (pose.head_center[2],),
            s=260,
            color="#f3f6fa",
            edgecolor="#26384d",
            linewidth=2.0,
            depthshade=True,
            zorder=7,
        )
        for name in SEGMENT_NAMES:
            self.axis_lines[name] = tuple(
                self.axes.plot([], [], [], color=color, linewidth=2.2, zorder=9)[0]
                for color in ("#d52b2b", "#2a9d55", "#2676d2")
            )  # type: ignore[assignment]
            origin = pose.axis_origins[name]
            self.segment_labels[name] = self.axes.text(
                origin[0],
                origin[1],
                origin[2],
                name,
                fontsize=8,
                color="#172535",
                ha="left",
                va="bottom",
                zorder=10,
            )
        self.update_pose({}, DEFAULT_SENSOR_MAPPING)

    def update_pose(
        self,
        orientations: Mapping[str, object],
        sensor_mapping: Mapping[str, int],
        enabled_segments: Collection[str] = DEFAULT_ENABLED_SEGMENTS,
    ) -> None:
        pose = compute_body_pose(orientations)  # type: ignore[arg-type]
        enabled = set(enabled_segments)
        for name, (start, end) in pose.tracked_segments.items():
            self.tracked_lines[name].set_data_3d(
                (start[0], end[0]), (start[1], end[1]), (start[2], end[2])
            )
        for line, (start, end) in zip(
            self.static_lines, pose.static_segments, strict=True
        ):
            line.set_data_3d(
                (start[0], end[0]), (start[1], end[1]), (start[2], end[2])
            )
        self.joints._offsets3d = (
            [point[0] for point in pose.joints],
            [point[1] for point in pose.joints],
            [point[2] for point in pose.joints],
        )
        self.head._offsets3d = (
            (pose.head_center[0],),
            (pose.head_center[1],),
            (pose.head_center[2],),
        )
        for name in SEGMENT_NAMES:
            active = name in enabled
            show_axes = active or name == "spine"
            self.tracked_lines[name].set_alpha(1.0 if active else 0.22)
            for line in self.axis_lines[name]:
                line.set_visible(show_axes)
                line.set_alpha(1.0 if active else 0.55)
            label = self.segment_labels[name]
            label.set_visible(show_axes)
            label.set_alpha(1.0 if active else 0.65)
            if not show_axes:
                continue
            origin = pose.axis_origins[name]
            frame = pose.axis_frames[name]
            for axis_index, line in enumerate(self.axis_lines[name]):
                end = origin + frame[:, axis_index] * 0.17
                line.set_data_3d(
                    (origin[0], end[0]),
                    (origin[1], end[1]),
                    (origin[2], end[2]),
                )
            state = "" if active else ""
            label.set_text(f"{name}  [S{sensor_mapping[name]}{state}]")
            label.set_position((origin[0] + 0.025, origin[1] + 0.025))
            label.set_3d_properties(origin[2] + 0.035)
        self.canvas.draw_idle()


class MotionCaptureWindow(CalibrationWindowMixin, QMainWindow):
    """Main GUI, UDP socket owner, and bridge to the pure motion model."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("MPU6050 UDP — PyQt motion capture")
        self.resize(1500, 900)
        self.setMinimumSize(1080, 680)
        self.settings_store = QSettings("Neuromorph", "MPU UDP Viewer Guided")
        self.config = load_config(self.settings_store)
        self.model = MotionCaptureModel(
            self.config.sensor_mapping,
            self.config.axis_maps,
            self.config.sensor_id_mode,
        )
        self.model.set_enabled_segments(self.config.enabled_segments)
        self.sock: socket.socket | None = None
        self.receiver_thread: threading.Thread | None = None
        self.receiver_stop: threading.Event | None = None
        self.send_lock = threading.Lock()
        self.events: queue.Queue[tuple[str, float, object, object]] = queue.Queue()
        self.settings_dialog: SettingsDialog | None = None
        self.guided_dialog: GuidedCalibrationDialog | None = None
        self.last_calibration_result: CalibrationResult | None = None
        self.calibration_document: dict[str, object] | None = None
        self.streaming_requested = False
        self.hardware_calibration_pending = False
        self.needs_redraw = True
        self._build_actions()
        self._build_menu()
        self._build_ui()
        self._set_connected_controls(False)
        self.event_timer = QTimer(self)
        self.event_timer.timeout.connect(self._process_events)
        self.event_timer.start(20)
        self.render_timer = QTimer(self)
        self.render_timer.timeout.connect(self._refresh_plot)
        self._apply_render_fps()

    def _build_actions(self) -> None:
        self.connect_action = QAction("Подключиться", self)
        self.connect_action.setShortcut(QKeySequence("Ctrl+O"))
        self.connect_action.triggered.connect(self.connect_device)
        self.disconnect_action = QAction("Отключиться", self)
        self.disconnect_action.triggered.connect(self.disconnect_device)
        self.import_profile_action = QAction("Импорт профиля калибровки…", self)
        self.import_profile_action.setShortcut(QKeySequence("Ctrl+I"))
        self.import_profile_action.triggered.connect(self.import_calibration_profile)
        self.export_profile_action = QAction("Сохранить профиль калибровки…", self)
        self.export_profile_action.setShortcut(QKeySequence("Ctrl+Shift+S"))
        self.export_profile_action.triggered.connect(self.export_calibration_profile)
        self.settings_action = QAction("Настройки…", self)
        self.settings_action.setShortcut(QKeySequence("Ctrl+,"))
        self.settings_action.triggered.connect(self.open_settings)
        self.neutral_action = QAction("Запомнить N-позу", self)
        self.neutral_action.setShortcut(QKeySequence("Ctrl+K"))
        self.neutral_action.triggered.connect(self.capture_neutral_pose)
        self.start_action = QAction("START", self)
        self.start_action.triggered.connect(self.start_stream)
        self.stop_action = QAction("STOP", self)
        self.stop_action.triggered.connect(self.stop_stream)
        self.calibration_action = QAction("КАЛИБРОВКА…", self)
        self.calibration_action.triggered.connect(self.open_calibration_dialog)
        self.guided_action = QAction("Комплексная N → T → вперёд…", self)
        self.guided_action.triggered.connect(self.open_guided_calibration)
        self.status_action = QAction("Запросить STATUS", self)
        self.status_action.triggered.connect(lambda: self.send_command("STATUS"))

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("Файл")
        file_menu.addAction(self.connect_action)
        file_menu.addAction(self.disconnect_action)
        file_menu.addSeparator()
        exit_action = QAction("Выход", self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(self.import_profile_action)
        file_menu.addAction(self.export_profile_action)
        file_menu.addSeparator()
        file_menu.addAction(exit_action)
        edit_menu = self.menuBar().addMenu("Правка")
        edit_menu.addAction(self.settings_action)
        edit_menu.addAction(self.neutral_action)
        device_menu = self.menuBar().addMenu("Устройство")
        device_menu.addActions(
            (
                self.start_action,
                self.stop_action,
                self.calibration_action,
                self.guided_action,
                self.status_action,
            )
        )
        help_menu = self.menuBar().addMenu("Справка")
        about_action = QAction("О программе", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def _build_ui(self) -> None:
        central = QWidget()
        outer = QVBoxLayout(central)
        outer.setContentsMargins(10, 8, 10, 10)
        outer.setSpacing(7)
        controls = QFrame()
        controls.setFrameShape(QFrame.Shape.StyledPanel)
        bar = QHBoxLayout(controls)
        bar.setContentsMargins(8, 7, 8, 7)
        self.connect_button = QPushButton("Подключиться")
        self.connect_button.clicked.connect(self.connect_device)
        self.disconnect_button = QPushButton("Отключиться")
        self.disconnect_button.clicked.connect(self.disconnect_device)
        self.start_button = QPushButton("START")
        self.start_button.setObjectName("startButton")
        self.start_button.clicked.connect(self.start_stream)
        self.stop_button = QPushButton("STOP")
        self.stop_button.setObjectName("stopButton")
        self.stop_button.clicked.connect(self.stop_stream)
        self.calibration_button = QPushButton("КАЛИБРОВКА")
        self.calibration_button.clicked.connect(self.open_calibration_dialog)
        settings_button = QPushButton("Настройки…")
        settings_button.clicked.connect(self.open_settings)
        for button in (
            self.connect_button,
            self.disconnect_button,
            self.start_button,
            self.stop_button,
            self.calibration_button,
            settings_button,
        ):
            bar.addWidget(button)
        bar.addSpacing(10)
        self.connection_label = QLabel("Не подключено")
        self.connection_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        bar.addWidget(self.connection_label, 1)
        self.packet_label = QLabel("UDP: 0 · кадров: 0 · Q: 0")
        bar.addWidget(self.packet_label)
        outer.addWidget(controls)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.human_canvas = HumanCanvas()
        splitter.addWidget(self.human_canvas)
        monitor_panel = QWidget()
        monitor_layout = QVBoxLayout(monitor_panel)
        monitor_layout.setContentsMargins(4, 0, 0, 0)
        monitor_header = QHBoxLayout()
        monitor_header.addWidget(QLabel("<b>Монитор UDP-пакетов</b>"))
        monitor_header.addStretch(1)
        clear_button = QPushButton("Очистить")
        clear_button.clicked.connect(lambda: self.monitor.clear())
        monitor_header.addWidget(clear_button)
        monitor_layout.addLayout(monitor_header)
        self.monitor = QPlainTextEdit()
        self.monitor.setReadOnly(True)
        self.monitor.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.monitor.document().setMaximumBlockCount(4000)
        font = self.monitor.font()
        font.setFamilies(("DejaVu Sans Mono", "monospace"))
        font.setPointSize(9)
        self.monitor.setFont(font)
        monitor_layout.addWidget(self.monitor, 1)
        command_box = QGroupBox("Произвольная команда")
        command_layout = QHBoxLayout(command_box)
        self.command_edit = QLineEdit()
        self.command_edit.setPlaceholderText("PING, STATUS, GET_CONFIG…")
        self.command_edit.returnPressed.connect(self.send_manual_command)
        send_button = QPushButton("Отправить")
        send_button.clicked.connect(self.send_manual_command)
        command_layout.addWidget(self.command_edit, 1)
        command_layout.addWidget(send_button)
        monitor_layout.addWidget(command_box)
        splitter.addWidget(monitor_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes((900, 500))
        outer.addWidget(splitter, 1)
        self.setCentralWidget(central)
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage(
            f"Настройки: {self.config.device_ip}:{self.config.device_port}, "
            f"поток {self.config.stream_rate_hz} кадр/с"
        )
        self.setStyleSheet(
            "QPushButton { min-height: 27px; padding: 2px 10px; }"
            "QPushButton#startButton { color: #176b35; font-weight: 700; }"
            "QPushButton#stopButton { color: #a52d2d; font-weight: 700; }"
        )

    def _set_connected_controls(self, connected: bool) -> None:
        for widget in (self.connect_button, self.connect_action):
            widget.setEnabled(not connected)
        for widget in (
            self.disconnect_button,
            self.disconnect_action,
            self.start_button,
            self.start_action,
            self.stop_button,
            self.stop_action,
            self.calibration_button,
            self.calibration_action,
            self.guided_action,
            self.status_action,
        ):
            widget.setEnabled(connected)

    def open_settings(self) -> None:
        if self.settings_dialog is not None:
            self.settings_dialog.raise_()
            self.settings_dialog.activateWindow()
            return
        dialog = SettingsDialog(self.config, self)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.config_applied.connect(self.apply_configuration)
        dialog.destroyed.connect(self._settings_closed)
        self.settings_dialog = dialog
        dialog.show()

    def _settings_closed(self) -> None:
        self.settings_dialog = None

    def apply_configuration(self, new: ViewerConfig) -> None:
        old = self.config
        endpoint_changed = (old.device_ip, old.device_port) != (
            new.device_ip,
            new.device_port,
        )
        model_changed = (
            old.sensor_id_mode != new.sensor_id_mode
            or old.sensor_mapping != new.sensor_mapping
            or old.axis_maps != new.axis_maps
            or old.enabled_segments != new.enabled_segments
        )
        rate_changed = old.stream_rate_hz != new.stream_rate_hz
        render_changed = old.render_fps != new.render_fps
        was_connected = self.sock is not None
        was_streaming = self.streaming_requested
        self.config = new.copy()
        save_config(self.settings_store, self.config)
        if model_changed:
            self.model.configure(
                self.config.sensor_mapping,
                self.config.axis_maps,
                self.config.sensor_id_mode,
            )
            self.model.set_enabled_segments(self.config.enabled_segments)
            self.last_calibration_result = None
            self.calibration_document = None
            self.needs_redraw = True
            self._append_monitor(
                f"[{self._clock()}] CONFIG sensors/mapping/axes changed; N-pose required"
            )
        if render_changed:
            self._apply_render_fps()
        if endpoint_changed and was_connected:
            self.disconnect_device(log=True)
            if self.connect_device() and was_streaming:
                QTimer.singleShot(180, self.start_stream)
        elif rate_changed and was_connected:
            self.send_command(f"SET_RATE {self.config.stream_rate_hz}")
        self.statusBar().showMessage(
            f"Настройки применены: {self.config.device_ip}:{self.config.device_port}, "
            f"поток {self.config.stream_rate_hz} кадр/с, GUI {self.config.render_fps} FPS",
            6000,
        )

    def _apply_render_fps(self) -> None:
        interval_ms = max(1, round(1000 / self.config.render_fps))
        if self.render_timer.isActive():
            self.render_timer.setInterval(interval_ms)
        else:
            self.render_timer.start(interval_ms)

    def connect_device(self) -> bool:
        self.disconnect_device(log=False)
        sock: socket.socket | None = None
        try:
            endpoint = socket.getaddrinfo(
                self.config.device_ip,
                self.config.device_port,
                socket.AF_INET,
                socket.SOCK_DGRAM,
            )[0][4]
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
            sock.bind(("0.0.0.0", 0))
            sock.connect(endpoint)
            sock.settimeout(0.25)
        except OSError as error:
            if sock is not None:
                sock.close()
            QMessageBox.critical(self, "Ошибка UDP", str(error))
            return False
        stop_event = threading.Event()
        self.sock = sock
        self.receiver_stop = stop_event
        self.receiver_thread = threading.Thread(
            target=self._receive_loop,
            args=(sock, stop_event),
            name="pyqt-udp-mocap",
            daemon=True,
        )
        self.receiver_thread.start()
        local_ip, local_port = sock.getsockname()
        self.connection_label.setText(
            f"UDP {endpoint[0]}:{endpoint[1]} · локальный порт {local_port}"
        )
        self._set_connected_controls(True)
        self._append_monitor(
            f"[{self._clock()}] OPEN {local_ip}:{local_port} → "
            f"{endpoint[0]}:{endpoint[1]}"
        )
        self.send_command("HELLO", warn=False)
        QTimer.singleShot(100, lambda: self.send_command("STATUS", warn=False))
        return True

    def disconnect_device(self, _checked: bool = False, log: bool = True) -> None:
        sock = self.sock
        self.sock = None
        stop_event = self.receiver_stop
        self.receiver_stop = None
        if stop_event is not None:
            stop_event.set()
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
        thread = self.receiver_thread
        self.receiver_thread = None
        if (
            thread is not None
            and thread.is_alive()
            and thread is not threading.current_thread()
        ):
            thread.join(timeout=0.4)
        self.streaming_requested = False
        self.hardware_calibration_pending = False
        self.connection_label.setText("Не подключено")
        self._set_connected_controls(False)
        if log and sock is not None:
            self._append_monitor(f"[{self._clock()}] CLOSE локальный UDP-сокет")

    def _receive_loop(
        self, sock: socket.socket, stop_event: threading.Event
    ) -> None:
        while not stop_event.is_set() and self.sock is sock:
            try:
                payload, address = sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError as error:
                if not stop_event.is_set() and self.sock is sock:
                    self.events.put(("error", time.time(), str(error), None))
                break
            self.events.put(("packet", time.time(), payload, address))

    def send_command(self, command: str, warn: bool = True) -> bool:
        command = command.strip()
        if not command:
            return False
        sock = self.sock
        if sock is None:
            if warn:
                QMessageBox.warning(
                    self, "Нет соединения", "Сначала подключитесь к ESP32."
                )
            return False
        try:
            payload = (command + "\n").encode("ascii")
        except UnicodeEncodeError:
            QMessageBox.warning(
                self, "Некорректная команда", "Команда должна быть в ASCII."
            )
            return False
        try:
            with self.send_lock:
                sock.send(payload)
        except OSError as error:
            self._append_monitor(f"[{self._clock()}] TX ERROR {error}")
            return False
        self._append_monitor(f"[{self._clock()}] TX ({len(payload)} B)\n{command}")
        return True

    def send_manual_command(self) -> None:
        if self.send_command(self.command_edit.text()):
            self.command_edit.clear()

    def start_stream(self) -> None:
        if self.sock is None:
            QMessageBox.warning(
                self, "Нет соединения", "Сначала подключитесь к ESP32."
            )
            return
        self.streaming_requested = True
        self.model.request_neutral()
        self.needs_redraw = True
        self.send_command(f"SET_RATE {self.config.stream_rate_hz}", warn=False)
        current_sock = self.sock
        QTimer.singleShot(
            80,
            lambda: self.send_command("START", warn=False)
            if self.sock is current_sock
            else None,
        )
        self.statusBar().showMessage(
            "START: примите N-позу и не двигайтесь до захвата общего кадра", 7000
        )

    def stop_stream(self) -> None:
        if self.send_command("STOP"):
            self.streaming_requested = False
            self.statusBar().showMessage("Команда STOP отправлена", 4000)

    def capture_neutral_pose(self) -> None:
        if self.sock is None:
            QMessageBox.warning(
                self, "Нет соединения", "Для N-позы нужен активный поток ESP32."
            )
            return
        self.model.request_neutral()
        self.needs_redraw = True
        self._append_monitor(
            f"[{self._clock()}] N-POSE requested: корпус прямо, руки вниз"
        )
        self.statusBar().showMessage(
            "Ожидание свежего общего кадра для N-позы…", 7000
        )

    def open_calibration_dialog(self) -> None:
        box = QMessageBox(self)
        box.setWindowTitle("Калибровка")
        box.setIcon(QMessageBox.Icon.Question)
        box.setText("Выберите тип калибровки")
        box.setInformativeText(
            "N-поза не меняет offsets MPU6050. CALIB_GYRO подходит для надетых "
            "датчиков. Полная CALIB — только для снятых модулей, лежащих Z вверх."
        )
        neutral = box.addButton("Только N-поза", QMessageBox.ButtonRole.AcceptRole)
        gyro = box.addButton("Гироскоп + N-поза", QMessageBox.ButtonRole.ActionRole)
        guided = box.addButton(
            "Комплексная N → T → вперёд", QMessageBox.ButtonRole.ActionRole
        )
        full = box.addButton(
            "Полная MPU6050", QMessageBox.ButtonRole.DestructiveRole
        )
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.exec()
        if box.clickedButton() is guided:
            self.open_guided_calibration()
        elif box.clickedButton() is neutral:
            self.capture_neutral_pose()
        elif box.clickedButton() is gyro and self.send_command("CALIB_GYRO"):
            self.hardware_calibration_pending = True
            self.statusBar().showMessage(
                "CALIB_GYRO: не двигайтесь до ACK … DONE", 10000
            )
        elif box.clickedButton() is full:
            answer = QMessageBox.warning(
                self,
                "Полная калибровка",
                "Все датчики сняты, лежат неподвижно локальной Z вверх?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer == QMessageBox.StandardButton.Yes and self.send_command("CALIB"):
                self.hardware_calibration_pending = True

    def _process_events(self) -> None:
        for _ in range(250):
            try:
                event_type, timestamp, payload, address = self.events.get_nowait()
            except queue.Empty:
                break
            if event_type == "error":
                self._append_monitor(
                    f"[{self._format_time(timestamp)}] RX ERROR {payload}"
                )
                continue
            raw = payload if isinstance(payload, bytes) else bytes()
            remote = address if isinstance(address, tuple) else ("?", 0)
            text = raw.decode("utf-8", errors="replace").rstrip("\x00\r\n")
            self.connection_label.setText(f"Подключено к {remote[0]}:{remote[1]}")
            self._append_monitor(
                f"[{self._format_time(timestamp)}] RX {remote[0]}:{remote[1]} "
                f"({len(raw)} B)\n{text}"
            )
            result = self.model.handle_datagram(raw, time.monotonic())
            self.needs_redraw = self.needs_redraw or result.pose_changed
            if result.neutral_captured:
                self._append_monitor(
                    f"[{self._clock()}] N-POSE captured for "
                    f"{len(self.config.enabled_segments)} enabled segments"
                )
                self.statusBar().showMessage(
                    "N-поза захвачена для включённых датчиков",
                    6000,
                )
            for message in result.messages:
                if message.startswith(
                    ("ACK CALIB DONE", "ACK CALIB_GYRO DONE")
                ) and self.hardware_calibration_pending:
                    self.hardware_calibration_pending = False
                    self.model.set_drift_compensation(
                        {name: (0.0, 0.0, 0.0) for name in SEGMENT_NAMES}
                    )
                    self.last_calibration_result = None
                    self.calibration_document = None
                    self.model.request_neutral()
                    self.needs_redraw = True
                    self.statusBar().showMessage(
                        "Аппаратная калибровка завершена; примите N-позу", 9000
                    )
        mode = self.model.active_sensor_id_mode or (
            "определение…"
            if self.config.sensor_id_mode == "auto"
            else self.config.sensor_id_mode
        )
        self.packet_label.setText(
            f"UDP: {self.model.udp_packets} · кадров: {self.model.sample_frames} · "
            f"Q: {self.model.quaternion_count} · ID: {mode}"
        )

    def _refresh_plot(self) -> None:
        if self.needs_redraw:
            self.human_canvas.update_pose(
                self.model.orientations(), self.config.sensor_mapping,
                self.config.enabled_segments,
            )
            self.needs_redraw = False

    def _append_monitor(self, text: str) -> None:
        self.monitor.appendPlainText(text.rstrip())
        bar = self.monitor.verticalScrollBar()
        bar.setValue(bar.maximum())

    @staticmethod
    def _clock() -> str:
        return datetime.now().strftime("%H:%M:%S.%f")[:-3]

    @staticmethod
    def _format_time(timestamp: float) -> str:
        return datetime.fromtimestamp(timestamp).strftime("%H:%M:%S.%f")[:-3]

    def show_about(self) -> None:
        QMessageBox.about(
            self,
            "MPU UDP Viewer",
            "PyQt6-визуализатор пяти MPU6050.\n\n"
            "Кинематика перенесена из Blender. Оси: X — красная, "
            "Y — зеленая, Z — синяя.",
        )

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        if self.sock is not None and self.streaming_requested:
            self.send_command("STOP", warn=False)
        self.disconnect_device(log=False)
        event.accept()


def main() -> int:
    app = QApplication(sys.argv)
    app.setOrganizationName("Neuromorph")
    app.setApplicationName("MPU UDP Viewer")
    window = MotionCaptureWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
