# Adapted from scripts/ER2026/noitom_comparison/settings_dialog.py for visual capture.
"""Extend the existing settings window with independent Visual segment choices."""
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QCheckBox, QLabel, QTabWidget, QVBoxLayout, QWidget
from pyqt_mocap.mocap_core import DEFAULT_ENABLED_SEGMENTS, SEGMENT_NAMES
from pyqt_mocap.mpu_udp_viewer import SEGMENT_TITLES, SettingsDialog


def load_visual_segments(settings):
    return frozenset(name for name in SEGMENT_NAMES if settings.value(
        f"visual/enabled/{name}", name in DEFAULT_ENABLED_SEGMENTS, type=bool))


def save_visual_segments(settings, enabled):
    for name in SEGMENT_NAMES:
        settings.setValue(f"visual/enabled/{name}", name in enabled)
    settings.sync()


class ComparisonSettingsDialog(SettingsDialog):
    visual_segments_applied = pyqtSignal(object)

    def __init__(self, config, enabled_visual_segments, parent=None):
        super().__init__(config, parent)
        self.setWindowTitle("Настройки IMU + Visual")
        self.visual_checks = {}
        tab = QWidget()
        layout = QVBoxLayout(tab)
        heading = QLabel("Сегменты Visual для захвата движения")
        layout.addWidget(heading)
        for name in SEGMENT_NAMES:
            check = QCheckBox("Корпус (спина)" if name == "spine" else SEGMENT_TITLES[name])
            check.setChecked(name in enabled_visual_segments)
            self.visual_checks[name] = check
            layout.addWidget(check)
        note = QLabel(
            "Выбранные сегменты участвуют в отображении позы Visual. "
            "Отключённые остаются в исходной позе и показаны бледным цветом. "
            "Спина по умолчанию выключена.\n\n"
            "В датасет всегда записываются все пять измеряемых сегментов. "
            "Настройки датчиков собственной системы задаются отдельно на вкладке «Датчики».")
        note.setWordWrap(True)
        layout.addWidget(note)
        layout.addStretch()
        self.findChild(QTabWidget).addTab(tab, "Visual")

    def _apply(self):
        config = self._collect()
        if config is None:
            return False
        # A Visual-only change must not stop an ongoing own-system BVH recording.
        if config != self.config:
            self.config = config.copy()
            self.config_applied.emit(config)
        self.visual_segments_applied.emit(frozenset(
            name for name, check in self.visual_checks.items() if check.isChecked()))
        return True
