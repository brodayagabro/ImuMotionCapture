"""Extend the existing settings window with independent Noitom segment choices."""
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QCheckBox, QLabel, QTabWidget, QVBoxLayout, QWidget
from pyqt_mocap.mocap_core import DEFAULT_ENABLED_SEGMENTS, SEGMENT_NAMES
from pyqt_mocap.mpu_udp_viewer import SEGMENT_TITLES, SettingsDialog


def load_noitom_segments(settings):
    return frozenset(name for name in SEGMENT_NAMES if settings.value(
        f"noitom/enabled/{name}", name in DEFAULT_ENABLED_SEGMENTS, type=bool))


def save_noitom_segments(settings, enabled):
    for name in SEGMENT_NAMES:
        settings.setValue(f"noitom/enabled/{name}", name in enabled)
    settings.sync()


class ComparisonSettingsDialog(SettingsDialog):
    noitom_segments_applied = pyqtSignal(object)

    def __init__(self, config, enabled_noitom_segments, parent=None):
        super().__init__(config, parent)
        self.setWindowTitle("Настройки IMU + Noitom")
        self.noitom_checks = {}
        tab = QWidget()
        layout = QVBoxLayout(tab)
        heading = QLabel("Сегменты Noitom для захвата движения")
        layout.addWidget(heading)
        for name in SEGMENT_NAMES:
            check = QCheckBox("Корпус (спина)" if name == "spine" else SEGMENT_TITLES[name])
            check.setChecked(name in enabled_noitom_segments)
            self.noitom_checks[name] = check
            layout.addWidget(check)
        note = QLabel(
            "Выбранные сегменты участвуют в отображении и записи позы Noitom. "
            "Отключённые остаются в исходной позе и показаны бледным цветом. "
            "Спина по умолчанию выключена.\n\n"
            "Исходные данные костюма сохраняются полностью. "
            "Настройки датчиков собственной системы задаются отдельно на вкладке «Датчики».")
        note.setWordWrap(True)
        layout.addWidget(note)
        layout.addStretch()
        self.findChild(QTabWidget).addTab(tab, "Noitom")

    def _apply(self):
        config = self._collect()
        if config is None:
            return False
        # A Noitom-only change must not stop an ongoing own-system BVH recording.
        if config != self.config:
            self.config = config.copy()
            self.config_applied.emit(config)
        self.noitom_segments_applied.emit(frozenset(
            name for name, check in self.noitom_checks.items() if check.isChecked()))
        return True
