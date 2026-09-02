"""PyQt dialog for the guided N -> T -> forward calibration workflow."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import time

from PyQt6.QtCore import QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .calibration import (
    POSE_NAMES,
    CalibrationResult,
    CapturedPose,
    PoseRecorder,
    calibrate_three_poses,
)
from .mocap_core import DEFAULT_AXIS_MAPS


Snapshot = Mapping[str, tuple[object, float, int]]
SnapshotProvider = Callable[[], Snapshot]

POSE_TITLES = {
    "n_pose": "1/3 — N-поза",
    "t_pose": "2/3 — T-поза",
    "forward_pose": "3/3 — руки вперёд",
}
POSE_INSTRUCTIONS = {
    "n_pose": (
        "Стойте прямо. Руки свободно опущены, локти разогнуты, ладони к бёдрам. "
        "После нажатия не двигайтесь 5 секунд."
    ),
    "t_pose": (
        "Разведите обе прямые руки горизонтально в стороны. Корпус прямо, "
        "локти не сгибать. После нажатия не двигайтесь 5 секунд."
    ),
    "forward_pose": (
        "Вытяните обе прямые руки горизонтально вперёд. Оси Z датчиков рук "
        "должны смотреть вверх. После нажатия не двигайтесь 5 секунд."
    ),
}


class GuidedCalibrationDialog(QDialog):
    """Record three stationary pose windows and emit a calibration result."""

    result_ready = pyqtSignal(object)

    def __init__(
        self,
        snapshot_provider: SnapshotProvider,
        parent: QWidget | None = None,
        capture_duration_s: float = 5.0,
    ) -> None:
        super().__init__(parent)
        self.snapshot_provider = snapshot_provider
        self.capture_duration_s = float(capture_duration_s)
        self.stage_index = 0
        self.recorder: PoseRecorder | None = None
        self.captures: dict[str, CapturedPose] = {}
        self.capture_started_s = 0.0
        self.result: CalibrationResult | None = None

        self.setWindowTitle("Комплексная калибровка N → T → вперёд")
        self.setMinimumWidth(590)
        root = QVBoxLayout(self)
        self.title_label = QLabel()
        self.title_label.setStyleSheet("font-size: 16px; font-weight: 700;")
        root.addWidget(self.title_label)
        self.instruction_label = QLabel()
        self.instruction_label.setWordWrap(True)
        self.instruction_label.setMinimumHeight(72)
        root.addWidget(self.instruction_label)
        self.progress = QProgressBar()
        self.progress.setRange(0, round(self.capture_duration_s * 1000))
        self.progress.setValue(0)
        self.progress.setFormat("Готово к записи")
        root.addWidget(self.progress)
        self.sample_label = QLabel("Кадры ещё не записывались")
        root.addWidget(self.sample_label)

        buttons = QHBoxLayout()
        self.capture_button = QPushButton("Записать 5 секунд")
        self.capture_button.clicked.connect(self._start_capture)
        self.cancel_button = QPushButton("Отмена")
        self.cancel_button.clicked.connect(self.reject)
        buttons.addWidget(self.capture_button)
        buttons.addStretch(1)
        buttons.addWidget(self.cancel_button)
        root.addLayout(buttons)

        self.timer = QTimer(self)
        self.timer.setInterval(40)
        self.timer.timeout.connect(self._tick)
        self._show_stage()

    def _show_stage(self) -> None:
        pose_name = POSE_NAMES[self.stage_index]
        self.title_label.setText(POSE_TITLES[pose_name])
        self.instruction_label.setText(POSE_INSTRUCTIONS[pose_name])
        self.progress.setValue(0)
        self.progress.setFormat("Примите позу и нажмите кнопку")
        self.capture_button.setEnabled(True)
        self.capture_button.setText("Записать 5 секунд")

    def _start_capture(self) -> None:
        snapshot = self.snapshot_provider()
        missing = [name for name in DEFAULT_AXIS_MAPS if name not in snapshot]
        if missing:
            QMessageBox.warning(
                self,
                "Нет полного кадра",
                "Не получены свежие данные включённых датчиков: " + ", ".join(missing),
            )
            return
        self.recorder = PoseRecorder()
        self.capture_started_s = time.monotonic()
        self.capture_button.setEnabled(False)
        self.progress.setValue(0)
        self.progress.setFormat("Не двигайтесь: %v / %m мс")
        self.timer.start()

    def _tick(self) -> None:
        if self.recorder is None:
            return
        self.recorder.add_snapshot(self.snapshot_provider())
        elapsed_s = time.monotonic() - self.capture_started_s
        self.progress.setValue(
            min(self.progress.maximum(), round(elapsed_s * 1000))
        )
        counts = {
            name: len(values) for name, values in self.recorder.values.items()
        }
        self.sample_label.setText(
            "Кадры: " + " · ".join(f"{name}={counts[name]}" for name in counts)
        )
        if elapsed_s < self.capture_duration_s:
            return
        self.timer.stop()
        pose_name = POSE_NAMES[self.stage_index]
        try:
            capture = self.recorder.finish(pose_name)
        except ValueError as error:
            self.recorder = None
            self.capture_button.setEnabled(True)
            self.progress.setFormat("Запись не принята")
            QMessageBox.warning(self, "Повторите этап", str(error))
            return
        self.captures[pose_name] = capture
        self.recorder = None
        self.stage_index += 1
        if self.stage_index < len(POSE_NAMES):
            self._show_stage()
            return
        self._finish_calibration()

    def _finish_calibration(self) -> None:
        try:
            self.result = calibrate_three_poses(self.captures, DEFAULT_AXIS_MAPS)
        except ValueError as error:
            QMessageBox.critical(self, "Ошибка калибровки", str(error))
            self.reject()
            return
        self.title_label.setText("Калибровка завершена")
        score = max(self.result.scores_deg.values())
        alignment = self.result.max_axis_alignment_deg
        drift = self.result.max_drift_deg_s
        self.instruction_label.setText(
            f"Максимальная ошибка направления: {score:.1f}°. "
            f"Поправка согласования осей: до {alignment:.1f}°. "
            f"Оценка остаточного дрейфа: {drift:.3f}°/с. "
            "Профиль применён; сохраните его через меню «Файл»."
        )
        self.progress.setValue(self.progress.maximum())
        self.progress.setFormat("Готово")
        self.capture_button.setText("Закрыть")
        self.capture_button.setEnabled(True)
        try:
            self.capture_button.clicked.disconnect()
        except TypeError:
            pass
        self.capture_button.clicked.connect(self.accept)
        self.cancel_button.setVisible(False)
        self.result_ready.emit(self.result)

    def reject(self) -> None:
        self.timer.stop()
        super().reject()
