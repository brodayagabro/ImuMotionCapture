"""PyQt dialog for the guided A -> T -> forward -> P -> A workflow."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import time
import math

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
    calibrate_five_poses,
)
from .mocap_core import DEFAULT_AXIS_MAPS
from .pose_preview import PosePreview
from .semaphore_calibration import (
    SEMAPHORE_POSES, LABELS, INSTRUCTIONS, base_pose, calibrate_semaphore,
)


Snapshot = Mapping[str, tuple[object, float, int]]
SnapshotProvider = Callable[[], Snapshot]

POSE_TITLES = {
    "a_pose": "1/5 — A-поза: руки вниз",
    "t_pose": "2/5 — T-поза",
    "forward_pose": "3/5 — руки вперёд",
    "p_pose": "4/5 — ладони вместе перед грудиной",
    "return_a_pose": "5/5 — A-поза: снова руки вниз",
}
POSE_INSTRUCTIONS = {
    "p_pose": (
        "Прижмите верхние части рук и локти к корпусу. Соедините ладони перед "
        "грудиной, предплечья направьте вверх и к центру. Внутренний угол между "
        "плечом и предплечьем — около 45°. Не поднимайте плечи. Во время записи "
        "стойте неподвижно."
    ),
    "a_pose": (
        "Стойте прямо. Руки свободно опущены, локти разогнуты, ладони к бёдрам. "
        "Во время записи стойте неподвижно."
    ),
    "t_pose": (
        "Разведите обе прямые руки горизонтально в стороны. Корпус прямо, "
        "локти не сгибать. Во время записи стойте неподвижно."
    ),
    "forward_pose": (
        "Вытяните обе прямые руки горизонтально вперёд. Оси Z датчиков рук "
        "должны смотреть вверх. Во время записи стойте неподвижно."
    ),
    "return_a_pose": (
        "Вернитесь в исходную A-позу: руки свободно вниз, локти разогнуты, "
        "ладони к бёдрам, корпус прямо. Во время записи стойте неподвижно."
    ),
}


class GuidedCalibrationDialog(QDialog):
    """Record five stationary pose windows and emit a calibration result."""

    result_ready = pyqtSignal(object)

    def __init__(
        self,
        snapshot_provider: SnapshotProvider,
        parent: QWidget | None = None,
        capture_duration_s: float = 5.0,
        preparation_duration_s: float = 5.0,
        semaphore: bool = False,
        preferred_axis_maps=None,
        prior_alignment=None,
        enabled_segments=None,
    ) -> None:
        super().__init__(parent)
        self.semaphore = semaphore
        self.pose_names = SEMAPHORE_POSES if semaphore else POSE_NAMES
        self.preferred_axis_maps = preferred_axis_maps or DEFAULT_AXIS_MAPS
        self.prior_alignment = prior_alignment
        self.enabled_segments = enabled_segments
        self.snapshot_provider = snapshot_provider
        self.capture_duration_s = float(capture_duration_s)
        self.preparation_duration_s = float(preparation_duration_s)
        if not all(math.isfinite(v) and v > 0 for v in
                   (self.capture_duration_s, self.preparation_duration_s)):
            raise ValueError("Durations must be positive and finite")
        self.phase = "idle"
        self.phase_started_s = 0.
        self.stage_index = 0
        self.recorder: PoseRecorder | None = None
        self.captures: dict[str, CapturedPose] = {}
        self.capture_started_s = 0.0
        self.result: CalibrationResult | None = None

        self.setWindowTitle("Калибровка A → T → вперёд → P → A")
        if semaphore:
            self.setWindowTitle("Семафорная калибровка XZ — один круг")
        self.setMinimumWidth(590)
        root = QVBoxLayout(self)
        self.title_label = QLabel()
        self.title_label.setStyleSheet("font-size: 16px; font-weight: 700;")
        root.addWidget(self.title_label)
        self.instruction_label = QLabel()
        self.instruction_label.setWordWrap(True)
        self.instruction_label.setMinimumHeight(72)
        root.addWidget(self.instruction_label)
        self.preview = PosePreview(self)
        root.addWidget(self.preview)
        self.phase_label = QLabel("A → T → руки вперёд → P → A. Один запуск для всех этапов.")
        if semaphore:
            self.phase_label.setText("A-поза → А → У → Ж → Е → Д → A-поза")
        self.phase_label.setStyleSheet("font-weight: 700; font-size: 15px;")
        root.addWidget(self.phase_label)
        self.progress = QProgressBar()
        self.progress.setRange(0, round(self.capture_duration_s * 1000))
        self.progress.setValue(0)
        self.progress.setFormat("Готово к записи")
        root.addWidget(self.progress)
        self.sample_label = QLabel("Кадры ещё не записывались")
        root.addWidget(self.sample_label)

        buttons = QHBoxLayout()
        self.capture_button = QPushButton("Начать калибровку")
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
        pose_name = self.pose_names[self.stage_index]
        if self.semaphore:
            base = base_pose(pose_name)
            label = (f"Буква {LABELS[base]}"
                     if base in LABELS else "A-поза: руки вниз")
            self.title_label.setText(f"{self.stage_index + 1}/{len(self.pose_names)} — {label}")
            self.instruction_label.setText(
                INSTRUCTIONS.get(base, POSE_INSTRUCTIONS.get(pose_name, ""))
                + " Корпус неподвижен. Левая рука на примере слева, правая справа."
            )
        else:
            self.title_label.setText(POSE_TITLES[pose_name])
            self.instruction_label.setText(POSE_INSTRUCTIONS[pose_name])
        self.progress.setValue(0)
        self.progress.setFormat("После запуска этапы записываются автоматически")
        self.capture_button.setEnabled(True)
        self.capture_button.setText("Начать калибровку")
        previous = self.pose_names[max(0, self.stage_index - 1)]
        self.preview.set_transition(previous, pose_name, 0.)

    def _start_capture(self) -> None:
        self._show_stage()
        self.phase = "preparing"
        self.phase_started_s = time.monotonic()
        self.capture_button.setEnabled(False)
        self.progress.setRange(0, round(self.preparation_duration_s * 1000))
        self.progress.setFormat("Подготовка: %v / %m мс")
        self.phase_label.setText(f"Примите позу · запись через {self.preparation_duration_s:g} с")
        self.sample_label.setText("Переход: данные пока не записываются")
        self.timer.start()

    def _begin_recording(self) -> None:
        snapshot = self.snapshot_provider()
        missing = [name for name in DEFAULT_AXIS_MAPS if name not in snapshot]
        if missing:
            self.timer.stop()
            self.phase = "idle"
            self.capture_button.setEnabled(True)
            self.capture_button.setText("Повторить этап")
            self.phase_label.setText("Ожидание данных датчиков")
            QMessageBox.warning(
                self,
                "Нет полного кадра",
                "Не получены свежие данные включённых датчиков: " + ", ".join(missing),
            )
            return
        self.recorder = PoseRecorder()
        # The snapshot at the boundary may still belong to the movement phase.
        self.recorder.last_generation = {name: value[2] for name, value in snapshot.items()}
        self.capture_started_s = time.monotonic()
        self.phase = "recording"
        self.preview.set_transition(self.pose_names[self.stage_index], self.pose_names[self.stage_index], 1.)
        self.phase_label.setText("Запись — не двигайтесь")
        self.capture_button.setEnabled(False)
        self.progress.setValue(0)
        self.progress.setRange(0, round(self.capture_duration_s * 1000))
        self.progress.setFormat("Не двигайтесь: %v / %m мс")
        self.timer.start()

    def _tick(self) -> None:
        if self.phase == "preparing":
            elapsed = time.monotonic() - self.phase_started_s
            self.progress.setValue(min(self.progress.maximum(), round(elapsed * 1000)))
            self.phase_label.setText(f"Примите позу · запись через {max(0, math.ceil(self.preparation_duration_s - elapsed))} с")
            self.preview.set_transition(
                self.pose_names[max(0, self.stage_index - 1)], self.pose_names[self.stage_index],
                elapsed / self.preparation_duration_s)
            if elapsed >= self.preparation_duration_s:
                self._begin_recording()
            return
        if self.recorder is None:
            return
        snapshot = {name: value for name, value in self.snapshot_provider().items()
                    if value[1] >= self.capture_started_s}
        self.recorder.add_snapshot(snapshot)
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
        pose_name = self.pose_names[self.stage_index]
        try:
            capture = self.recorder.finish(pose_name)
        except ValueError as error:
            self.phase = "idle"
            self.recorder = None
            self.capture_button.setEnabled(True)
            self.capture_button.setText("Повторить этап")
            self.phase_label.setText("Этап не принят — повторите после проверки датчиков")
            self.progress.setFormat("Запись не принята")
            QMessageBox.warning(self, "Повторите этап", str(error))
            return
        self.captures[pose_name] = capture
        self.recorder = None
        self.stage_index += 1
        if self.stage_index < len(self.pose_names):
            self._start_capture()
            return
        self.phase = "finished"
        self._finish_calibration()

    def _finish_calibration(self) -> None:
        try:
            if self.semaphore:
                self.result = calibrate_semaphore(
                    self.captures, self.preferred_axis_maps,
                    self.prior_alignment, self.enabled_segments,
                )
            else:
                self.result = calibrate_five_poses(self.captures, self.preferred_axis_maps)
        except ValueError as error:
            QMessageBox.critical(self, "Ошибка калибровки", str(error))
            self.reject()
            return
        self.title_label.setText("Калибровка завершена")
        self.phase_label.setText(f"Все этапы записаны: {len(self.pose_names)}")
        score = max(self.result.scores_deg.values())
        alignment = self.result.max_axis_alignment_deg
        drift = self.result.max_drift_deg_s
        self.instruction_label.setText(
            f"Максимальная ошибка направления: {score:.1f}°. "
            f"Поправка согласования осей: до {alignment:.1f}°. "
            f"Оценка остаточного дрейфа: {drift:.3f}°/с. "
            "Профиль применён; сохраните его через меню «Файл»."
        )
        if self.semaphore:
            closure = max(self.result.closure_error_deg.values())
            self.instruction_label.setText(self.instruction_label.text() +
                f" Ошибка возврата в A-позу: до {closure:.1f}°. "
                "Калибруется плоскость XZ; полное согласование в 3D по одному кругу не проверяется.")
            if closure > 15.:
                self.instruction_label.setText(self.instruction_label.text() +
                    " Возврат отличается более чем на 15° — рекомендуется повторить калибровку.")
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
