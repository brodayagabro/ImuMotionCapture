"""Participant display shared by the current and archived workflows."""
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QProgressBar
from .config import BLOCK_LABELS
from .experiment_controller import State
from .semaphore_model import NEUTRAL
from .semaphore_renderer import BACKGROUND, INK, ACCENT, SemaphoreRenderer, WordStrip


class ParticipantWindow(QWidget):
    def __init__(self, on_hidden, mirror_mode=True):
        super().__init__()
        self.on_hidden = on_hidden
        self.setWindowTitle("Семафор · участник")
        self.setMinimumSize(480, 550)
        self.resize(820, 850)
        self.setStyleSheet(f"QWidget {{ background: {BACKGROUND}; color: {INK}; font-family: 'Segoe UI'; }}")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 24)
        self.caption = QLabel("Приготовьтесь")
        self.caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.caption.setStyleSheet("font-size: 23px;")
        self.caption.setMinimumHeight(60)
        layout.addWidget(self.caption)
        self.word = WordStrip()
        layout.addWidget(self.word)
        self.avatar = SemaphoreRenderer(mirror_mode=mirror_mode)
        layout.addWidget(self.avatar, 1)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(7)
        self.progress.setStyleSheet(f"QProgressBar {{ border: none; background: #dce1df; border-radius: 3px; }}"
                                   f"QProgressBar::chunk {{ background: {ACCENT}; border-radius: 3px; }}")
        layout.addWidget(self.progress)
        self.cue = QLabel("Повторяйте положение рук")
        self.cue.setWordWrap(True)
        self.cue.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cue.setStyleSheet("font-size: 18px; padding-top: 10px;")
        layout.addWidget(self.cue)
        QShortcut(QKeySequence("F11"), self, activated=self.toggle_fullscreen)
        QShortcut(QKeySequence("Escape"), self, activated=self.showNormal)

    def toggle_fullscreen(self):
        self.showNormal() if self.isFullScreen() else self.showFullScreen()

    def closeEvent(self, event):
        self.on_hidden()
        event.accept()

    def show_status(self, caption, cue):
        self.caption.setText(caption)
        self.cue.setText(cue)
        self.word.set_word('', None)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.avatar.set_pose(NEUTRAL.left_angle_deg, NEUTRAL.right_angle_deg)

    def show_export_progress(self, percent, phase):
        self.show_status(f'Сохранение записи · {percent}%', phase)
        self.progress.setValue(percent * 10)

    def display(self, controller):
        c = controller
        pose = c.pose()
        self.avatar.mirror_mode = c.config.mirror_mode
        self.avatar.set_pose(pose.left_angle_deg, pose.right_angle_deg)
        visible_word = c.phase in {State.WORD_PREVIEW, State.TRANSITION, State.HOLD,
                                  State.RETURN_NEUTRAL, State.POST_TRIAL}
        self.word.set_word(c.trial.word if visible_word else "", c.letter_index if visible_word else None)
        self.progress.setValue(round(c.letter_progress() * 1000))
        caption = BLOCK_LABELS[c.trial.block]
        cue = {State.PREPARE: "Приготовьтесь", State.WORD_PREVIEW: "Посмотрите на слово",
               State.TRANSITION: "Повторяйте движение", State.HOLD: "Удерживайте положение",
               State.RETURN_NEUTRAL: "Опустите руки", State.POST_TRIAL: "Отдых",
               State.INTER_TRIAL: "Отдых", State.PAUSED: "Пауза", State.FINISHED: "Спасибо за участие"}.get(c.state, "Приготовьтесь")
        if c.state == State.READY and c.trial_index > 0:
            caption, cue = "Блок завершён · отдых", "Далее: " + BLOCK_LABELS[c.trial.block]
        elif c.state == State.FINISHED:
            caption = "Завершено"
        if c.trial.training_trial and c.state != State.FINISHED:
            caption = "Тренировка · " + caption
        self.caption.setText(caption)
        self.cue.setText(cue)

