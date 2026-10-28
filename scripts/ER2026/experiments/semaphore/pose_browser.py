"""Manual verification of every stimulus before running the protocol."""
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QCheckBox, QComboBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget
from .semaphore_model import SEMAPHORE_POSES, letter_group
from .semaphore_renderer import SemaphoreRenderer


class PoseBrowser(QWidget):
    pose_selected = pyqtSignal(object, bool, str)

    def __init__(self, mirror_mode=True, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Проверьте буквы, стороны рук и отражение до эксперимента."))
        bar = QHBoxLayout()
        previous, following = QPushButton("← Предыдущая"), QPushButton("Следующая →")
        self.letters = QComboBox()
        self.letters.addItems(sorted(SEMAPHORE_POSES))
        for widget in (previous, self.letters, following):
            bar.addWidget(widget)
        layout.addLayout(bar)
        self.mirror = QCheckBox("mirror_mode=true: левая рука участника справа на изображении")
        self.mirror.setChecked(mirror_mode)
        layout.addWidget(self.mirror)
        self.renderer = SemaphoreRenderer(mirror_mode=mirror_mode)
        self.renderer.show_arm_labels = True
        layout.addWidget(self.renderer, 1)
        self.description = QLabel()
        self.description.setWordWrap(True)
        layout.addWidget(self.description)
        previous.clicked.connect(lambda: self.step(-1))
        following.clicked.connect(lambda: self.step(1))
        self.letters.currentTextChanged.connect(self.update_pose)
        self.mirror.toggled.connect(self.update_pose)
        self.update_pose()

    def step(self, delta):
        self.letters.setCurrentIndex((self.letters.currentIndex() + delta) % self.letters.count())

    def update_pose(self, *_):
        letter = self.letters.currentText()
        pose = SEMAPHORE_POSES[letter]
        self.renderer.mirror_mode = self.mirror.isChecked()
        self.renderer.set_pose(pose.left_angle_deg, pose.right_angle_deg)
        self.description.setText(f"{letter} · {letter_group(letter)} · левая {pose.left_angle_deg:g}° · правая {pose.right_angle_deg:g}°\n"
            "Углы со стороны участника: 0° вправо, 90° вверх, 180° влево, 270° вниз.\n"
            "Переключатель здесь проверяет отражение; протокол эксперимента он не меняет.")
        self.pose_selected.emit(pose, self.mirror.isChecked(), letter)
