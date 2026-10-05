"""Scalable QPainter avatar; knows only two arm angles and a mirror transform."""
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import QWidget

from .semaphore_model import NEUTRAL, SemaphorePose, arm_geometry

BACKGROUND = "#f3f4f1"
INK = "#38434c"
ACCENT = "#287f87"


class SemaphoreRenderer(QWidget):
    def __init__(self, parent=None, *, mirror_mode=True):
        super().__init__(parent)
        self.pose = NEUTRAL
        self.mirror_mode = mirror_mode
        self.show_arm_labels = False
        self.setMinimumSize(240, 280)
        self.setAccessibleName("Аватар: положение рук")

    def set_pose(self, left_angle, right_angle):
        self.pose = SemaphorePose(left_angle, right_angle)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(BACKGROUND))
        scale = min(self.width() / 2.65, self.height() / 2.55)
        def point(xy):
            return QPointF(self.width() / 2 + xy[0] * scale, self.height() / 2 - xy[1] * scale)
        def line(a, b, color, width):
            p.setPen(QPen(QColor(color), width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            p.drawLine(point(a), point(b))
        width = max(4., scale * .038)
        line((0, .4), (0, -.4), INK, width)
        line((-.23, .25), (.23, .25), INK, width)
        line((0, -.4), (-.18, -.97), INK, width)
        line((0, -.4), (.18, -.97), INK, width)
        p.setPen(QPen(QColor(INK), width))
        p.setBrush(QColor(BACKGROUND))
        p.drawEllipse(point((0, .65)), scale * .15, scale * .15)
        for name, (start, end) in arm_geometry(self.pose, self.mirror_mode).items():
            line(start, end, ACCENT, width * 1.5)
            p.setBrush(QColor(ACCENT))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(point(end), width * .9, width * .9)
            if self.show_arm_labels:
                p.setPen(QColor(INK))
                font = QFont("Segoe UI", max(10, round(scale * .075)))
                p.setFont(font)
                p.drawText(point((start[0], start[1] - .15)), "Л" if name == "left" else "П")


class WordStrip(QWidget):
    """Fixed letter cells: highlighting never changes text layout."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.word, self.index = "", None
        self.setMinimumHeight(90)

    def set_word(self, word, index=None):
        if (word, index) != (self.word, self.index):
            self.word, self.index = word, index
            self.setAccessibleName("Слово " + word + (f", буква {index + 1}" if index is not None else ""))
            self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(BACKGROUND))
        if not self.word:
            return
        cell = min(76., (self.width() - 32) / len(self.word), self.height() - 8.)
        left = (self.width() - cell * len(self.word)) / 2
        font = QFont("Segoe UI")
        font.setPixelSize(round(cell * .65))
        font.setWeight(QFont.Weight.DemiBold)
        p.setFont(font)
        for i, letter in enumerate(self.word):
            rect = QRectF(left + i * cell + 3, 4, cell - 6, self.height() - 8)
            if i == self.index:
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor(ACCENT))
                p.drawRoundedRect(rect, 12, 12)
            p.setPen(QColor("#ffffff" if i == self.index else INK))
            p.drawText(rect, Qt.AlignmentFlag.AlignCenter, letter)
