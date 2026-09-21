"""Vector pose examples with an animated transition and a static target."""

import math

from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import QWidget
from .semaphore_calibration import base_pose, target_direction


class PosePreview(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.previous = "a_pose"
        self.target = "a_pose"
        self.fraction = 1.
        self.setMinimumHeight(235)
        self.setAccessibleName("Пример перехода между позами калибровки")

    def set_transition(self, previous, target, fraction):
        self.previous, self.target = previous, target
        self.fraction = min(1., max(0., fraction))
        self.update()

    @staticmethod
    def direction(pose, side, forearm=False):
        if pose.startswith("sem_"):
            segment = ("forearm" if forearm else "shoulder") + (".L" if side < 0 else ".R")
            target = target_direction(pose, segment)
            if target is None:
                # Illustrative 20-degree elbow flexion, NEVER a solver target.
                return (-math.cos(math.radians(25)), 0., -math.sin(math.radians(25)))
            return tuple(target)
        if pose == "p_pose" and forearm:
            return (-side * math.sqrt(.5), 0., math.sqrt(.5))
        if pose == "t_pose":
            return (side, 0., 0.)
        if pose == "forward_pose":
            return (0., 1., 0.)
        return (0., 0., -1.)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#f2f5fa"))
        scale = min(self.width() / 5.5, (self.height() - 42) / 1.9)

        def draw_person(center, fraction, side_view=False):
            def project(point):
                x, y, z = point
                if side_view:
                    return QPointF(center + scale * (.12 * x - .9 * y),
                                   self.height() - 15 - scale * z)
                return QPointF(center + scale * (x - .65 * y),
                               self.height() - 15 - scale * (z + .18 * y))

            def line(a, b, color="#516074", width=4):
                painter.setPen(QPen(QColor(color), width, Qt.PenStyle.SolidLine,
                                    Qt.PenCapStyle.RoundCap))
                painter.drawLine(project(a), project(b))

            line((0, 0, .8), (0, 0, 1.4))
            line((-.27, 0, 1.3), (.27, 0, 1.3))
            for side in (-1, 1):
                line((0, 0, .8), (side * .13, 0, .4))
                line((side * .13, 0, .4), (side * .15, 0, .02))
                def direction(forearm):
                    start = self.direction(self.previous, side, forearm)
                    end = self.direction(self.target, side, forearm)
                    if abs(start[1]) < 1e-9 and abs(end[1]) < 1e-9:
                        start_angle = math.atan2(start[0], -start[2])
                        end_angle = math.atan2(end[0], -end[2])
                        delta = (end_angle - start_angle + math.pi) % (2 * math.pi) - math.pi
                        angle = start_angle + fraction * delta
                        return (math.sin(angle), 0., -math.cos(angle))
                    values = [(1 - fraction) * a + fraction * b for a, b in zip(start, end)]
                    norm = math.sqrt(sum(v * v for v in values))
                    return tuple(v / norm for v in values)

                upper, lower = direction(False), direction(True)
                shoulder = (side * .27, 0, 1.3)
                elbow = tuple(a + .35 * b for a, b in zip(shoulder, upper))
                wrist = tuple(a + .35 * b for a, b in zip(elbow, lower))
                line(shoulder, elbow, "#176bb0")
                line(elbow, wrist, "#176bb0")
                painter.setPen(QPen(QColor("#176bb0"), 5))
                painter.drawPoint(project(wrist))
            painter.setPen(QPen(QColor("#516074"), 3))
            painter.drawEllipse(project((0, 0, 1.59)), scale * .12, scale * .12)

        draw_person(self.width() * .28, self.fraction)
        side_view = self.target == "forward_pose"
        draw_person(self.width() * .76, 1., side_view)
        painter.setPen(QColor("#24344a"))
        painter.drawText(12, 22, "Пример перехода")
        painter.drawText(int(self.width() * .57), 22,
                         "Поза для записи · вид сбоку" if side_view else "Поза для записи")
        painter.drawText(QPointF(self.width() * .49, self.height() * .5), "→")
        if base_pose(self.target) == "sem_d":
            painter.drawText(12, 42, "Сгиб правого локтя условный: угол не калибруется")
