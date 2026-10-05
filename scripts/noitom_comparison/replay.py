"""Replay independent timestamped streams in the SDK-free comparison canvas."""
import heapq
import json
from pathlib import Path
import time
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget
from pyqt_mocap.mocap_core import DEFAULT_SENSOR_MAPPING
from .comparison_canvas import ComparisonOpenGLCanvas


def session_frames(path):
    """Merge by receive timestamp only; frame indices are never joined."""
    def stream(source):
        with (Path(path) / f"{source}_frames.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                yield row["relative_timestamp_ns"], source, row
    yield from heapq.merge(stream("own"), stream("noitom"), key=lambda item: item[0])


class ReplayWindow(QWidget):
    def __init__(self, path):
        super().__init__()
        self.path = Path(path)
        self.setWindowTitle("IMU + Noitom · replay · " + self.path.name)
        self.resize(1000, 800)
        layout = QVBoxLayout(self)
        self.label = QLabel(str(self.path))
        layout.addWidget(self.label)
        self.canvas = ComparisonOpenGLCanvas()
        layout.addWidget(self.canvas, 1)
        button = QPushButton("Воспроизвести сначала")
        button.clicked.connect(self.restart)
        layout.addWidget(button)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.frames = None
        self.restart()

    def restart(self):
        if self.frames is not None:
            self.frames.close()
        self.canvas.update_own_pose({}, DEFAULT_SENSOR_MAPPING)
        self.canvas.set_noitom_visible(False)
        self.frames = session_frames(self.path)
        try:
            self.next_frame = next(self.frames, None)
        except (OSError, ValueError) as error:
            self.label.setText(str(error))
            return
        self.t0 = time.monotonic_ns()
        self.timer.start(16)

    def tick(self):
        now = time.monotonic_ns() - self.t0
        try:
            for _ in range(500):
                if self.next_frame is None or self.next_frame[0] > now:
                    break
                _, source, frame = self.next_frame
                orientations = frame.get("segment_orientations", {})
                if source == "own":
                    self.canvas.update_own_pose(orientations, frame.get("sensor_mapping", DEFAULT_SENSOR_MAPPING),
                                               frame.get("enabled_segments", DEFAULT_SENSOR_MAPPING))
                elif not frame.get("mapping_error"):
                    self.canvas.update_noitom_pose(orientations, frame.get("enabled_segments", DEFAULT_SENSOR_MAPPING))
                self.next_frame = next(self.frames, None)
        except (OSError, ValueError) as error:
            self.timer.stop()
            self.label.setText(str(error))
            return
        self.label.setText(f"{self.path.name} · {now / 1e9:.2f} s")
        if self.next_frame is None:
            self.timer.stop()
            self.label.setText(self.path.name + " · завершено")

    def closeEvent(self, event):
        self.timer.stop()
        if self.frames is not None:
            self.frames.close()
        event.accept()
