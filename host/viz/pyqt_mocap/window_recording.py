"""BVH recording controls and the viewer skeleton adapter."""

from datetime import datetime
from pathlib import Path

import numpy as np
from PyQt6.QtWidgets import QFileDialog, QMessageBox

from storage.bvh import BvhRecording, Joint
from .mocap_core import compute_body_pose, quaternion_to_matrix

BVH_HEIGHT_M = 1.70
HEAD_TOP_OFFSET_M = .10
HAND_LENGTH_M = .18


def export_geometry():
    """Uniform scale and floor origin; height includes the head End Site."""
    joints = np.asarray(compute_body_pose({}).joints)
    floor = min(joints[13, 2], joints[14, 2])
    height = joints[2, 2] + HEAD_TOP_OFFSET_M - floor
    return BVH_HEIGHT_M / height, np.array((0., 0., floor))


def create_recording(fps):
    pose = compute_body_pose({})
    scale, origin = export_geometry()
    p = (np.asarray(pose.joints) - origin) * scale
    # Extra Spine node at pelvis keeps the untracked legs independent of torso tilt.
    specs = [
        ("Hips", None, p[0], None),
        ("Spine", 0, p[0], "spine"),
        ("Chest", 1, p[1], None),
        ("Head", 2, p[2], None),
        ("LeftArm", 1, p[3], "shoulder.L"),
        ("LeftForeArm", 4, p[5], "forearm.L"),
        ("LeftHand", 5, p[7], None),
        ("RightArm", 1, p[4], "shoulder.R"),
        ("RightForeArm", 7, p[6], "forearm.R"),
        ("RightHand", 8, p[8], None),
        ("LeftUpLeg", 0, p[9], None),
        ("LeftLeg", 10, p[11], None),
        ("LeftFoot", 11, p[13], None),
        ("RightUpLeg", 0, p[10], None),
        ("RightLeg", 13, p[12], None),
        ("RightFoot", 14, p[14], None),
    ]
    joints = []
    for name, parent, position, source in specs:
        offset = np.zeros(3) if parent is None else position - specs[parent][2]
        end = (0., .13, 0.) if name.endswith("Foot") else (0., 0., 0.)
        if name == "Head":
            end = (0., 0., HEAD_TOP_OFFSET_M)
        end = tuple(np.asarray(end) * scale)
        if name.endswith("Hand"):
            # No wrist IMU: the hand follows the forearm. A zero End Site
            # makes Blender invent a hand as long as the parent forearm.
            end = (0., 0., -HAND_LENGTH_M)
        joints.append(Joint(name, parent, tuple(offset), source, end))
    return BvhRecording(joints, p[0], fps, units_per_meter=1.)


class RecordingWindowMixin:
    def toggle_bvh_recording(self):
        if self.bvh_active:
            self.stop_bvh_recording()
        else:
            self.start_bvh_recording()

    def start_bvh_recording(self):
        if self.sock is None or self.model.neutral_pending:
            QMessageBox.information(self, "BVH", "Подключитесь к датчикам и примите A-позу.")
            return
        if not self._confirm_bvh_discard():
            return
        self.bvh_recording = create_recording(self.config.stream_rate_hz)
        self.bvh_active = True
        self.record_bvh_button.setText("Остановить запись")
        self.bvh_dirty = False
        self.record_bvh_action.setEnabled(False)
        self.stop_bvh_action.setEnabled(True)
        self.save_bvh_action.setEnabled(False)
        self.bvh_label.setText("BVH: запись, ожидание кадров")

    def capture_bvh_frame(self, timestamp):
        if self.bvh_active:
            rotations = {name: quaternion_to_matrix(q)
                         for name, q in self.model.orientations().items()}
            self.bvh_recording.append(timestamp, rotations)
            self.bvh_dirty = True
            self.bvh_label.setText(f"BVH: запись · {len(self.bvh_recording.frames)} отсчётов")

    def stop_bvh_recording(self):
        self.bvh_active = False
        self.record_bvh_button.setText("Начать запись")
        self.record_bvh_action.setEnabled(True)
        self.stop_bvh_action.setEnabled(False)
        count = len(self.bvh_recording.frames) if self.bvh_recording else 0
        self.save_bvh_action.setEnabled(count > 0)
        self.bvh_label.setText(f"BVH: остановлено · {count} отсчётов")

    def save_bvh_recording(self):
        if self.bvh_active:
            self.stop_bvh_recording()
        if not self.bvh_recording or not self.bvh_recording.frames:
            return False
        filename, _ = QFileDialog.getSaveFileName(
            self, "Экспорт BVH", datetime.now().strftime("capture_%Y%m%d_%H%M%S.bvh"),
            "Biovision Hierarchy (*.bvh)")
        if not filename:
            return False
        path = Path(filename)
        if path.suffix.lower() != ".bvh":
            path = path.with_suffix(".bvh")
            if path.exists() and QMessageBox.question(
                self, "BVH", f"Перезаписать {path}?"
            ) != QMessageBox.StandardButton.Yes:
                return False
        try:
            count = self.bvh_recording.save(path)
        except (OSError, ValueError) as error:
            QMessageBox.critical(self, "Ошибка экспорта BVH", str(error))
            return False
        self.bvh_dirty = False
        self.bvh_label.setText(f"BVH: сохранено {count} кадров")
        self.statusBar().showMessage(f"Сохранено: {path}", 10000)
        return True

    def _confirm_bvh_discard(self):
        if not self.bvh_dirty:
            return True
        answer = QMessageBox.question(
            self, "Запись BVH", "Сохранить записанное движение?",
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard |
            QMessageBox.StandardButton.Cancel, QMessageBox.StandardButton.Save)
        if answer == QMessageBox.StandardButton.Save:
            return self.save_bvh_recording()
        return answer == QMessageBox.StandardButton.Discard
