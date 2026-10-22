"""Noitom pose annotation UI; no SDK, sockets, hardware or OpenGL required."""
from copy import deepcopy
from pathlib import Path
import sys
import time

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QFontDatabase, QKeySequence, QShortcut
from PyQt6.QtWidgets import (QApplication, QComboBox, QDoubleSpinBox, QHBoxLayout,
    QHeaderView, QLabel, QLineEdit, QMainWindow, QMessageBox, QPushButton, QSlider,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from .annotations import (LABELS, angular_speed, candidates, interval, pose_hint,
                          read_document, save_document, validate_intervals)
from .io import indexed_records
from .motion import load_bvh, require_pair


class Annotator(QMainWindow):
    def __init__(self, records, labels):
        super().__init__()
        # Qt's offscreen Windows plugin does not discover system fonts.
        if not QFontDatabase.families():
            font = Path('C:/Windows/Fonts/segoeui.ttf')
            if font.exists() and QFontDatabase.addApplicationFont(str(font)) >= 0:
                QApplication.instance().setFont(QFont('Segoe UI', 10))
        self.root, self.labels_root = Path(records).resolve(), Path(labels).resolve()
        if self.labels_root == self.root or self.labels_root.is_relative_to(self.root):
            raise ValueError('Labels must be outside source records')
        self.records = [row for row in indexed_records(self.root) if 'noitom' in row['paths']]
        if not self.records:
            raise ValueError('No indexed Noitom recordings')
        self.document, self.motion, self.motions = None, None, {}
        self.selected, self.undo_stack = None, []
        self.setWindowTitle('ER2026 — разметка семафора по Noitom')
        self.resize(1320, 920)
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        top = QHBoxLayout()
        layout.addLayout(top)
        self.record = QComboBox()
        self.record.setMinimumContentsLength(30)
        self.record.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.record.addItems([f"{r['subject_id']} · {r['word_set_id']} · {r['recording_id']}" for r in self.records])
        top.addWidget(self.record, 1)
        top.addWidget(QLabel('Источник разметки: Noitom'))
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        plots = QHBoxLayout()
        layout.addLayout(plots, 2)
        self.views, self.lines = [], {}
        for title in ('Спереди · левая рука участника справа', 'Сбоку · глубина / высота'):
            view = pg.PlotWidget(title=title)
            view.setAspectLocked()
            view.setXRange(-1.5, 1.5)
            view.setYRange(-.9, 1.5)
            view.showGrid(x=True, y=True, alpha=.15)
            plots.addWidget(view)
            self.views.append(view)
        self.lines['noitom'] = [v.plot(pen=pg.mkPen('#51c7fa', width=3), symbol='o', symbolSize=5)
                                for v in self.views]
        self.hint = QLabel('Подсказка — только гипотеза; подтверждайте по позе. Е и Э совпадают в словаре.')
        self.hint.setWordWrap(True)
        layout.addWidget(self.hint)
        self.timeline = pg.PlotWidget(title='Максимальная угловая скорость четырёх сегментов рук, °/с')
        self.timeline.setLabel('bottom', 'Время от начала BVH', units='s')
        layout.addWidget(self.timeline, 1)
        self.region = pg.LinearRegionItem((0., 1.))
        self.timeline.addItem(self.region)
        self.cursor = pg.InfiniteLine(angle=90, movable=False, pen='#ffffff')
        self.timeline.addItem(self.cursor)
        self.speed_curve = self.timeline.plot(pen='#51c7fa')
        controls = QHBoxLayout()
        layout.addLayout(controls)
        self.play = QPushButton('Пуск / Пауза [Space]')
        controls.addWidget(self.play)
        self.rate = QComboBox()
        self.rate.addItems(['0.25', '0.5', '1', '2'])
        self.rate.setCurrentIndex(2)
        controls.addWidget(self.rate)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        controls.addWidget(self.slider, 1)
        self.time_label = QLabel()
        controls.addWidget(self.time_label)
        edit = QHBoxLayout()
        layout.addLayout(edit)
        self.start, self.end = QDoubleSpinBox(), QDoubleSpinBox()
        for label, spin in (('Начало, с', self.start), ('Конец, с', self.end)):
            edit.addWidget(QLabel(label))
            spin.setDecimals(3)
            spin.setSingleStep(.02)
            edit.addWidget(spin)
        self.label = QComboBox()
        self.label.addItems(LABELS)
        edit.addWidget(self.label)
        self.note = QLineEdit()
        self.note.setPlaceholderText('Комментарий / слово / сомнения')
        edit.addWidget(self.note, 1)
        actions = QHBoxLayout()
        layout.addLayout(actions)
        self.new_button = QPushButton('Новый интервал')
        self.confirm = QPushButton('Подтвердить и сохранить [Ctrl+S]')
        self.next_button = QPushButton('Следующее устойчивое положение [N]')
        self.delete = QPushButton('Удалить выбранный')
        self.undo = QPushButton('Отменить [Ctrl+Z]')
        for button in (self.new_button, self.confirm, self.next_button, self.delete, self.undo):
            actions.addWidget(button)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(['Начало, с', 'Конец, с', 'Метка', 'Проверено', 'Комментарий'])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.table, 1)
        help_label = QLabel('Границы: [начало, конец). Неразмеченное время не участвует в обучении. '
                            'TRANSITION / BAD / UNKNOWN исключаются; NEUTRAL — отдельный класс. '
                            'Обе проекции показывают Noitom: спереди и сбоку.')
        help_label.setWordWrap(True)
        layout.addWidget(help_label)
        self.timer = QTimer(self)
        self.timer.setInterval(30)
        self.timer.timeout.connect(self.tick)
        self.record.currentIndexChanged.connect(self.load_record)
        self.slider.valueChanged.connect(self.refresh_pose)
        self.region.sigRegionChanged.connect(self.region_changed)
        self.start.valueChanged.connect(self.spin_changed)
        self.end.valueChanged.connect(self.spin_changed)
        self.table.cellClicked.connect(self.select_interval)
        self.play.clicked.connect(self.toggle_play)
        self.confirm.clicked.connect(self.commit)
        self.new_button.clicked.connect(self.new_interval)
        self.next_button.clicked.connect(self.next_candidate)
        self.delete.clicked.connect(self.delete_interval)
        self.undo.clicked.connect(self.undo_change)
        self.shortcuts = []
        for key, callback in (('Space', self.toggle_play), ('Ctrl+S', self.commit),
                              ('Ctrl+Z', self.undo_change), ('N', self.next_candidate)):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(callback)
            self.shortcuts.append(shortcut)
        self.timeline.scene().sigMouseClicked.connect(self.seek)
        self.load_record(0)
        if self.motion is None and len(self.records) > 1:
            self.record.setCurrentIndex(1)

    def load_record(self, index):
        self.timer.stop()
        row = self.records[index]
        try:
            motions = {source: load_bvh(path) for source, path in row['paths'].items()}
            if len(motions) == 2:
                require_pair(motions['custom'], motions['noitom'])
            path = self.labels_root / (row['recording_id'] + '.json')
            document = read_document(path, row, motions)
        except (OSError, ValueError, IndexError) as error:
            self.motion, self.document = None, None
            self.status.setText(f'Запись недоступна: {error}. Выберите другую запись.')
            self.table.setRowCount(0)
            for curves in self.lines.values():
                for curve in curves:
                    curve.setData([], [])
            return
        self.motions, self.document, self.path = motions, document, path
        # Keep paired source hashes in the document, but use only Noitom for review.
        self.body = {'noitom': motions['noitom'].body_positions()}
        self.body_rot = {'noitom': motions['noitom'].body_rotations()}
        self.directions = motions['noitom'].directions()
        self.undo_stack = []
        self.selected = None
        self.rebuild_table()
        self.prepare_motion()
        self.status.setText(f"{row['recording_id']} · метки: {path} · автосохранение после подтверждения")

    def prepare_motion(self):
        if self.document is None:
            return
        self.motion = self.motions['noitom']
        self.proposals = candidates(self.motion)
        self.slider.setRange(0, self.motion.count - 1)
        for spin in (self.start, self.end):
            spin.setMaximum(self.motion.duration)
            spin.setSingleStep(self.motion.dt)
        self.region.setBounds((0., self.motion.duration))
        self.region.setRegion((0., min(1., self.motion.duration)))
        self.region_changed()
        self.new_interval()
        self.timeline.setXRange(0, self.motion.duration)
        self.speed_curve.setData(np.arange(self.motion.count) * self.motion.dt, angular_speed(self.motion))
        self.slider.setValue(0)
        self.refresh_pose()

    def refresh_pose(self):
        if self.motion is None:
            return
        frame = self.slider.value()
        stamp = frame * self.motion.dt
        self.cursor.setValue(stamp)
        self.time_label.setText(f'{stamp:.2f} / {self.motion.duration:.2f} с')
        for source, curves in self.lines.items():
            for view_index, curve in enumerate(curves):
                if source not in self.motions:
                    curve.setData([], [])
                    continue
                motion = self.motions[source]
                positions = self.body[source][frame]
                vertices = []
                for i, name in enumerate(motion.names):
                    parent = motion.parents[i]
                    if parent is not None and name in ('Chest', 'Head', 'LeftArm', 'LeftForeArm', 'RightArm', 'RightForeArm'):
                        vertices.extend((positions[parent], positions[i], [np.nan] * 3))
                    if name in ('LeftForeArm', 'RightForeArm'):
                        wrist = positions[i] + self.body_rot[source][frame, i] @ motion.ends[i]
                        vertices.extend((positions[i], wrist, [np.nan] * 3))
                vertices = np.asarray(vertices)
                curve.setData(-vertices[:, 0] if view_index == 0 else vertices[:, 2], vertices[:, 1], connect='finite')
        hints = pose_hint(self.motion, frame, self.directions)
        self.hint.setText('Подсказки по направлению плеч (не разметка): ' +
                          ' · '.join(f'{letter}: {error:.0f}°' for error, letter in hints) +
                          '   | Е и Э имеют одну позу. Сверяйте обе проекции.')

    def toggle_play(self):
        if self.motion is None:
            return
        if self.timer.isActive():
            self.timer.stop()
        else:
            if self.slider.value() == self.slider.maximum():
                self.slider.setValue(0)
            self.play_started = time.perf_counter()
            self.play_frame = self.slider.value()
            self.timer.start()

    def tick(self):
        frame = self.play_frame + int((time.perf_counter() - self.play_started) * float(self.rate.currentText()) / self.motion.dt)
        self.slider.setValue(min(frame, self.slider.maximum()))
        if frame >= self.slider.maximum():
            self.timer.stop()

    def seek(self, event):
        if self.motion is None or event.button() != Qt.MouseButton.LeftButton:
            return
        point = self.timeline.plotItem.vb.mapSceneToView(event.scenePos())
        self.timer.stop()
        self.slider.setValue(round(point.x() / self.motion.dt))

    def region_changed(self):
        a, b = self.region.getRegion()
        for spin, value in ((self.start, a), (self.end, b)):
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)

    def spin_changed(self):
        self.region.setRegion((self.start.value(), self.end.value()))

    def rebuild_table(self):
        items = self.document['intervals']
        self.table.setRowCount(len(items))
        for row, item in enumerate(items):
            values = (f"{item['start_s']:.3f}", f"{item['end_s']:.3f}", item['label'],
                      'Да' if item['reviewed'] else 'Нет', item.get('annotator_note', ''))
            for col, value in enumerate(values):
                self.table.setItem(row, col, QTableWidgetItem(value))

    def select_interval(self, row, _column=0):
        if self.document is None:
            return
        item = self.document['intervals'][row]
        self.selected = item['id']
        self.region.setRegion((item['start_s'], item['end_s']))
        self.label.setCurrentText(item['label'])
        self.note.setText(item.get('annotator_note', ''))
        self.slider.setValue(round(item['start_s'] / self.motion.dt))

    def new_interval(self):
        self.selected = None
        self.table.clearSelection()
        self.note.clear()

    def next_candidate(self):
        if self.motion is None:
            return
        current = self.slider.value() * self.motion.dt
        for a, b in self.proposals:
            overlaps = any(a < x['end_s'] and b > x['start_s'] for x in self.document['intervals'])
            if a > current + self.motion.dt and not overlaps:
                self.new_interval()
                self.region.setRegion((a, b))
                self.slider.setValue(round(a / self.motion.dt))
                return
        self.status.setText('Дальше нет свободных устойчивых интервалов. Можно задать границы вручную.')

    def persist(self, items):
        validate_intervals(items, self.motion.duration)
        previous = deepcopy(self.document['intervals'])
        updated = deepcopy(self.document)
        updated['intervals'] = sorted(items, key=lambda x: x['start_s'])
        save_document(self.path, updated, self.motion.duration)
        self.undo_stack.append(previous)
        self.document = updated
        self.rebuild_table()
        self.status.setText(f'Сохранено интервалов: {len(items)} · {self.path}')

    def commit(self):
        if self.document is None:
            return
        item = interval(self.start.value(), self.end.value(), self.label.currentText(), self.note.text())
        item['annotation_source'] = 'noitom'
        item['annotation_tool'] = 'qt'
        if self.selected is not None:
            item['id'] = self.selected
        items = [x for x in self.document['intervals'] if x['id'] != self.selected] + [item]
        try:
            self.persist(items)
            self.selected = item['id']
        except (ValueError, OSError) as error:
            QMessageBox.warning(self, 'Метки не сохранены', str(error))

    def delete_interval(self):
        if self.document is None or self.selected is None:
            return
        try:
            self.persist([x for x in self.document['intervals'] if x['id'] != self.selected])
            self.new_interval()
        except (ValueError, OSError) as error:
            QMessageBox.warning(self, 'Метки не сохранены', str(error))

    def undo_change(self):
        if not self.undo_stack:
            return
        items = self.undo_stack[-1]
        try:
            updated = deepcopy(self.document)
            updated['intervals'] = items
            save_document(self.path, updated, self.motion.duration)
            self.undo_stack.pop()
            self.document = updated
            self.new_interval()
            self.rebuild_table()
        except (ValueError, OSError) as error:
            QMessageBox.warning(self, 'Метки не сохранены', str(error))


def launch(records, labels):
    app = QApplication.instance() or QApplication(sys.argv[:1])
    window = Annotator(records, labels)
    window.show()
    app.exec()
