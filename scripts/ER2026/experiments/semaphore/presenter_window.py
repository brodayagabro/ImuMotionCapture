"""Independent timed recordings: one word set, selected suits, one CSV row."""
from dataclasses import replace
from pathlib import Path
import csv
import math
import time

from PyQt6.QtCore import QTimer, QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout,
    QHBoxLayout, QLabel, QLineEdit, QMainWindow, QPushButton, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget, QProgressBar)
from .config import BLOCK_LABELS, ROOT, validate_participant
from .participant_window import ParticipantWindow
from .recorder_client import RecorderClient
from .word_loop import WordLoop


class PresenterWindow(QMainWindow):
    def __init__(self, config, *, participant='P001', host='127.0.0.1', port=5055,
                 seed=None, debug=False, backend=None):
        super().__init__()
        self.config, self.seed, self.debug, self.backend = config, seed, debug, backend
        self.host, self.port = host, port
        self.session_id = None
        self.loop = None
        self.controller = None
        self.deadline_ns = None
        self.starting = False
        self.display_started = False
        self.status = {}
        self.sessions_root = ROOT / 'records'
        self._last_index = None
        self.participant_window = ParticipantWindow(self._participant_hidden, config.mirror_mode)
        self.client = RecorderClient(self, continuous_control=True)
        self.client.controlled.connect(self._controlled)
        self.client.failed.connect(self._connection_failed)
        self.setWindowTitle('Семафор · запись костюмов' + (' · ПРЕДПРОСМОТР' if debug else ''))
        self.resize(850, 690)
        self.setStyleSheet("QWidget { font-family: 'Segoe UI'; font-size: 13px; } QPushButton { padding: 8px; }")
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        title = QLabel('Независимая запись набора слов')
        title.setStyleSheet('font-size: 23px; font-weight: 600; padding: 6px;')
        layout.addWidget(title)
        form = QFormLayout()
        self.participant_edit = QLineEdit(participant)
        self.word_set_combo = QComboBox()
        for block in config.blocks:
            self.word_set_combo.addItem(f'{config.word_set_ids[block]} · {BLOCK_LABELS[block]} · {len(config.words[block])} слов', config.word_set_ids[block])
        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setRange(1, 3600)
        self.duration_spin.setDecimals(1)
        self.duration_spin.setSuffix(' с')
        self.duration_spin.setValue(config.recording_seconds)
        self.export_fps_spin = QDoubleSpinBox()
        self.export_fps_spin.setRange(1, 100)
        self.export_fps_spin.setDecimals(1)
        self.export_fps_spin.setSuffix(' кадр/с')
        self.export_fps_spin.setValue(config.export_fps)
        self.export_fps_spin.setToolTip('Предел частоты выбранных BVH. Если выбранный костюм передаёт реже, частота файлов будет снижена.')
        self.transition_spin = QDoubleSpinBox()
        self.transition_spin.setRange(.01, 10)
        self.transition_spin.setDecimals(2)
        self.transition_spin.setSuffix(' с')
        self.transition_spin.setValue(config.timing.transition)
        self.mirror_check = QCheckBox('Зеркальный показ')
        self.mirror_check.setChecked(config.mirror_mode)
        self.screen_combo = QComboBox()
        for i, screen in enumerate(QApplication.screens()):
            self.screen_combo.addItem(f'Экран {i + 1} · {screen.name()}', i)
        self.fullscreen_check = QCheckBox('Полный экран участника (F11 / Esc)')
        self.fullscreen_check.setChecked(True)
        for label, widget in (('Испытуемый', self.participant_edit), ('Набор слов', self.word_set_combo),
                ('Продолжительность', self.duration_spin), ('Частота BVH', self.export_fps_spin),
                ('Переход между буквами', self.transition_spin),
                ('Экран участника', self.screen_combo)):
            form.addRow(label, widget)
        layout.addLayout(form)
        options = QHBoxLayout()
        options.addWidget(self.mirror_check)
        options.addWidget(self.fullscreen_check)
        layout.addLayout(options)
        self.source_label = QLabel('Подключите костюмы в окне захвата')
        self.calibration_label = QLabel('Калибровка: ожидается')
        layout.addWidget(self.source_label)
        layout.addWidget(self.calibration_label)
        calibration = QHBoxLayout()
        self.calibrate_button = QPushButton('Калибровка')
        self.imu_button = QPushButton('IMU-калибровка')
        self.noitom_button = QPushButton('Noitom откалиброван в Axis Studio')
        self.calibrate_button.clicked.connect(lambda: self._send('calibrate'))
        self.imu_button.clicked.connect(lambda: self._send('imu_calibrate'))
        self.noitom_button.clicked.connect(lambda: self._send('confirm_noitom_calibration'))
        for button in (self.calibrate_button, self.imu_button, self.noitom_button):
            calibration.addWidget(button)
        layout.addLayout(calibration)
        controls = QHBoxLayout()
        self.start_button = QPushButton('Предпросмотр' if debug else 'Начать запись')
        self.stop_button = QPushButton('Остановить досрочно')
        self.connect_button = QPushButton('Подключить рекордер')
        self.connect_button.setVisible(backend is None and not debug)
        self.connect_button.clicked.connect(self._connect)
        self.start_button.clicked.connect(self._start)
        self.stop_button.clicked.connect(self._stop)
        self.timer_label = QLabel('05:00')
        self.timer_label.setStyleSheet('font-size: 28px; font-weight: 600;')
        for widget in (self.connect_button, self.start_button, self.stop_button, self.timer_label):
            controls.addWidget(widget)
        layout.addLayout(controls)
        self.result_label = QLabel('Удержание буквы: 1,6 с. После окончания набора показ повторяется.')
        self.result_label.setWordWrap(True)
        layout.addWidget(self.result_label)
        self.export_progress = QProgressBar()
        self.export_progress.setRange(0, 100)
        self.export_progress.setValue(0)
        self.export_progress.setFormat('Сохранение BVH: %p%')
        self.export_progress.hide()
        layout.addWidget(self.export_progress)
        self.records_label = QLabel('Успешные записи · records/records.csv')
        layout.addWidget(self.records_label)
        self.records_table = QTableWidget(0, 5)
        self.records_table.setHorizontalHeaderLabels(['Начало', 'Испытуемый', 'Набор', 'custom.bvh', 'noitom.bvh'])
        self.records_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.records_table, 1)
        open_records = QPushButton('Открыть папку записей')
        open_records.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.sessions_root))))
        layout.addWidget(open_records)
        self.poll_timer = QTimer(self)
        self.poll_timer.setInterval(250)
        self.poll_timer.timeout.connect(self._poll)
        self.poll_timer.start()
        self.render_timer = QTimer(self)
        self.render_timer.setInterval(16)
        self.render_timer.timeout.connect(self._tick)
        self.render_timer.start()
        self._refresh()
        self._load_records()
        if backend is None and not debug:
            QTimer.singleShot(0, self._connect)

    def _connect(self):
        self.client.connect_recorder(self.host, self.port)

    def _send(self, command, **payload):
        try:
            if self.backend is not None:
                result = self.backend({'command': command, **payload})
                self._controlled({'accepted': True, 'command': command, **result})
            else:
                self.client.send_control(command, **payload)
        except (ValueError, OSError, RuntimeError) as error:
            self._controlled({'accepted': False, 'command': command, 'reason': str(error)})

    def _poll(self):
        if not self.debug and (self.backend is not None or self.client.control_supported):
            if not self.client.control_pending:
                self._send('status', recording_session_id=self.session_id)
        self._load_records()

    def _controlled(self, message):
        if not message.get('accepted'):
            self.result_label.setText('Ошибка: ' + str(message.get('reason')))
            if message.get('command') == 'start_recording':
                self.starting = False
                self._end_display()
                self.participant_window.show_status('Запись не началась', str(message.get('reason')))
            self._refresh()
            return
        self.status = message
        self.sessions_root = Path(message.get('records_root', self.sessions_root))
        if message.get('command') == 'start_recording':
            self.session_id = message.get('last_recording_id')
            self.starting = False
        own = 'поток есть' if message.get('own_fresh') else 'нет свежих данных'
        noitom = 'поток есть' if message.get('noitom_fresh') else 'нет свежих данных'
        sources = message.get('recording_sources', ['custom', 'noitom'])
        if 'custom' not in sources:
            own = 'не записывается'
        if 'noitom' not in sources:
            noitom = 'не записывается'
        self.source_label.setText(f'ESP32: {own}    |    Noitom: {noitom}')
        own_cal = message.get('own_calibration_method') == 'neutral_a_semaphore_a_t_u_neutral_a_sensor_y'
        own_cal_text = ('готова' if own_cal else 'нужна') if 'custom' in sources else 'не требуется'
        noitom_cal_text = ('подтверждена' if message.get('noitom_calibration') else 'нужна') if 'noitom' in sources else 'не требуется'
        self.calibration_label.setText('Калибровка: ' + ('выполняется' if message.get('calibration_running') else
            f'ESP32 {own_cal_text} · Noitom {noitom_cal_text}'))
        if self.session_id and message.get('last_recording_id') == self.session_id:
            if message.get('ready') and not self.display_started:
                self.deadline_ns = message['recording_t1_ns']
                self._begin_display()
            status = message.get('last_recording_status')
            progress = message.get('export_progress') or {}
            if status in {'saving', 'complete', 'error'}:
                self.export_progress.show()
                self.export_progress.setValue(int(progress.get('percent', 0)))
            if status in {'complete', 'incomplete', 'error'}:
                self._end_display()
                if status == 'complete':
                    data = message.get('last_export', {})
                    files = ' и '.join(name + '.bvh' for name in message.get('last_recording_sources', ['custom', 'noitom']))
                    self.result_label.setText(f'Сохранено: {files} · {data.get("frames")} кадров · '
                        f'{data.get("fps", 0):.3f} Гц\n{message.get("last_recording_path")}')
                    self.participant_window.show_status('Запись сохранена', 'Спасибо за участие')
                else:
                    self.result_label.setText('Запись неполная, в records.csv не добавлена: ' + str(message.get('last_recording_error')))
                    self.participant_window.show_status('Запись остановлена', str(message.get('last_recording_error')))
                self.session_id = None
            elif status == 'saving':
                self._end_display()
                phase = progress.get('phase', 'Подготовка сохранения')
                self.result_label.setText(phase)
                self.participant_window.show_export_progress(int(progress.get('percent', 0)), phase)
        self._refresh()

    def _show_participant(self):
        screen = QApplication.screens()[self.screen_combo.currentData() or 0]
        self.participant_window.showNormal()
        self.participant_window.windowHandle().setScreen(screen)
        self.participant_window.move(screen.geometry().topLeft())
        if self.fullscreen_check.isChecked():
            self.participant_window.showFullScreen()

    def _start(self):
        try:
            subject = validate_participant(self.participant_edit.text().strip())
        except ValueError as error:
            self.result_label.setText(str(error))
            return
        self.starting = True
        self.display_started = False
        self.deadline_ns = None
        self.export_progress.setValue(0)
        self.export_progress.hide()
        self.result_label.setText('Ожидание подтверждения записи выбранных костюмов…')
        self.participant_window.show_status('Приготовьтесь', 'Ожидание начала записи выбранных костюмов…')
        self._show_participant()
        if self.debug:
            self.session_id = 'preview'
            self.deadline_ns = time.perf_counter_ns() + round(self.duration_spin.value() * 1e9)
            self._begin_display()
            self.starting = False
        else:
            self._send('start_recording', subject_id=subject, word_set_id=self.word_set_combo.currentData(),
                       duration_s=self.duration_spin.value(), export_fps=self.export_fps_spin.value())
        self._refresh()

    def _begin_display(self):
        self.display_started = True
        config = replace(self.config, mirror_mode=self.mirror_check.isChecked(),
                         timing=replace(self.config.timing, transition=self.transition_spin.value()))
        self.loop = WordLoop(config, self.word_set_combo.currentData(), seed=self.seed)
        self.controller = self.loop.controller
        self.result_label.setText('Предпросмотр · данные не записываются' if self.debug else 'Идёт синхронная запись двух костюмов')

    def _tick(self):
        remaining = self.duration_spin.value() if self.deadline_ns is None else max(0., (self.deadline_ns - time.perf_counter_ns()) / 1e9)
        seconds = math.ceil(remaining)
        self.timer_label.setText(f'{seconds // 60:02}:{seconds % 60:02}')
        if self.loop:
            if remaining <= 0:
                self._end_display()
                if self.debug:
                    self.session_id = None
                    self._refresh()
            else:
                self.loop.tick()
                self.controller = self.loop.controller
                self.participant_window.display(self.controller)

    def _end_display(self):
        if self.loop:
            self.loop.stop()
            self.participant_window.display(self.loop.controller)
        self.loop = None

    def _stop(self):
        if self.session_id and not self.debug:
            self._send('stop_recording', recording_session_id=self.session_id)
        elif self.debug:
            self.session_id = None
        self._end_display()
        self._refresh()

    def _participant_hidden(self):
        if self.session_id:
            self._stop()

    def _connection_failed(self, message):
        self._end_display()
        self.starting = False
        self.result_label.setText('Связь потеряна. Запись будет помечена неполной: ' + message)
        self.participant_window.show_status('Связь потеряна', message)
        self.status = {}
        self._refresh()

    def _refresh(self):
        busy = bool(self.starting or self.session_id or self.status.get('saving') or self.status.get('arming') or self.status.get('ready'))
        cal_busy = self.status.get('calibration_running')
        sources = self.status.get('recording_sources', ['custom', 'noitom'])
        own_ready = (self.status.get('own_streaming', self.status.get('own_fresh')) and
                     self.status.get('own_fresh') and
                     self.status.get('own_calibration_method') == 'neutral_a_semaphore_a_t_u_neutral_a_sensor_y')
        noitom_ready = (self.status.get('noitom_streaming', self.status.get('noitom_fresh')) and
                        self.status.get('noitom_fresh') and self.status.get('noitom_calibration'))
        ready = (bool(sources) and ('custom' not in sources or own_ready)
                 and ('noitom' not in sources or noitom_ready))
        self.start_button.setEnabled(bool(not busy and not cal_busy and (self.debug or ready)))
        self.stop_button.setEnabled(bool(self.session_id) and not self.status.get('saving'))
        for widget in (self.participant_edit, self.word_set_combo, self.duration_spin, self.export_fps_spin, self.transition_spin,
                       self.mirror_check, self.screen_combo, self.fullscreen_check):
            widget.setEnabled(not busy)
        for button in (self.calibrate_button, self.imu_button, self.noitom_button):
            source = 'noitom' if button is self.noitom_button else 'custom'
            button.setEnabled(not busy and not cal_busy and not self.debug and source in sources)

    def _load_records(self):
        path = self.sessions_root / 'records.csv'
        try:
            key = (str(path), path.stat().st_mtime_ns)
            if key == self._last_index:
                return
            with path.open(encoding='utf-8', newline='') as stream:
                rows = list(csv.reader(stream))
            self.records_table.setRowCount(max(0, len(rows) - 1))
            for i, row in enumerate(reversed(rows[1:])):
                for j, value in enumerate(row[:5]):
                    self.records_table.setItem(i, j, QTableWidgetItem(value))
            self._last_index = key
            self.records_label.setText('Успешные записи · ' + str(path))
        except FileNotFoundError:
            return
        except (OSError, csv.Error) as error:
            self.records_label.setText('Не удалось прочитать records.csv: ' + str(error))

    def closeEvent(self, event):
        self._stop()
        self.poll_timer.stop()
        self.render_timer.stop()
        self.client.close()
        self.participant_window.hide()
        event.accept()

    def showEvent(self, event):
        self.poll_timer.start()
        self.render_timer.start()
        super().showEvent(event)
