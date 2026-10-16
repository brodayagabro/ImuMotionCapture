"""Two independent Qt windows: operator controls and a clean participant view."""
from dataclasses import replace
import secrets
import time
import uuid
from pathlib import Path

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFormLayout, QGridLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QMainWindow, QProgressBar,
    QPushButton, QTabWidget, QVBoxLayout, QWidget,
    QDoubleSpinBox, QFileDialog, QScrollArea, QSizePolicy,
)

from .config import BLOCK_LABELS, ROOT, generate_trials, validate_participant
from .dataset_tab import DatasetTab
from .experiment_controller import ExperimentController, State, TIMED_STATES
from .pose_browser import PoseBrowser
from .recorder_client import RecorderClient
from .semaphore_renderer import ACCENT, BACKGROUND, INK, SemaphoreRenderer, WordStrip
from .session_log import SessionLog
from .participant_profile import load_profile, save_profile
from .series_coordinator import SeriesCoordinator


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


class PresenterWindow(QMainWindow):
    def __init__(self, config, *, participant="P001", host="127.0.0.1", port=5055, seed=None, debug=False):
        super().__init__()
        self.config, self.host, self.port, self.debug = config, host, port, debug
        self.controller = self.session_log = None
        self._catalog_session = None
        self._profile_participant = None
        sessions_path = Path(config.sessions_dir)
        self.sessions_root = (sessions_path if sessions_path.is_absolute() else ROOT / sessions_path).resolve()
        self._connection_lost = self._closing = self._log_failed = False
        self._fps_start, self._frames, self._fps = time.monotonic(), 0, 0.
        self.client = RecorderClient(self)
        self.series = SeriesCoordinator(self)
        self.participant_window = ParticipantWindow(self._participant_hidden, config.mirror_mode)
        self.setWindowTitle("Semaphore Experiment Presenter · оператор")
        self.resize(780, 780)
        self.setStyleSheet("QWidget { font-family: 'Segoe UI'; font-size: 13px; }"
            "QPushButton { padding: 8px 12px; } QGroupBox { margin-top: 12px; padding-top: 12px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 12px; }")
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        title = QLabel("Семафорный эксперимент")
        title.setStyleSheet("font-size: 24px; font-weight: 600; padding: 6px;")
        layout.addWidget(title)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)
        experiment = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(experiment)
        self.tabs.addTab(scroll, "Эксперимент")
        el = QVBoxLayout(experiment)
        self.settings_box = QGroupBox("Протокол")
        form = QFormLayout(self.settings_box)
        self.participant_edit = QLineEdit(participant)
        self.participant_edit.setPlaceholderText("Код участника, например P001")
        form.addRow("Код участника", self.participant_edit)
        self.handedness = QComboBox()
        for title, value in (("Не указана", ""), ("Правая", "right"), ("Левая", "left"), ("Обе", "ambidextrous")):
            self.handedness.addItem(title, value)
        form.addRow("Ведущая рука", self.handedness)
        self.participant_notes = QLineEdit()
        self.participant_notes.setMaxLength(2000)
        self.participant_notes.setPlaceholderText("Необязательное примечание об участнике")
        form.addRow("Примечание", self.participant_notes)
        save_participant = QPushButton("Сохранить карточку участника")
        save_participant.clicked.connect(self._save_participant_profile)
        form.addRow(save_participant)
        self.block_order = QComboBox()
        self.block_order.addItem("Нижний → верхний", ("lower", "upper"))
        self.block_order.addItem("Верхний → нижний", ("upper", "lower"))
        self.block_order.setCurrentIndex(0 if config.blocks[0] == "lower" else 1)
        form.addRow("Порядок блоков", self.block_order)
        self.seed_edit = QLineEdit("" if seed is None else str(seed))
        self.seed_edit.setPlaceholderText("Автоматически; сохранится в сессии")
        form.addRow("Seed", self.seed_edit)
        self.short_series = QCheckBox('Короткие серии с повторной калибровкой')
        self.short_series.setChecked(config.series_seconds > 0)
        form.addRow(self.short_series)
        self.series_minutes = QDoubleSpinBox()
        self.series_minutes.setRange(.1, 120.)
        self.series_minutes.setValue((config.series_seconds or 300.) / 60.)
        self.series_minutes.setSuffix(' мин')
        presets = QHBoxLayout()
        for minutes in (5, 10):
            button = QPushButton(f'{minutes} минут')
            button.clicked.connect(lambda checked=False, m=minutes: self.series_minutes.setValue(m))
            presets.addWidget(button)
        presets.addWidget(self.series_minutes)
        form.addRow('Целевая длительность', presets)
        self.timing_inputs = {}
        for key, title, value in (('hold', 'Удержание буквы', config.timing.hold),
                                 ('transition', 'Переход', config.timing.transition),
                                 ('inter_trial', 'Между словами', config.timing.inter_trial),
                                 ('series_prepare', 'Подготовка перед серией', config.series_prepare)):
            spin = QDoubleSpinBox()
            spin.setRange(.01 if key in ('hold', 'transition') else 0., 120.)
            spin.setDecimals(2)
            spin.setSuffix(' с')
            spin.setValue(value)
            self.timing_inputs[key] = spin
            form.addRow(title, spin)
        count = sum(len(w) for w in config.words.values()) * config.repetitions
        form.addRow("Объём", QLabel(f"{count} проб · {config.repetitions} повтора · два блока"))
        form.addRow("Показ", QLabel("Левая рука справа на изображении" if config.mirror_mode else "Левая рука слева на изображении"))
        self.offline = QCheckBox("Явно запустить без рекордера (только отладка)")
        if debug:
            form.addRow(self.offline)
        else:
            self.offline.hide()
        el.addWidget(self.settings_box)
        network = QHBoxLayout()
        self.connect_button = QPushButton("Подключить рекордер")
        self.connect_button.clicked.connect(self.connect_recorder)
        network.addWidget(self.connect_button)
        self.connection_label = QLabel(f"DISCONNECTED · {host}:{port}")
        self.connection_label.setWordWrap(True)
        network.addWidget(self.connection_label, 1)
        el.addLayout(network)
        self.series_summary = QLabel('Создайте план, чтобы увидеть серии и расчётное время.')
        self.series_summary.setWordWrap(True)
        el.addWidget(self.series_summary)
        self.calibration_status = QLabel('Калибровки: подключите рекордер')
        self.calibration_status.setWordWrap(True)
        el.addWidget(self.calibration_status)
        series_bar = QGridLayout()
        self.series_buttons = {}
        for key, label, callback in (
            ('calibrate', 'Калибровка IMU', lambda: self.client.send_control('calibrate')),
            ('noitom', 'Подтвердить калибровку Noitom', lambda: self.client.send_control('confirm_noitom_calibration')),
            ('next', 'Начать серию', self.series.start),
            ('load', 'Продолжить сохранённый план', self._load_series_plan)):
            button = QPushButton(label)
            button.clicked.connect(lambda checked=False, fn=callback: self._series_action(fn))
            position = len(self.series_buttons)
            series_bar.addWidget(button, position // 2, position % 2)
            self.series_buttons[key] = button
        self.series_buttons['noitom'].setToolTip('Нажмите после выполнения новой калибровки Noitom в Axis Studio')
        el.addLayout(series_bar)
        self.stats = QLabel("Проверьте позы и запустите тренировку перед экспериментом.")
        self.stats.setMinimumHeight(100)
        self.stats.setWordWrap(True)
        self.stats.setStyleSheet("padding: 10px; background: #edf2f2; border-radius: 6px;")
        el.addWidget(self.stats)
        self.buttons = {}
        grid = QGridLayout()
        actions = [
            ("start", "Начать эксперимент", lambda: self.start_session(False)),
            ("training", "Тренировка", lambda: self.start_session(True)),
            ("block", "Начать блок", lambda: self.action("start_block")),
            ("pause", "Пауза", lambda: self.action("pause")),
            ("resume", "Продолжить", lambda: self.action("resume")),
            ("stop", "Завершить", lambda: self.action("stop")),
            ("repeat", "Повторить пробу", lambda: self.action("repeat_trial")),
            ("skip", "Пропустить пробу", lambda: self.action("skip_trial")),
        ]
        for i, (key, label, action) in enumerate(actions):
            button = QPushButton(label)
            button.clicked.connect(action)
            grid.addWidget(button, i // 3, i % 3)
            self.buttons[key] = button
        el.addLayout(grid)
        self.notice = QLabel("")
        self.notice.setWordWrap(True)
        self.notice.setStyleSheet("color: #925218; padding: 4px;")
        el.addWidget(self.notice)
        self.path_label = QLabel("")
        self.path_label.setWordWrap(True)
        self.path_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        el.addWidget(self.path_label)
        el.addStretch()
        self.pose_browser = PoseBrowser(config.mirror_mode)
        self.pose_browser.pose_selected.connect(self.preview_pose)
        self.tabs.addTab(self.pose_browser, "Проверка поз")
        self.dataset_tab = DatasetTab(self.sessions_root)
        self.tabs.addTab(self.dataset_tab, "Датасет")
        self.participant_edit.editingFinished.connect(self._load_participant_profile)
        self._load_participant_profile()
        display = QHBoxLayout()
        self.screens = QComboBox()
        for i, screen in enumerate(QApplication.screens()):
            g = screen.geometry()
            self.screens.addItem(f"Экран {i+1} · {screen.name()} · {g.width()}×{g.height()}", screen)
        display.addWidget(self.screens, 1)
        show = QPushButton("Окно участника")
        show.clicked.connect(self.show_participant)
        display.addWidget(show)
        fullscreen = QPushButton("Полный экран · F11")
        fullscreen.clicked.connect(self.fullscreen_participant)
        display.addWidget(fullscreen)
        layout.addLayout(display)
        self.client.status_changed.connect(self.connection_label.setText)
        self.client.readiness_changed.connect(self._readiness_changed)
        self.client.failed.connect(self._wire_failed)
        self.client.acknowledged.connect(self._acknowledged)
        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.setInterval(16)
        self.timer.timeout.connect(self._tick)
        self.timer.start()
        self._refresh()

    def _load_participant_profile(self):
        try:
            participant = self.participant_edit.text().strip()
            if participant == self._profile_participant:
                return
            profile = load_profile(self.sessions_root, participant)
            self.handedness.setCurrentIndex(self.handedness.findData(profile["handedness"]))
            self.participant_notes.setText(profile["notes"])
            self._profile_participant = participant
        except (ValueError, OSError) as error:
            self.notice.setText("Карточка участника: " + str(error))

    def _save_participant_profile(self):
        try:
            participant = self.participant_edit.text().strip()
            save_profile(self.sessions_root, participant,
                         {"handedness": self.handedness.currentData(), "notes": self.participant_notes.text()})
            self.notice.setText("Карточка участника сохранена")
            self.dataset_tab.request_refresh()
        except (ValueError, OSError) as error:
            self.notice.setText("Карточка участника: " + str(error))

    def connect_recorder(self):
        try:
            self.client.connect_recorder(self.host, self.port)
        except ValueError as error:
            self.notice.setText(str(error))

    def show_participant(self):
        view = self.participant_window
        view.show()
        screen = self.screens.currentData()
        if screen is not None:
            view.windowHandle().setScreen(screen)
            g = screen.availableGeometry()
            if not view.isFullScreen():
                view.move(g.x() + max(0, (g.width() - view.width()) // 2), g.y() + 20)
        view.raise_()

    def fullscreen_participant(self):
        self.show_participant()
        self.participant_window.toggle_fullscreen()

    def _participant_hidden(self):
        if self.controller and self.controller.active and not self._closing:
            self.controller.pause(reason="participant_window_hidden")
            self.controller.recovery_required = self.controller.trial_open
            self.notice.setText("Окно участника закрыто. Откройте его и повторите пробу.")

    def preview_pose(self, pose, mirror, letter):
        if self.controller and self.controller.active:
            return
        view = self.participant_window
        view.avatar.mirror_mode = mirror
        view.avatar.set_pose(pose.left_angle_deg, pose.right_angle_deg)
        view.word.set_word(letter, 0)
        view.caption.setText("Проверка положения рук")
        view.cue.setText("Повторяйте положение рук")
        view.progress.setValue(0)

    def start_session(self, training=False):
        if self.series.busy or self._closing or (self.controller and self.controller.active):
            return
        if self.dataset_tab.busy:
            self.notice.setText("Дождитесь окончания обновления реестра")
            return
        offline = self.debug and self.offline.isChecked()
        series_mode = self.short_series.isChecked() and not training
        if not series_mode and ((not self.client.ready and not offline) or self.client.pending):
            self.notice.setText("Нужны готовый рекордер и подтверждения предыдущих событий.")
            return
        try:
            participant = validate_participant(self.participant_edit.text().strip())
            seed = int(self.seed_edit.text()) if self.seed_edit.text().strip() else secrets.randbits(32)
            config = self._selected_config()
            if series_mode:
                self.seed_edit.setText(str(seed))
                self.series.create(config, participant, seed,
                    {'handedness': self.handedness.currentData(), 'notes': self.participant_notes.text()})
                self.notice.setText('План сохранён. Выполните калибровки, затем нажмите «Начать серию».')
                return
            trials = generate_trials(config, seed, training=training)
            if self.session_log:
                self.session_log.close()
            self.session_log = SessionLog(config, trials, participant, seed, offline=offline,
                participant_info={"handedness": self.handedness.currentData(), "notes": self.participant_notes.text()})
            self._log_failed = self._connection_lost = False
            self.controller = ExperimentController(config, trials, self.session_log.context, self._emit, offline=offline)
            self.controller.set_recorder_ready(self.client.ready)
            self.controller.start()
            self.path_label.setText("Сессия: " + str(self.session_log.path))
            self.notice.setText("Запуск без рекордера: меток в MoCap не будет." if offline else "")
            self.tabs.setCurrentIndex(0)
            self.show_participant()
            self._refresh()
        except (ValueError, OSError) as error:
            self.notice.setText("Не удалось начать: " + str(error))

    def _emit(self, payload):
        recording_id = (self.controller.context.get('recording_session_id') if payload.get('series_id')
                        else self.client.recording_session_id)
        payload = {**payload, "event_id": uuid.uuid4().hex,
                   "recording_session_id": None if self.controller.offline else recording_id}
        if not self._log_failed:
            try:
                self.session_log.append(payload)
                if payload.get('series_id') and self.series.plan:
                    self.series.plan.mark_event(payload)
            except (OSError, ValueError) as error:
                self._logging_error(error)
        if not self.controller.offline and (recording_id != self.client.recording_session_id or not self.client.send_event(payload)):
            self._connection_lost = True
            self._wire_failed("Не отправлено событие " + payload["type"] + " · " + payload["event_id"])

    def _readiness_changed(self, ready):
        if not ready:
            # Defer controller changes until the next timer turn; send failures
            # must not re-enter a phase transition while it is emitting events.
            self._connection_lost = True
            if self.controller and not self.controller.active and not self._closing:
                self.dataset_tab.request_refresh()

    def _wire_failed(self, message):
        if self.series.phase in ('starting', 'initializing', 'stopping', 'saving'):
            self.series.phase = 'recovering'
            self.series.current = self.series.plan.data['series'][-1]
        if self.controller and self.controller.offline:
            return
        if self.session_log and not self.session_log.closed and not self._log_failed:
            try:
                self.session_log.issue(message)
            except OSError as error:
                self._logging_error(error)
        self.notice.setText(message + ". Восстановите связь и повторите текущую пробу.")

    def _selected_config(self):
        timing = replace(self.config.timing, **{k: v.value() for k, v in self.timing_inputs.items() if k != 'series_prepare'})
        return replace(self.config, timing=timing, blocks=self.block_order.currentData(),
                       series_seconds=self.series_minutes.value() * 60 if self.short_series.isChecked() else 0.,
                       series_prepare=self.timing_inputs['series_prepare'].value())

    def _series_action(self, callback):
        try:
            callback()
        except (ValueError, OSError) as error:
            self.notice.setText(str(error))

    def _load_series_plan(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Продолжить эксперимент', str(self.sessions_root / 'experiments'), 'План (plan.json)')
        if path:
            self.short_series.setChecked(True)
            self.series.load(path)

    def _launch_series(self, plan, trials, context, offline):
        if self.session_log:
            self.session_log.close()
        context = {**context, 'initial_attempts': dict(plan.data['next_attempts']),
                   'recording_session_id': None if offline else plan.data['series'][-1]['recording_session_id']}
        self.session_log = SessionLog(plan.config, trials, plan.data['participant_id'], plan.data['random_seed'],
            participant_info=plan.data['participant_info'], offline=offline, series_context=context)
        self._log_failed = self._connection_lost = False
        self.controller = ExperimentController(plan.config, trials, self.session_log.context, self._emit, offline=offline)
        self.controller.set_recorder_ready(self.client.ready)
        self.controller.start()
        if self._closing:
            self.controller.stop()
        else:
            self.controller.start_block()
        self.path_label.setText('Серия: ' + str(self.session_log.path))
        self.show_participant()

    def _acknowledged(self, ack):
        if self.session_log and not self.session_log.closed and not self._log_failed:
            try:
                self.session_log.acknowledge(ack)
                if self.controller.state == State.FINISHED and not self.client.pending:
                    self.session_log._save_metadata()
            except OSError as error:
                self._logging_error(error)

    def _logging_error(self, error):
        self._log_failed = True
        self.notice.setText("Ошибка сохранения сессии: " + str(error) + ". Эксперимент остановлен.")

    def action(self, name):
        if not self.controller:
            return
        try:
            if name in {"start_block", "repeat_trial", "resume"}:
                self.show_participant()
            getattr(self.controller, name)()
            self._refresh()
        except ValueError as error:
            self.notice.setText(str(error))

    def _tick(self):
        c = self.controller
        if c:
            if (c.active and c.context.get('series_id') and not c.offline and self.client.ready
                    and self.client.recording_session_id != c.context.get('recording_session_id')):
                self.series.interrupted = True
                c.stop()
            lost, self._connection_lost = self._connection_lost, False
            if lost:
                c.set_recorder_ready(False)
            c.set_recorder_ready(self.client.ready)
            if self._log_failed:
                c.stop()
            c.tick()
            if not c.active and not self.client.pending and not self._log_failed and not self.series.busy:
                self._finish_session_log()
        try:
            self.series.tick()
        except (ValueError, OSError) as error:
            self._logging_error(error)
        self._frames += 1
        now = time.monotonic()
        if now - self._fps_start >= 1:
            self._fps = self._frames / (now - self._fps_start)
            self._frames, self._fps_start = 0, now
        self._refresh()
        if self._closing and not self.client.pending:
            self.close()

    def _refresh(self):
        c = self.controller
        active = c is not None and c.active
        ready = self.client.ready or (self.debug and self.offline.isChecked())
        self.settings_box.setEnabled(not active and not self._closing and not self.series.busy)
        self.tabs.setTabEnabled(1, not active and not self._closing)
        self.dataset_tab.refresh.setEnabled(not active and not self._closing and not self.dataset_tab.busy)
        self.connect_button.setEnabled(not self._closing and (not active or not self.client.ready))
        self.buttons["start"].setEnabled(not active and ready and not self.client.pending and not self._closing and not self.dataset_tab.busy)
        self.buttons["training"].setEnabled(self.buttons["start"].isEnabled() and not self.series.busy)
        if self.short_series.isChecked():
            self.buttons['start'].setText('Создать план эксперимента')
            self.buttons['start'].setEnabled(not active and not self.series.busy and not self._closing)
            self.buttons['stop'].setText('Завершить серию досрочно')
        else:
            self.buttons['start'].setText('Начать эксперимент')
            self.buttons['stop'].setText('Завершить')
        for key, button in self.series_buttons.items():
            button.setEnabled(not active and not self.series.busy and not self._closing and
                (key == 'load' or self.client.control_supported or (self.debug and self.offline.isChecked())))
        status = self.client.control_state
        noitom_cal = status.get('noitom_calibration') or {}
        previous = self.series.plan.data['previous_calibrations'] if self.series.plan else {}
        calibrated = (status.get('own_calibration_method') == 'neutral_a_semaphore_a_t_u_neutral_a_sensor_y'
                      and status.get('own_calibration_id')
                      and noitom_cal.get('calibration_id')
                      and status['own_calibration_id'] != previous.get('own_calibration_id')
                      and noitom_cal['calibration_id'] != previous.get('noitom_calibration_id')
                      and not status.get('calibration_running') and not status.get('ready')
                      and not status.get('saving'))
        self.series_buttons['next'].setEnabled(bool(self.series.plan and self.series.plan.remaining and
            not active and not self.series.busy and not self._closing and not self.client.control_pending
            and not self.client.pending and
            ((self.client.control_supported and calibrated) or (self.debug and self.offline.isChecked()))))
        phase_label = {'idle': 'ожидание плана', 'between': 'между сериями', 'complete': 'завершено',
                       'starting': 'запуск записи', 'initializing': 'подготовка записи', 'running': 'показ',
                       'draining': 'подтверждение меток', 'stopping': 'остановка записи', 'saving': 'сохранение',
                       'recovering': 'восстановление серии'}[self.series.phase]
        self.calibration_status.setText('IMU: ' + (status.get('own_calibration_id') or 'нет профиля') +
            ' · Noitom: ' + (noitom_cal.get('calibration_id') or 'нет подтверждения') +
            ' · запись: ' + ('идёт' if self.client.ready else 'остановлена') + ' · ' + phase_label)
        for key in ("block", "pause", "resume", "stop", "repeat", "skip"):
            self.buttons[key].setEnabled(bool(active and not self._closing))
        if c:
            self.buttons["block"].setEnabled(active and c.state == State.READY and ready and not self._closing)
            self.buttons["block"].setText("Начать следующий блок" if c.trial_index else "Начать блок")
            self.buttons["pause"].setEnabled(active and c.state in TIMED_STATES and not self._closing)
            self.buttons["resume"].setEnabled(c.state == State.PAUSED and not c.recovery_required and ready and not self._closing)
            for key in ("repeat", "skip"):
                self.buttons[key].setEnabled(active and c.trial_open and ready and not self._closing)
            t = c.trial
            words = sum(1 for row in c.trials if row.block == t.block and row.repeat == t.repeat)
            letter = "—" if c.letter_index is None else f"{t.word[c.letter_index]} · {c.letter_index+1}/{len(t.word)}"
            repeats = 1 if t.training_trial else c.config.repetitions
            text = (f"{'Тренировка' if t.training_trial else 'Эксперимент'} · проба {c.trial_index+1}/{len(c.trials)} · попытка {c.attempt}\n"
                    f"{BLOCK_LABELS[t.block]} · слово {t.word_index}/{words} · повтор {t.repeat}/{repeats}\n"
                    f"{t.word} · буква {letter} · состояние {c.state.value}")
            if c.recovery_required:
                text += "\nТребуется повтор текущей пробы"
            if self.debug:
                pose = c.pose()
                text += f"\nЛ {pose.left_angle_deg:.1f}° · П {pose.right_angle_deg:.1f}° · GUI {self._fps:.1f} FPS"
            text += f"\nОжидают подтверждения: {len(self.client.pending)}"
            if c.context.get('series_id') and self.series.plan:
                from .config import series_duration
                remaining_seconds = series_duration(c.config, c.trials[c.trial_index:])
                text += f'\nСерия {c.context["series_index"]} · выполнено {len(self.series.plan.data["completed_trial_ids"])}'
                text += f' · осталось {len(self.series.plan.remaining)} · до конца серии ≈ {remaining_seconds/60:.1f} мин'
                text += f' · номер пробы в полном плане: {t.trial_id}'
            self.stats.setText(text)
            if active or self.tabs.currentIndex() == 0:
                self.participant_window.display(c)

    def closeEvent(self, event):
        self._closing = True
        if self.controller:
            self.controller.stop()
        disconnected_recovery = (self.series.phase in ('recovering', 'draining')
                                 and not self.client.control_supported)
        # The active series remains on disk. Loading this plan reconciles and
        # stops its recording by ID after reconnecting, before any new stimuli.
        if self.client.pending or (self.series.busy and not disconnected_recovery) or self.client.control_pending:
            self.notice.setText("Завершение: ожидаются подтверждения последних событий…")
            event.ignore()
            return
        self._finish_session_log()
        if self.dataset_tab.busy:
            event.ignore()
            return
        self.timer.stop()
        self.client.close()
        self.participant_window.close()
        event.accept()

    def _finish_session_log(self):
        if not self.session_log or self._catalog_session == self.session_log.context["session_id"]:
            return
        try:
            self.session_log.close()
            self._catalog_session = self.session_log.context["session_id"]
            self.dataset_tab.request_refresh()
        except OSError as error:
            self._log_failed = True
            self.notice.setText("Ошибка сохранения сессии: " + str(error))
            # Do not trap the application in an infinite close loop on a full disk.
            self._catalog_session = self.session_log.context["session_id"]
