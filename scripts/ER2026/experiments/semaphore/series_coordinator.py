"""Recording/calibration handshake around the existing presenter controller."""
import time

from .series_plan import SeriesPlan
from .config import series_duration


class SeriesCoordinator:
    def __init__(self, window):
        self.window = window
        self.plan = None
        self.phase = 'idle'
        self.current = self.trials = None
        self.last_poll = 0.
        self.pending_start_response = None
        self.interrupted = False
        self.window.client.controlled.connect(self.response)

    @property
    def busy(self):
        return self.phase not in ('idle', 'between', 'complete')

    def create(self, config, participant, seed, info):
        if self.busy:
            raise ValueError('Сначала завершите текущую серию')
        self.plan = SeriesPlan.create(self.window.sessions_root, config, participant, seed, info)
        self.phase = 'between'
        self.show_plan()

    def show_plan(self):
        groups = self.plan.groups()
        summary = '; '.join(f'{i}: {len(g)} проб, {series_duration(self.plan.config, g)/60:.1f} мин'
                            for i, g in enumerate(groups, len(self.plan.data['series']) + 1))
        self.window.series_summary.setText(f'Выполнено {len(self.plan.data["completed_trial_ids"])}/{len(self.plan.trials)} · '
            f'осталось {len(self.plan.remaining)} проб · серий осталось: {len(groups)}\n{summary or "Всё выполнено"}')
        self.window.path_label.setText('План: ' + str(self.plan.path))

    def load(self, path):
        if self.busy:
            raise ValueError('Сначала завершите текущую серию')
        self.plan = SeriesPlan.load(path)
        self.window.participant_edit.setText(self.plan.data['participant_id'])
        self.window.seed_edit.setText(str(self.plan.data['random_seed']))
        self.phase = 'recovering' if self.plan.data['active_series_id'] else 'between'
        if self.phase == 'recovering':
            self.current = self.plan.data['series'][-1]
            self.interrupted = True
            if self.current.get('synthetic'):
                self._finish({}, interrupted=True)
        self.show_plan()

    def start(self):
        if not self.plan or self.busy:
            raise ValueError('Сначала создайте или загрузите план; дождитесь сохранения серии')
        if self.window.client.pending or self.window.client.control_pending:
            raise ValueError('Дождитесь подтверждений рекордера')
        offline = self.window.debug and self.window.offline.isChecked()
        self.current, self.trials = self.plan.begin()
        self.interrupted = False
        self.phase = 'starting'
        context = {**self.current, 'experiment_id': self.plan.data['experiment_id'],
                   'participant_id': self.plan.data['participant_id'], 'random_seed': self.plan.data['random_seed']}
        if offline:
            context.update(own_calibration_id=None, noitom_calibration_id=None, synthetic=True)
            self.current.update(synthetic=True)
            self.plan.save()
            self._launch(context, True)
        else:
            try:
                self.window.client.send_control('start_recording', context=context,
                                                previous_calibrations=self.plan.data['previous_calibrations'])
            except ValueError:
                self.plan.finish('not_started')
                self.phase = 'between'
                raise

    def _launch(self, context, offline):
        try:
            self.window._launch_series(self.plan, self.trials, context, offline)
            self.current.update(presenter_path=str(self.window.session_log.path),
                                presenter_session_id=self.window.session_log.context['session_id'], status='running')
            self.plan.save()
            self.phase = 'running'
        except (OSError, ValueError) as error:
            self.window.notice.setText('Ошибка запуска серии: ' + str(error))
            self.window._log_failed = True
            self.interrupted = True
            self.phase = 'recovering'
            if offline:
                self._finish({}, interrupted=True)

    def response(self, result):
        try:
            self._response(result)
        except (OSError, ValueError) as error:
            self.window.notice.setText('Не удалось сохранить состояние серии: ' + str(error))
            self.window._log_failed = True
            self.interrupted = True
            if self.current:
                self.phase = 'recovering'

    def _response(self, result):
        command = result['command']
        if not result.get('accepted'):
            self.window.notice.setText('Рекордер: ' + result.get('reason', 'команда отклонена'))
            if command == 'start_recording' and self.phase == 'starting':
                if result.get('reason') == 'control_timeout':
                    self.phase = 'recovering'
                    self.interrupted = True
                else:
                    self.plan.finish('not_started', reason=result.get('reason'))
                    self.phase = 'between'
            elif command == 'stop_recording' and self.phase in ('stopping', 'recovering'):
                self.phase = 'recovering'
                self.interrupted = True
            return
        if command == 'start_recording' and self.phase == 'starting':
            self.pending_start_response = result
            self.plan.attach_recording(result)
            self.phase = 'initializing'
        if command == 'stop_recording' and self.phase in ('stopping', 'recovering'):
            self.phase = 'saving'
        if command == 'status':
            if self.phase == 'initializing':
                if result.get('last_recording_error'):
                    self.window.notice.setText('Ошибка записи: ' + result['last_recording_error'])
                    self.phase = 'draining'
                elif result.get('recording_initialized') and result.get('recording_session_id') == self.plan.data['series'][-1].get('recording_session_id'):
                    self._launch(self.pending_start_response['series_context'], False)
                elif not result.get('ready'):
                    self.phase = 'recovering'
            elif self.phase == 'saving' and not result.get('saving'):
                self._finish(result)
            elif self.phase == 'recovering':
                self.interrupted = True
                rid = self.current.get('recording_session_id')
                remote = result.get('series_context') or {}
                if not rid and remote.get('series_id') == self.current['series_id']:
                    self.plan.attach_recording(result)
                    rid = result['recording_session_id']
                if rid and result.get('ready') and result.get('recording_session_id') == rid:
                    self.window.client.send_control('stop_recording', recording_session_id=rid)
                elif not result.get('saving'):
                    self._finish(result, interrupted=True)

    def _finish(self, result, interrupted=False):
        own_recording = result.get('last_recording_id') == self.plan.data['series'][-1].get('recording_session_id')
        status = result.get('last_recording_status', 'unknown') if own_recording else 'unknown'
        outcome = (self.window.session_log.metadata.get('outcome')
                   if self.window.session_log and not interrupted and not self.interrupted else 'interrupted')
        self.plan.finish('interrupted' if interrupted or self.interrupted else 'finished',
                         recording_status=status, outcome=outcome)
        self.phase = 'complete' if not self.plan.remaining else 'between'
        self.window.notice.setText(f'Серия сохранена · качество записи: {status}. ' +
            ('Эксперимент завершён.' if self.phase == 'complete' else 'Повторите калибровку IMU и Noitom перед следующей серией.'))
        self.window._finish_session_log()
        self.show_plan()

    def tick(self):
        w = self.window
        if self.phase == 'running' and w.controller and not w.controller.active:
            self.phase = 'draining'
        if self.phase == 'draining' and not w.client.pending:
            if w.controller and w.controller.offline:
                self.plan.finish('offline', outcome=w.session_log.metadata.get('outcome'), recording_status='not_recorded')
                self.phase = 'complete' if not self.plan.remaining else 'between'
                w._finish_session_log()
                self.show_plan()
            elif w.client.control_supported and not w.client.control_pending:
                rid = self.plan.data['series'][-1].get('recording_session_id')
                if not w.client.ready or w.client.recording_session_id != rid:
                    self.phase = 'recovering'
                else:
                    w.client.send_control('stop_recording', recording_session_id=rid)
                    self.phase = 'stopping'
        if (self.phase in ('between', 'initializing', 'saving', 'recovering') and w.client.control_supported
                and not w.client.control_pending and time.monotonic() - self.last_poll > .5):
            self.last_poll = time.monotonic()
            w.client.send_control('status')
