"""Persistent full experiment plan; series reuse the existing trial controller."""
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

from .config import Trial, config_from_snapshot, generate_trials, plan_series, series_duration, validate_participant


class SeriesPlan:
    def __init__(self, path, data):
        self.path, self.data = Path(path), data
        self.config = config_from_snapshot(data['config'])
        self.trials = tuple(Trial(**row) for row in data['trials'])

    @classmethod
    def create(cls, root, config, participant, seed, participant_info):
        validate_participant(participant)
        eid = uuid.uuid4().hex
        trials = generate_trials(config, seed)
        plan = cls(Path(root) / 'experiments' / eid / 'plan.json', {
            'schema_version': 1, 'experiment_id': eid, 'participant_id': participant,
            'participant_info': participant_info, 'random_seed': seed,
            'created_at': datetime.now(timezone.utc).isoformat(), 'config': asdict(config),
            'trials': [asdict(t) for t in trials], 'completed_trial_ids': [], 'next_attempts': {},
            'series': [], 'active_series_id': None, 'previous_calibrations': {}})
        plan.save()
        return plan

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding='utf-8-sig'))
        if not isinstance(data, dict) or data.get('schema_version') != 1:
            raise ValueError('Неизвестный формат плана')
        try:
            for key in ('experiment_id', 'participant_id', 'participant_info', 'random_seed',
                        'completed_trial_ids', 'next_attempts', 'series', 'active_series_id', 'previous_calibrations'):
                data[key]
            validate_participant(data['participant_id'])
            result = cls(path, data)
            ids = [t.trial_id for t in result.trials]
            if not ids or len(ids) != len(set(ids)) or not set(data['completed_trial_ids']) <= set(ids):
                raise ValueError('Некорректные номера проб')
            if (not isinstance(data['series'], list) or not isinstance(data['next_attempts'], dict)
                    or not isinstance(data['previous_calibrations'], dict)):
                raise ValueError('Некорректные состояния серий')
            if data['active_series_id'] and (not data['series'] or data['series'][-1]['series_id'] != data['active_series_id']):
                raise ValueError('Не найдена незавершённая серия')
            return result
        except (KeyError, TypeError, AttributeError) as error:
            raise ValueError('Повреждённый или неполный план эксперимента') from error

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix('.tmp')
        temp.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding='utf-8')
        temp.replace(self.path)

    @property
    def remaining(self):
        done = set(self.data['completed_trial_ids'])
        return tuple(t for t in self.trials if t.trial_id not in done)

    def groups(self):
        return plan_series(self.config, self.remaining)

    def begin(self):
        if self.data['active_series_id']:
            raise ValueError('Предыдущая серия ещё не завершена')
        groups = self.groups()
        if not groups:
            raise ValueError('Все пробы выполнены')
        series = {'series_id': uuid.uuid4().hex, 'series_index': len(self.data['series']) + 1,
                  'trial_ids': [t.trial_id for t in groups[0]],
                  'estimated_seconds': series_duration(self.config, groups[0]), 'status': 'starting'}
        series['series_prepare_s'] = self.config.series_prepare
        series['started_at'] = datetime.now(timezone.utc).isoformat()
        self.data['series'].append(series)
        self.data['active_series_id'] = series['series_id']
        self.save()
        return series, groups[0]

    def mark_event(self, payload):
        trial = payload.get('trial_id')
        if payload['type'] == 'trial_prepare' and trial is not None:
            # Reserve the next attempt before presenting; crash recovery cannot
            # accidentally reuse an attempt identity for an interrupted word.
            self.data['next_attempts'][str(trial)] = payload['attempt'] + 1
        elif payload['type'] == 'trial_end' and payload.get('outcome') in ('completed', 'skipped'):
            if trial not in self.data['completed_trial_ids']:
                self.data['completed_trial_ids'].append(trial)
        else:
            return
        self.save()

    def attach_recording(self, response):
        series = self.data['series'][-1]
        context = response['series_context']
        series.update(context, recording_session_id=response['recording_session_id'],
                      recording_path=response.get('recording_path'), status='recording')
        self.data['previous_calibrations'] = {k: context[k] for k in ('own_calibration_id', 'noitom_calibration_id')}
        self.save()

    def finish(self, status, **details):
        if self.data['active_series_id']:
            series = self.data['series'][-1]
            series.update(status=status, ended_at=datetime.now(timezone.utc).isoformat(),
                          completed_trial_ids=[t for t in series['trial_ids'] if t in self.data['completed_trial_ids']],
                          remaining_trial_ids=[t for t in series['trial_ids'] if t not in self.data['completed_trial_ids']], **details)
        self.data['active_series_id'] = None
        self.save()
