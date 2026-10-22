"""Noitom-only manual pose annotation for a Jupyter notebook.

ipywidgets/IPython are optional dependencies, imported only by the widget.
The JSON format and validation are shared with the Qt annotator.
"""
from copy import deepcopy
from html import escape
from pathlib import Path

import numpy as np

from scripts.ER2026.paths import OUTPUT_ROOT, RECORDS_ROOT
from .annotations import (LABELS, angular_speed, candidates, interval, read_document,
                          save_document, validate_intervals)
from .io import indexed_records
from .motion import load_bvh, require_pair


class NoitomAnnotator:
    def __init__(self, recording_id=None, records_root=RECORDS_ROOT, labels_root=OUTPUT_ROOT / 'annotations'):
        import ipywidgets as w

        self.records_root, self.labels_root = Path(records_root).resolve(), Path(labels_root).resolve()
        if self.labels_root == self.records_root or self.labels_root.is_relative_to(self.records_root):
            raise ValueError('Labels must be outside source records')
        self.rows = {r['recording_id']: r for r in indexed_records(self.records_root) if 'noitom' in r['paths']}
        if not self.rows:
            raise ValueError('No indexed Noitom recordings')
        if recording_id is not None and recording_id not in self.rows:
            raise ValueError(f'Unknown recording with Noitom: {recording_id}')
        self.document, self.motion, self.history = None, None, []
        self._updating = False
        options = [(f"{r['subject_id']} · {r['word_set_id']} · {rid}", rid) for rid, r in self.rows.items()]
        self.record = w.Dropdown(options=options, value=recording_id or options[0][1],
                                 description='Запись:', layout=w.Layout(width='100%'))
        self.frame = w.IntSlider(description='Кадр:', min=0, max=1, continuous_update=False,
                                 layout=w.Layout(width='75%'))
        self.play = w.Play(min=0, max=1, step=1, interval=200, description='Просмотр')
        self.link = w.jslink((self.play, 'value'), (self.frame, 'value'))
        self.start = w.BoundedFloatText(description='Начало, с:', min=0, max=1, step=.02, value=0.)
        self.end = w.BoundedFloatText(description='Конец, с:', min=0, max=1, step=.02, value=1.)
        self.label = w.Dropdown(options=LABELS, value='UNKNOWN', description='Метка:')
        self.note = w.Text(description='Заметка:', layout=w.Layout(width='100%'))
        self.saved = w.Dropdown(options=[('Новый интервал', '')], description='Изменить:',
                                layout=w.Layout(width='100%'))
        self.status, self.table = w.HTML(), w.HTML()
        self.output = w.Output()
        self.buttons = {}
        for name, text in (('start', 'Начало = текущий кадр'), ('end', 'Конец = текущий кадр'),
                           ('next', 'Следующая устойчивая поза'), ('new', 'Новый интервал'),
                           ('save', 'Подтвердить и сохранить'), ('delete', 'Удалить выбранный'), ('undo', 'Отменить')):
            self.buttons[name] = w.Button(description=text, layout=w.Layout(width='auto'))
        self.buttons['save'].button_style = 'success'
        self.buttons['start'].on_click(lambda _: self._mark('start'))
        self.buttons['end'].on_click(lambda _: self._mark('end'))
        self.buttons['next'].on_click(lambda _: self.next_candidate())
        self.buttons['new'].on_click(lambda _: self.new_interval())
        self.buttons['save'].on_click(lambda _: self.save())
        self.buttons['delete'].on_click(lambda _: self.delete())
        self.buttons['undo'].on_click(lambda _: self.undo())
        self.record.observe(self._on_record, names='value')
        self.frame.observe(self._draw, names='value')
        self.start.observe(self._draw, names='value')
        self.end.observe(self._draw, names='value')
        self.saved.observe(self._select, names='value')
        self.widget = w.VBox([
            w.HTML('<b>Разметка поз по Noitom</b>. На проекциях показан только Noitom. '
                   'Границы [начало, конец), секунды от кадра 0. Метка появляется только после подтверждения.'),
            self.record, self.status, w.HBox([self.play, self.frame]), self.output,
            w.HBox([self.start, self.end, self.label]), self.note,
            w.HBox([self.buttons[n] for n in ('start', 'end', 'next')]), self.saved,
            w.HBox([self.buttons[n] for n in ('new', 'save', 'delete', 'undo')]), self.table,
            w.HTML('Е и Э имеют одну эталонную позу. TRANSITION / BAD / UNKNOWN исключаются из обучения. '
                   'NEUTRAL — отдельный класс. При сомнении оставьте UNKNOWN и комментарий.')])
        if recording_id is None:
            for rid in self.rows:
                if self.load(rid):
                    break
        else:
            self.load(recording_id)

    def _message(self, text, error=False):
        self.status.value = f'<span style="color:{"#b42318" if error else "#175c43"}">{escape(str(text))}</span>'

    def load(self, recording_id):
        self._updating = True
        self.play.playing = False
        try:
            row = self.rows[recording_id]
            # Retain both source hashes for compatibility with the Qt annotator.
            motions = {s: load_bvh(p) for s, p in row['paths'].items()}
            if len(motions) == 2:
                require_pair(motions['custom'], motions['noitom'])
            motion = motions['noitom']
            path = self.labels_root / (recording_id + '.json')
            document = read_document(path, row, motions)
            self.motion, self.document, self.path = motion, document, path
            self.positions, self.rotations = motion.body_positions(), motion.body_rotations()
            self.speed, self.proposals = angular_speed(motion), candidates(motion)
            self.history = []
            self.record.value = recording_id
            self.play.value = self.frame.value = 0
            self.play.max = self.frame.max = motion.count - 1
            self.play.step = max(1, round(.2 / motion.dt))
            for control in (self.start, self.end):
                control.value = 0
                control.max = motion.duration
                control.step = motion.dt
            self.end.value = min(1., motion.duration)
            self._refresh_table()
            self._message(f'{motion.count} кадров · {1 / motion.dt:.3f} Гц · {motion.duration:.2f} с · {path}')
        except (OSError, ValueError, IndexError) as error:
            self.motion, self.document = None, None
            self.table.value = ''
            self._message(f'Запись недоступна: {error}. Выберите другую запись.', True)
            with self.output:
                self.output.clear_output(wait=False)
            return False
        finally:
            self._updating = False
        self._draw()
        return True

    def _on_record(self, change):
        if not self._updating:
            self.load(change['new'])

    def _draw(self, _change=None):
        if self._updating or self.motion is None:
            return
        import matplotlib.pyplot as plt
        from IPython.display import display

        motion, frame = self.motion, self.frame.value
        stamp = frame * motion.dt
        fig = plt.figure(figsize=(11, 5), layout='constrained')
        grid = fig.add_gridspec(2, 2, height_ratios=[2, 1])
        for col, title in enumerate(('Спереди: левая рука участника справа', 'Сбоку: глубина / высота')):
            ax = fig.add_subplot(grid[0, col])
            for i, name in enumerate(motion.names):
                parent = motion.parents[i]
                if parent is not None and name in ('Chest', 'Head', 'LeftArm', 'LeftForeArm', 'RightArm', 'RightForeArm'):
                    points = self.positions[frame, [parent, i]]
                    ax.plot(-points[:, 0] if col == 0 else points[:, 2], points[:, 1], 'o-', color='#0072B2')
                if name in ('LeftForeArm', 'RightForeArm'):
                    elbow = self.positions[frame, i]
                    wrist = elbow + self.rotations[frame, i] @ motion.ends[i]
                    points = np.stack((elbow, wrist))
                    ax.plot(-points[:, 0] if col == 0 else points[:, 2], points[:, 1], 'o-', color='#0072B2')
            ax.set(title=title, xlim=(-1.4, 1.4), ylim=(-.5, 1.3), xlabel='м', ylabel='Высота, м')
            ax.set_aspect('equal')
            ax.grid(alpha=.25)
        ax = fig.add_subplot(grid[1, :])
        ax.plot(np.arange(motion.count) * motion.dt, self.speed, color='#0072B2', lw=.6)
        for item in self.document['intervals']:
            ax.axvspan(item['start_s'], item['end_s'], color='#009E73', alpha=.12)
        ax.axvspan(self.start.value, self.end.value, color='#E69F00', alpha=.35)
        ax.axvline(stamp, color='#CC3311', lw=1)
        ax.set(xlim=(0, motion.duration), xlabel='Время BVH, с', ylabel='Скорость, °/с',
               title=f'Noitom · {stamp:.3f} с · кадр {frame} · жёлтый: текущий интервал, зелёный: сохранённые')
        ax.grid(alpha=.2)
        with self.output:
            self.output.clear_output(wait=True)
            display(fig)
        plt.close(fig)

    def _mark(self, name):
        if self.motion is not None:
            getattr(self, name).value = self.frame.value * self.motion.dt

    def _refresh_table(self):
        self._updating = True
        options = [('Новый интервал', '')]
        rows = []
        for item in self.document['intervals']:
            a, b = item['start_s'], item['end_s']
            options.append((f'{a:.3f}–{b:.3f} с · {item["label"]}', item['id']))
            rows.append('<tr>' + ''.join(f'<td>{escape(str(v))}</td>' for v in
                        (f'{a:.3f}', f'{b:.3f}', item['label'], item.get('annotator_note', ''))) + '</tr>')
        self.saved.options = options
        self.saved.value = ''
        self.table.value = ('<table><tr><th>Начало, с</th><th>Конец, с</th><th>Метка</th><th>Комментарий</th></tr>' +
                            ''.join(rows) + '</table>')
        self._updating = False

    def _select(self, change):
        if self._updating or not change['new'] or self.document is None:
            return
        item = next(i for i in self.document['intervals'] if i['id'] == change['new'])
        self._updating = True
        self.start.value, self.end.value = item['start_s'], item['end_s']
        self.label.value, self.note.value = item['label'], item.get('annotator_note', '')
        self.frame.value = round(item['start_s'] / self.motion.dt)
        self._updating = False
        self._draw()

    def new_interval(self):
        self.saved.value = ''
        self.note.value = ''

    def _persist(self, items):
        validate_intervals(items, self.motion.duration)
        document = deepcopy(self.document)
        document['intervals'] = sorted(items, key=lambda i: i['start_s'])
        save_document(self.path, document, self.motion.duration)
        self.history.append(deepcopy(self.document['intervals']))
        self.document = document
        self._refresh_table()
        self._message(f'Сохранено {len(items)} интервалов: {self.path}')
        self._draw()

    def save(self):
        if self.document is None:
            return False
        try:
            item = interval(self.start.value, self.end.value, self.label.value, self.note.value)
            item['annotation_source'] = 'noitom'
            item['annotation_tool'] = 'notebook'
            if self.saved.value:
                item['id'] = self.saved.value
            items = [i for i in self.document['intervals'] if i['id'] != self.saved.value] + [item]
            self._persist(items)
            self._updating = True
            self.saved.value = item['id']
            self._updating = False
            return True
        except (ValueError, OSError) as error:
            self._message(error, True)
            return False

    def delete(self):
        if self.document is None or not self.saved.value:
            return False
        try:
            self._persist([i for i in self.document['intervals'] if i['id'] != self.saved.value])
            return True
        except (ValueError, OSError) as error:
            self._message(error, True)
            return False

    def undo(self):
        if not self.history:
            return False
        try:
            document = deepcopy(self.document)
            document['intervals'] = self.history[-1]
            save_document(self.path, document, self.motion.duration)
            self.history.pop()
            self.document = document
            self._refresh_table()
            self._message('Предыдущее состояние разметки восстановлено и сохранено')
            self._draw()
            return True
        except (ValueError, OSError) as error:
            self._message(error, True)
            return False

    def next_candidate(self):
        if self.motion is None:
            return
        current = self.frame.value * self.motion.dt
        for a, b in self.proposals:
            if a > current + self.motion.dt and not any(a < i['end_s'] and b > i['start_s'] for i in self.document['intervals']):
                self.new_interval()
                self._updating = True
                self.start.value, self.end.value = a, b
                self.frame.value = round(a / self.motion.dt)
                self._updating = False
                self._draw()
                return
        self._message('Дальше нет свободных устойчивых участков. Задайте границы вручную.')

    def display(self):
        from IPython.display import display
        display(self.widget)
        return self
