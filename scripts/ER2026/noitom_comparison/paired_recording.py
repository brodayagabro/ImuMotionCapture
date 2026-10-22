"""Timed acquisition using the existing receiver claims and PC monotonic clock."""
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import json
import math
import threading
import time
import uuid

import numpy as np
from .paired_bvh import SEGMENTS, export_pair, identifier, normalized
from .noitom_adapter import NoitomAdapter


class PairSession:
    def __init__(self, root, metadata, now_ns, *, wall_time, max_gap_ns, max_fps):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.subject_id = identifier(metadata['subject_id'])
        self.word_set_id = identifier(metadata['word_set_id'])
        self.duration_s = float(metadata.get('duration_s', 300.))
        if not math.isfinite(self.duration_s) or not 1 <= self.duration_s <= 3600:
            raise ValueError('Продолжительность записи должна быть от 1 до 3600 секунд')
        self.arm_ns = self.t0_ns = now_ns
        self.last_heartbeat_ns = now_ns
        self.arm_wall = wall_time
        self.timestamp = None
        self.session_id = uuid.uuid4().hex
        self.path = self.root
        self.max_gap_ns, self.max_fps = max_gap_ns, max_fps
        self.stop_ns = self.deadline_ns = None
        self.pending = 0
        self.ready, self.done = threading.Event(), threading.Event()
        self.error = None
        self.own, self.noitom = [], []
        sources = metadata.get('recording_sources', ['custom', 'noitom'])
        if (not isinstance(sources, (list, tuple)) or not sources
                or any(name not in ('custom', 'noitom') for name in sources)
                or len(set(sources)) != len(sources)):
            raise ValueError('Выберите хотя бы один костюм: custom или noitom')
        self.sources = tuple(name for name in ('custom', 'noitom') if name in sources)
        self.hierarchy = None
        self.avatar = self.heading = self.last_own_key = self.last_noitom_key = None
        self.rejected = {'custom': 0, 'noitom': 0}
        self.coalesced_timestamps = {'custom': 0, 'noitom': 0}
        self.metadata = {**metadata, 'recording_sources': list(self.sources),
                         'status': 'arming', 'session_id': self.session_id}
        self.closing = self.exporting = False
        self.closing_started_ns = None
        self.export_progress = {'percent': 0, 'phase': 'Ожидание окончания записи'}
        self.journal('arming')

    def update_export_progress(self, percent, phase):
        # One immutable snapshot: GUI/TCP readers never wait for the exporter.
        self.export_progress = {'percent': max(self.export_progress['percent'], min(100, int(percent))), 'phase': phase}

    def journal(self, status, **extra):
        row = dict(session_id=self.session_id, timestamp=self.timestamp, status=status,
                   subject_id=self.subject_id, word_set_id=self.word_set_id,
                   path=str(self.path), error=self.error, **extra)
        row['recording_sources'] = list(self.sources)
        with (self.root / 'application.log').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')

    def initialize(self):
        streams = self.streams()
        if self.ready.is_set() or any(not stream for _, stream in streams):
            return
        self.t0_ns = max(stream[0][0] for _, stream in streams)
        for _, stream in streams:
            while len(stream) > 1 and stream[1][0] <= self.t0_ns:
                del stream[0]
            if self.t0_ns - stream[0][0] > self.max_gap_ns:
                raise ValueError('Нет непрерывных данных для общего T0')
        wall = self.arm_wall + timedelta(microseconds=(self.t0_ns - self.arm_ns) / 1000)
        self.timestamp = wall.isoformat()
        name = wall.strftime('%Y%m%d_%H%M%S_%f') + f'_{self.subject_id}_{self.word_set_id}_{self.session_id[:6]}'
        self.path = self.root / name
        self.path.mkdir(exist_ok=False)
        self.deadline_ns = self.t0_ns + round(self.duration_s * 1e9)
        self.metadata.update(status='recording', timestamp=self.timestamp, recording_t0_ns=self.t0_ns)
        self.journal('recording', t0_ns=self.t0_ns, t1_ns=self.deadline_ns,
                     duration_s=self.duration_s, calibration=self.metadata.get('own_system'),
                     noitom=self.metadata.get('noitom'))
        self.ready.set()

    def streams(self):
        return [(name, self.own if name == 'custom' else self.noitom) for name in self.sources]

    def finish(self):
        try:
            if not self.error:
                self.metadata.update(export_pair(self, self.update_export_progress))
                status = 'complete'
            else:
                status = 'incomplete'
            self.journal(status, rejected_frames=self.rejected,
                         coalesced_timestamps=self.coalesced_timestamps,
                         export={k: self.metadata[k] for k in ('frames', 'fps', 'frame_time', 'source_rates_hz',
                             'max_source_gaps_ns', 't0_ns', 't1_ns', 'skeleton', 'skeleton_height_m',
                             'joints', 'channels', 'root_translation', 'tracked_segments', 'export_fps_limit') if k in self.metadata})
            if not self.error:
                self.update_export_progress(100, 'BVH сохранены, строка добавлена в таблицу')
            else:
                self.update_export_progress(self.export_progress['percent'], 'Запись неполная')
            self.metadata['status'] = status
        except Exception as error:
            self.error = f'{type(error).__name__}: {error}'
            self.metadata.update(status='error', error=self.error)
            self.update_export_progress(self.export_progress['percent'], 'Ошибка сохранения: ' + self.error)
            try:
                self.journal('error')
            except OSError:
                pass
        finally:
            self.own.clear()
            self.noitom.clear()
            self.done.set()


@dataclass(frozen=True)
class PairClaim:
    session: PairSession
    timestamp_ns: int


class PairedRecorder:
    """Drop-in acquisition boundary; legacy JSONL recording remains a separate API."""
    def __init__(self, *, clock=time.perf_counter_ns, watchdog=True, max_gap_ms=500):
        # On Python <= 3.12 / Windows, monotonic_ns may use GetTickCount64
        # (15.625 ms ticks). QPC is monotonic and shared across processes;
        # reception, calibration, freshness checks and presenter use this epoch.
        self.clock, self.watchdog = clock, watchdog
        self.max_gap_ns = round(max_gap_ms * 1e6)
        self._lock = threading.RLock()
        self.active = None
        self.sessions = []
        self.noitom_adapter = NoitomAdapter()

    def start(self, root, metadata, *, t0_ns=None):
        with self._lock:
            if self.active or any(not s.done.is_set() for s in self.sessions):
                raise ValueError('Запись или экспорт ещё не завершены')
            now = self.clock() if t0_ns is None else t0_ns
            export_fps = float(metadata.get('export_fps', 50.))
            if not math.isfinite(export_fps) or not 1 <= export_fps <= 100:
                raise ValueError('Частота BVH должна быть от 1 до 100 Гц')
            wall_time = datetime.now().astimezone()
            s = PairSession(root, metadata, now, wall_time=wall_time, max_gap_ns=self.max_gap_ns,
                            max_fps=min(export_fps, float(metadata.get('max_fps', 100.))))
            self.active = s
            self.sessions.append(s)
            if self.watchdog:
                threading.Thread(target=self._watch, args=(s,), name='paired-deadline', daemon=True).start()
            return s

    def _watch(self, session):
        while not session.done.wait(.02):
            self.tick()

    def stamp_noitom_packet(self):
        return self.stamp_packet('noitom')

    def stamp_packet(self, source='custom'):
        # Unselected receivers keep running, without holding dataset claims.
        with self._lock:
            now, s = self.clock(), self.active
            if s is None or s.closing or source not in s.sources:
                return now, None
            s.pending += 1
            return now, PairClaim(s, now)

    def _append(self, s, source, timestamp, q, position):
        stream = s.own if source == 'custom' else s.noitom
        if timestamp < s.arm_ns or s.closing:
            return
        if stream and timestamp < stream[-1][0]:
            raise ValueError(f'{source}: время получения кадра уменьшилось на {stream[-1][0] - timestamp} нс')
        # Keep only the first bracketing sample after the fixed deadline.
        if s.deadline_ns is not None and stream and stream[-1][0] >= s.deadline_ns:
            return
        if stream and timestamp == stream[-1][0]:
            # Equal clock readings are not reversed packets. Keep the latest
            # valid sample at this instant, without inventing time or FPS.
            stream[-1] = (timestamp, q, np.asarray(position, float))
            s.coalesced_timestamps[source] += 1
            return
        stream.append((timestamp, q, np.asarray(position, float)))
        s.initialize()

    def complete_packet(self, claim, packet, frame=None):
        if claim is None:
            return
        with self._lock:
            s = claim.session
            try:
                if 'custom' in s.sources and not s.closing and not (s.deadline_ns is not None and s.own and s.own[-1][0] >= s.deadline_ns):
                    if frame is None or not frame.get('measurement_complete'):
                        s.rejected['custom'] += 1
                    else:
                        key = tuple(frame['measurement_key'])
                        if key == s.last_own_key:
                            s.rejected['custom'] += 1
                        else:
                            if s.last_own_key is not None and (key[0] - s.last_own_key[0]) % (2 ** 32) >= 2 ** 31:
                                raise ValueError('ESP32: счётчик кадров сбросился или кадры перепутаны')
                            s.last_own_key = key
                            q = normalized([frame['bvh_orientations'][name] for name in SEGMENTS])
                            self._append(s, 'custom', claim.timestamp_ns, q, np.zeros(3))
            except (ValueError, KeyError, TypeError, OSError) as error:
                self._close(s, str(error))
            finally:
                s.pending -= 1
                self._finish_if_ready(s)

    def noitom_frame(self, frame, orientations, mapping_error=None, diagnostics=None, *,
                     enabled_segments=None, heading_offset_rad=0.):
        claim = frame.capture_context
        if claim is None:
            return
        if 'noitom' not in claim.session.sources:
            self.release_claim(claim)
            return
        # Retarget all five measured segments independently of display checkboxes.
        # Do this before taking the acquisition lock so ESP32 stamping cannot wait
        # behind Noitom matrix calculations.
        capture_error = None
        try:
            mapped = self.noitom_adapter.orientations(frame, SEGMENTS, heading_offset_rad=heading_offset_rad)
            q = normalized([mapped[name] for name in SEGMENTS])
        except (ValueError, KeyError, TypeError) as error:
            capture_error = str(error)
        with self._lock:
            s = claim.session
            try:
                if s.closing or (s.deadline_ns is not None and s.noitom and s.noitom[-1][0] >= s.deadline_ns):
                    return
                key = (frame.frame_index, frame.sdk_timestamp)
                if key != (None, None) and key == s.last_noitom_key:
                    s.rejected['noitom'] += 1
                    return
                if (s.last_noitom_key is not None and frame.frame_index is not None and
                        s.last_noitom_key[0] is not None and
                        (frame.frame_index - s.last_noitom_key[0]) % (2 ** 32) >= 2 ** 31):
                    raise ValueError('Noitom: счётчик кадров сбросился или кадры перепутаны')
                s.last_noitom_key = key
                if mapping_error:
                    raise ValueError('Noitom: ' + mapping_error)
                if capture_error:
                    raise ValueError('Noitom: ' + capture_error)
                hierarchy = tuple(sorted((name, joint.parent) for name, joint in frame.joints.items()))
                if s.hierarchy is None:
                    s.hierarchy = hierarchy
                    s.avatar = (frame.avatar_index, frame.avatar_name)
                    s.heading = heading_offset_rad
                if (s.avatar != (frame.avatar_index, frame.avatar_name) or s.heading != heading_offset_rad or
                        hierarchy != s.hierarchy):
                    raise ValueError('Noitom: во время записи изменился аватар, скелет или направление')
                self._append(s, 'noitom', frame.host_timestamp_ns, q, np.zeros(3))
            except (ValueError, KeyError, TypeError, OSError) as error:
                self._close(s, str(error))
            finally:
                s.pending -= 1
                self._finish_if_ready(s)

    def release_claim(self, claim, error=None):
        if claim is not None:
            with self._lock:
                if error:
                    self._close(claim.session, str(error))
                claim.session.pending -= 1
                self._finish_if_ready(claim.session)

    def event(self, name, details=None, *, incomplete=False, timestamp_ns=None):
        # Stimulus markers are deliberately absent from the new dataset.
        if incomplete:
            with self._lock:
                self.tick()
                if self.active:
                    source = 'custom' if name.startswith('own_') else 'noitom' if name.startswith('noitom_') else None
                    if source is not None and source not in self.active.sources:
                        return
                    self._close(self.active, name + (': ' + str(details) if details else ''))

    def tick(self):
        with self._lock:
            s, now = self.active, self.clock()
            for closing in self.sessions:
                if closing.closing and not closing.exporting and now - closing.closing_started_ns > 5_000_000_000:
                    closing.error = 'Не завершена обработка полученных пакетов'
                    self._finish_if_ready(closing, force=True)
            if s is None:
                return
            if s.metadata.get('require_heartbeat') and now - s.last_heartbeat_ns > 5_000_000_000:
                self._close(s, 'Потеряна связь с демонстратором')
                return
            if not s.ready.is_set():
                if now - s.arm_ns > 3_000_000_000:
                    self._close(s, 'За 3 секунды не получены валидные кадры выбранных костюмов')
                return
            streams = s.streams()
            if now >= s.deadline_ns and all(stream[-1][0] >= s.deadline_ns for _, stream in streams):
                self._close(s)
            elif now > s.deadline_ns + self.max_gap_ns:
                self._close(s, 'Нет кадров для конца общего интервала T1')
            elif now < s.deadline_ns and any(now - stream[-1][0] > self.max_gap_ns for _, stream in streams):
                gaps = ', '.join(f'{name}: {(now - stream[-1][0]) / 1e6:.0f} мс'
                    for name, stream in streams
                    if now - stream[-1][0] > self.max_gap_ns)
                self._close(s, 'Длительный разрыв потока (' + gaps + ')')

    def _close(self, s, error=None):
        if s.closing:
            return
        s.error = error
        s.stop_ns = s.deadline_ns if error is None else self.clock()
        s.closing = True
        s.closing_started_ns = self.clock()
        s.metadata['status'] = 'saving'
        s.update_export_progress(0, 'Ожидание обработки принятых кадров')
        if self.active is s:
            self.active = None
        self._finish_if_ready(s)

    def _finish_if_ready(self, s, *, force=False):
        if s.closing and (s.pending == 0 or force) and not s.exporting:
            s.exporting = True
            threading.Thread(target=s.finish, name='paired-bvh-export', daemon=False).start()

    def stop(self, *, stop_ns=None):
        with self._lock:
            self.tick()
            s = self.active
            if s:
                self._close(s, 'Досрочная остановка оператором')
            return s

    def recording_state(self):
        with self._lock:
            s = self.active
            return {'ready': bool(s and s.ready.is_set()),
                    'recording_session_id': s.session_id if s else None,
                    'recording_t0_ns': s.t0_ns if s and s.ready.is_set() else None,
                    'recording_t1_ns': s.deadline_ns if s else None,
                    'arming': bool(s and not s.ready.is_set())}

    def heartbeat(self, session_id):
        with self._lock:
            if self.active and self.active.session_id == session_id:
                self.active.last_heartbeat_ns = self.clock()

    def external_event(self, payload, received_ns):
        return {'accepted': False, 'reason': 'stimulus_markers_disabled'}
