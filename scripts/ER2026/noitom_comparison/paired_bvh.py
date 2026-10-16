"""One host-time grid, identical skeletons, and a commit-last CSV index.

No packet-number synchronization, extrapolation, or hold-last resampling.
"""
import csv
import math
import os
from pathlib import Path
import re
import threading
import time

import numpy as np

from .bvh_skeleton import SEGMENTS, HEIGHT_M, SKELETON_ID, create_comparison_recording

INDEX_FIELDS = ('timestamp', 'subject_id', 'word_set_id', 'custom_bvh_path', 'noitom_bvh_path')
_index_lock = threading.Lock()


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,48}', value):
        raise ValueError('Код должен содержать 1–48 символов: латиница, цифры, _ или -')
    return value


def normalized(values):
    q = np.asarray(values, dtype=float)
    if q.ndim != 2 or q.shape[1] != 4 or not np.isfinite(q).all():
        raise ValueError('Некорректные кватернионы')
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(abs(norm - 1) > .1):
        raise ValueError('Некорректная норма кватерниона')
    return q / norm


def slerp(a, b, fraction):
    """Vectorized shortest-arc SLERP; normalized limit for tiny angles."""
    dot = np.sum(a * b, axis=-1, keepdims=True)
    b = np.where(dot < 0, -b, b)
    dot = np.clip(abs(dot), 0, 1)
    theta = np.arccos(dot)
    sine = np.sin(theta)
    safe = np.where(sine < 1e-6, 1, sine)
    q = np.where(sine < 1e-6, (1 - fraction) * a + fraction * b,
                 np.sin((1 - fraction) * theta) / safe * a + np.sin(fraction * theta) / safe * b)
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def common_grid(streams, t0_ns, t1_ns, max_gap_ns, max_fps=100.):
    rates, gaps = [], []
    duration = (t1_ns - t0_ns) / 1e9
    for samples in streams:
        times = np.array([s[0] for s in samples], dtype=np.int64)
        if len(times) < 2 or times[0] > t0_ns or times[-1] < t1_ns:
            raise ValueError('Поток не покрывает общий интервал T0–T1')
        steps = np.diff(times)
        if np.any(steps <= 0) or np.max(steps) > max_gap_ns:
            raise ValueError('Длительный разрыв или неверный порядок кадров')
        # Limit by both sustained throughput and median cadence, never by GUI FPS.
        rates.append(min((len(times) - 1) * 1e9 / int(times[-1] - times[0]), 1e9 / float(np.median(steps))))
        gaps.append(int(np.max(steps)))
    intervals = math.floor(duration * min(max_fps, *rates) + 1e-7)
    if intervals < 1 or intervals / duration < 1:
        raise ValueError('Недостаточная частота данных для BVH (минимум 1 Гц)')
    grid = t0_ns + np.rint(np.arange(intervals + 1) * ((t1_ns - t0_ns) / intervals)).astype(np.int64)
    return grid, intervals / duration, rates, gaps


def validate_bvh(path, count, fps, joints, progress=None):
    with Path(path).open(encoding='ascii') as stream:
        hierarchy = []
        for line in stream:
            if line.strip() == 'MOTION':
                break
            hierarchy.append(line)
        if stream.readline().strip() != f'Frames: {count}':
            raise ValueError('BVH: неверное количество кадров')
        dt = float(stream.readline().split(':', 1)[1])
        if abs(dt - 1 / fps) > 1e-12:
            raise ValueError('BVH: неверный Frame Time')
        width = 3 + 3 * len(joints)
        actual = 0
        for line in stream:
            row = np.fromstring(line, sep=' ')
            if len(row) != width or not np.isfinite(row).all():
                raise ValueError('BVH: повреждённые данные движения')
            actual += 1
            if progress is not None and (actual % 512 == 0 or actual == count):
                progress(actual / count)
        if actual != count or sum('CHANNELS' in s for s in hierarchy) != len(joints):
            raise ValueError('BVH: неполный файл')


def append_index(root, row):
    """Serialize append across threads/processes; flush only after pair validation."""
    root = Path(root)
    with _index_lock, (root / 'records.csv').open('a+', newline='', encoding='utf-8') as stream:
        if os.name == 'nt':
            import msvcrt
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            stream.seek(0)
            reader = csv.DictReader(stream)
            if reader.fieldnames is not None and reader.fieldnames != list(INDEX_FIELDS):
                raise ValueError('records.csv имеет другую структуру; файл не изменён')
            for existing in reader:
                if existing['custom_bvh_path'] == row['custom_bvh_path']:
                    if existing != row:
                        raise ValueError('Конфликт записи в records.csv')
                    return
            empty = reader.fieldnames is None
            stream.seek(0, os.SEEK_END)
            writer = csv.DictWriter(stream, INDEX_FIELDS)
            if empty:
                writer.writeheader()
            writer.writerow(row)
            stream.flush()
            os.fsync(stream.fileno())
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def quaternion_matrices(q):
    w, x, y, z = np.moveaxis(q, -1, 0)
    result = np.empty(q.shape[:-1] + (3, 3))
    result[..., 0, 0] = 1 - 2 * (y*y + z*z)
    result[..., 0, 1] = 2 * (x*y - w*z)
    result[..., 0, 2] = 2 * (x*z + w*y)
    result[..., 1, 0] = 2 * (x*y + w*z)
    result[..., 1, 1] = 1 - 2 * (x*x + z*z)
    result[..., 1, 2] = 2 * (y*z - w*x)
    result[..., 2, 0] = 2 * (x*z - w*y)
    result[..., 2, 1] = 2 * (y*z + w*x)
    result[..., 2, 2] = 1 - 2 * (x*x + y*y)
    return result


def export_pair(session, progress=None):
    def report(percent, phase):
        if progress is not None:
            progress(int(percent), phase)
        # Bound continuous Python work in the one background export thread.
        # Matrix conversion/interpolation is vectorized and releases the GIL.
        time.sleep(0)

    report(1, 'Проверка общей временной шкалы')
    grid, fps, rates, gaps = common_grid((session.own, session.noitom), session.t0_ns,
        session.stop_ns, session.max_gap_ns, session.max_fps)
    recordings = []
    for number, (name, source) in enumerate((('custom', session.own), ('noitom', session.noitom))):
        recording = create_comparison_recording(fps)
        recordings.append(recording)
        base = 2 + number * 40
        times = np.array([s[0] for s in source], dtype=np.int64)
        quaternions = np.asarray([s[1] for s in source])
        if quaternions.shape != (len(times), len(SEGMENTS), 4):
            raise ValueError(f'{name}: отсутствуют пять измеряемых сегментов')
        indices = np.searchsorted(times, grid, side='left')
        for start in range(0, len(grid), 256):
            stop = min(start + 256, len(grid))
            right = np.maximum(indices[start:stop], 1)
            left = right - 1
            fraction = (grid[start:stop] - times[left]) / (times[right] - times[left])
            q = slerp(quaternions[left], quaternions[right], fraction[:, None, None])
            matrices = quaternion_matrices(q)
            recording.append_batch((grid[start:stop] - session.t0_ns) / 1e9,
                {segment: matrices[:, index] for index, segment in enumerate(SEGMENTS)})
            report(base + 25 * stop / len(grid), f'Подготовка {name}.bvh')
        partial = session.path / (name + '.bvh.partial')
        recording.save(partial, uniform_grid=True,
            progress=lambda fraction, base=base, name=name: report(base + 25 + 15 * fraction, f'Запись {name}.bvh'))
    for index, (name, recording) in enumerate(zip(('custom', 'noitom'), recordings)):
        validate_bvh(session.path / (name + '.bvh.partial'), len(grid), fps, recording.joints,
            progress=lambda fraction, index=index, name=name: report(82 + index * 6 + 6 * fraction, f'Проверка {name}.bvh'))
    if recordings[0].joints != recordings[1].joints:
        raise ValueError('Скелеты BVH не совпадают')
    report(95, 'Завершение сохранения двух файлов')
    for name in ('custom', 'noitom'):
        source, target = session.path / (name + '.bvh.partial'), session.path / (name + '.bvh')
        if target.exists():
            raise FileExistsError(target)
        source.rename(target)
    report(97, 'Добавление строки в таблицу')
    append_index(session.root, dict(zip(INDEX_FIELDS, (session.timestamp, session.subject_id, session.word_set_id,
        f'{session.path.name}/custom.bvh', f'{session.path.name}/noitom.bvh'))))
    return {'frames': len(grid), 'fps': fps, 'frame_time': 1 / fps, 'source_rates_hz': rates,
            'max_source_gaps_ns': gaps, 't0_ns': session.t0_ns, 't1_ns': session.stop_ns,
            'skeleton': SKELETON_ID, 'skeleton_height_m': HEIGHT_M, 'joints': len(recordings[0].joints),
            'channels': 3 + 3 * len(recordings[0].joints), 'root_translation': 'fixed_common_origin',
            'tracked_segments': list(SEGMENTS), 'export_fps_limit': session.max_fps}
