"""Reviewed half-open intervals in BVH-relative seconds, bound to source hashes."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import uuid

import numpy as np

from scripts.ER2026.experiments.semaphore.semaphore_model import SEMAPHORE_POSES
from .io import atomic_json

SPECIAL = ('NEUTRAL', 'TRANSITION', 'BAD', 'UNKNOWN')
LABELS = tuple(SEMAPHORE_POSES) + SPECIAL
EXCLUDED = frozenset(('TRANSITION', 'BAD', 'UNKNOWN'))


def validate_intervals(intervals, duration):
    previous = 0.
    ids = set()
    for item in sorted(intervals, key=lambda value: value['start_s']):
        start, end = item['start_s'], item['end_s']
        if not np.isfinite([start, end]).all() or not 0 <= start < end <= duration + 1e-8:
            raise ValueError('Interval must satisfy 0 <= start < end <= BVH duration')
        if start < previous - 1e-8:
            raise ValueError('Annotation intervals overlap')
        if item.get('label') not in LABELS:
            raise ValueError(f"Unknown label: {item.get('label')}")
        if not isinstance(item.get('reviewed'), bool):
            raise ValueError('reviewed must be boolean')
        if not item.get('id') or item['id'] in ids:
            raise ValueError('Interval IDs must be unique')
        ids.add(item['id'])
        previous = end


def new_document(row, motions):
    return dict(schema_version=1, recording_id=row['recording_id'], subject_id=row['subject_id'],
                timestamp=row['timestamp'], word_set_id=row['word_set_id'],
                time_basis='seconds from BVH frame 0; half-open [start_s,end_s)',
                sources={name: dict(sha256=m.sha256, frames=m.count, frame_time=m.dt)
                         for name, m in motions.items()}, intervals=[])


def read_document(path, row, motions):
    if not Path(path).exists():
        return new_document(row, motions)
    document = json.loads(Path(path).read_text(encoding='utf-8'))
    if document.get('schema_version') != 1 or document.get('recording_id') != row['recording_id']:
        raise ValueError('Annotation version/recording mismatch')
    if document.get('subject_id') != row['subject_id'] or document.get('timestamp') != row['timestamp']:
        raise ValueError('Annotation metadata mismatch')
    for source, motion in motions.items():
        expected = dict(sha256=motion.sha256, frames=motion.count, frame_time=motion.dt)
        if document.get('sources', {}).get(source) != expected:
            raise ValueError(f'Source changed or was not annotated: {source}')
    validate_intervals(document['intervals'], min(m.duration for m in motions.values()))
    return document


def save_document(path, document, duration):
    validate_intervals(document['intervals'], duration)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        shutil.copyfile(path, path.with_suffix('.json.bak'))
    document['updated_at'] = datetime.now(timezone.utc).isoformat()
    atomic_json(path, document)


def interval(start, end, label, note=''):
    return dict(id=uuid.uuid4().hex, start_s=float(start), end_s=float(end), label=label,
                reviewed=True, annotator_note=note)


def angular_speed(motion):
    directions = motion.directions()
    cosine = np.sum(directions[1:] * directions[:-1], axis=-1)
    speed = np.degrees(np.arccos(np.clip(cosine, -1, 1))) / motion.dt
    return np.r_[0., speed.max(axis=1)]


def candidates(motion, threshold=18., minimum_s=.5, trim_s=.10):
    """Unlabelled low-speed regions only. A human must confirm every interval."""
    stable = angular_speed(motion) < threshold
    edges = np.diff(np.r_[False, stable, False].astype(int))
    result = []
    for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        a, b = start * motion.dt + trim_s, min(end * motion.dt, motion.duration) - trim_s
        if b - a >= minimum_s:
            result.append((a, b))
    return result


def pose_hint(motion, frame, directions=None):
    """Frontal arm directions only; this is a hint, not a ground-truth label."""
    vectors = (motion.directions() if directions is None else directions)[frame, [0, 2]]
    scored = []
    for letter, pose in SEMAPHORE_POSES.items():
        angles = np.radians([pose.left_angle_deg, pose.right_angle_deg])
        target = np.column_stack((np.cos(angles), np.sin(angles), np.zeros(2)))
        error = np.degrees(np.arccos(np.clip(np.sum(vectors * target, axis=1), -1, 1))).mean()
        scored.append((error, letter))
    return sorted(scored)[:3]
