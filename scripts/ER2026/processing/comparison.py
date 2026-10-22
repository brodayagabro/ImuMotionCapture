"""Per-recording metrics; undefined correlations are null, never perfect scores."""
import numpy as np
from .motion import ARMS, POINTS, require_pair


def correlation(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    if len(a) < 3 or np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return None
    a, b = a - a.mean(), b - b.mean()
    return float(np.clip(np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b)), -1., 1.))


def angle_degrees(a, b):
    relative = np.swapaxes(a, -1, -2) @ b
    cosine = (np.trace(relative, axis1=-2, axis2=-1) - 1) / 2
    return np.degrees(np.arccos(np.clip(cosine, -1, 1)))


def summary(values):
    return dict(rms=float(np.sqrt(np.mean(values ** 2))), mean=float(np.mean(values)),
                median=float(np.median(values)), p95=float(np.percentile(values, 95)))


def compare(a, b, recording_id):
    require_pair(a, b)
    rows = []
    for aligned in (False, True):
        frame = 'spine' if aligned else 'world'
        ar, br = (a.body_rotations(), b.body_rotations()) if aligned else (a.rotations, b.rotations)
        ap, bp = a.points(aligned), b.points(aligned)
        for name in ('Spine', *ARMS):
            i = a.index(name)
            rows.append(dict(recording_id=recording_id, frame=frame, kind='rotation_deg', joint=name,
                             **summary(angle_degrees(ar[:, i], br[:, i]))))
        for i, name in enumerate(POINTS):
            delta = (ap[:, i] - bp[:, i]) * 1000
            rows.append(dict(recording_id=recording_id, frame=frame, kind='position_mm', joint=name,
                             **summary(np.linalg.norm(delta, axis=1)),
                             **{f'pearson_{axis}': correlation(ap[:, i, j], bp[:, i, j])
                                for j, axis in enumerate('xyz')},
                             **{f'rmse_{axis}': float(np.sqrt(np.mean(delta[:, j] ** 2)))
                                for j, axis in enumerate('xyz')}))
    da, db = a.directions(), b.directions()
    for name in ARMS:
        i = a.index(name)
        parent = a.parents[i]
        local_a = np.swapaxes(a.rotations[:, parent], -1, -2) @ a.rotations[:, i]
        local_b = np.swapaxes(b.rotations[:, parent], -1, -2) @ b.rotations[:, i]
        rows.append(dict(recording_id=recording_id, frame='parent', kind='rotation_deg', joint=name,
                         **summary(angle_degrees(local_a, local_b))))
    for side, upper, lower in (('LeftElbow', 0, 1), ('RightElbow', 2, 3)):
        flex_a = np.degrees(np.arccos(np.clip(np.sum(da[:, upper] * da[:, lower], axis=-1), -1, 1)))
        flex_b = np.degrees(np.arccos(np.clip(np.sum(db[:, upper] * db[:, lower], axis=-1), -1, 1)))
        delta = flex_a - flex_b
        rows.append(dict(recording_id=recording_id, frame='anatomical', kind='elbow_flexion_deg', joint=side,
                         **summary(np.abs(delta)), bias=float(delta.mean()), pearson=correlation(flex_a, flex_b)))
    for i, name in enumerate(ARMS):
        error = np.degrees(np.arccos(np.clip(np.sum(da[:, i] * db[:, i], axis=-1), -1, 1)))
        rows.append(dict(recording_id=recording_id, frame='spine', kind='direction_deg', joint=name,
                         **summary(error), **{f'pearson_{axis}': correlation(da[:, i, j], db[:, i, j])
                                               for j, axis in enumerate('xyz')}))
    # Diagnostic only. Positive lag: Noitom motion occurs later than Custom.
    sa = np.linalg.norm(np.diff(da, axis=0), axis=-1).mean(axis=1)
    sb = np.linalg.norm(np.diff(db, axis=0), axis=-1).mean(axis=1)
    limit = min(int(.5 / a.dt), len(sa) // 4)
    lags = []
    for lag in range(-limit, limit + 1):
        x = sa[max(0, -lag):len(sa) - max(0, lag)]
        y = sb[max(0, lag):len(sb) - max(0, -lag)]
        score = correlation(x, y)
        if score is not None:
            lags.append((score, lag))
    best = max(lags) if lags else None
    diagnostic = dict(best_lag_s=best[1] * a.dt if best else None,
                      speed_correlation=best[0] if best else None,
                      search_limit_s=limit * a.dt, applied=False,
                      at_search_boundary=abs(best[1]) == limit if best else False)
    return rows, diagnostic


def time_metrics(a, b, recording_id, seconds=10.):
    """Small disjoint bins reveal drift hidden by a full-recording RMS."""
    require_pair(a, b)
    ar, br = a.body_rotations(), b.body_rotations()
    angles = angle_degrees(ar, br)
    positions = np.linalg.norm((a.points() - b.points()) * 1000, axis=-1)
    directions = np.degrees(np.arccos(np.clip(np.sum(a.directions() * b.directions(), axis=-1), -1, 1)))
    width = max(1, round(seconds / a.dt))
    rows = []
    for start in range(0, a.count, width):
        end = min(start + width, a.count)
        if end - start < 2:
            continue
        for kind, names, values in (('rotation_deg', ARMS, angles[:, [a.index(n) for n in ARMS]]),
                                    ('position_mm', POINTS, positions), ('direction_deg', ARMS, directions)):
            for j, name in enumerate(names):
                rows.append(dict(recording_id=recording_id, start_s=start * a.dt,
                                 end_s=min(end * a.dt, a.duration), kind=kind, joint=name,
                                 **summary(values[start:end, j])))
    return rows
