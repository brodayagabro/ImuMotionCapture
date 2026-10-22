"""Reproducible pose baseline, fitted only to manually reviewed intervals.

Nearest class centroid on standardized mean/std bone directions in 250 ms
windows. Evaluation leaves out entire participants (or recordings). Scaling
is fitted again in every training fold. No pickle, automatic pseudo-labels,
assumptions about other modalities, or random frame split.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from .annotations import EXCLUDED, read_document
from .io import atomic_json, indexed_records, write_csv
from .motion import load_bvh

WINDOW_S = .25
STEP_S = .125
TRIM_S = .1
FEATURE_VERSION = 'spine_bone_direction_mean_std_v1'


def canonical(label):
    return 'Е/Э' if label in ('Е', 'Э') else label


def feature(values):
    values = np.asarray(values, dtype=float).reshape(len(values), 12)
    return np.r_[values.mean(axis=0), values.std(axis=0)]


def windows(motion, start, end, step_s=STEP_S):
    width = max(2, int(np.ceil(WINDOW_S / motion.dt - 1e-8)))
    step = max(1, round(step_s / motion.dt))
    if step_s >= WINDOW_S:
        step = max(step, width)
    first = int(np.ceil(start / motion.dt - 1e-8))
    stop = min(motion.count, int(np.ceil(end / motion.dt - 1e-8)))
    for i in range(first, stop - width + 1, step):
        yield i, i + width


def fit(x, y):
    x, y = np.asarray(x, float), np.asarray(y)
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale = np.maximum(scale, .05)  # Avoid magnifying tiny sensor jitter.
    z = (x - mean) / scale
    classes = np.unique(y)
    centers = np.stack([z[y == label].mean(axis=0) for label in classes])
    return dict(mean=mean, scale=scale, classes=classes, centers=centers)


def predict(model, x):
    z = (np.asarray(x) - model['mean']) / model['scale']
    distances = np.linalg.norm(z[:, None] - model['centers'][None], axis=-1)
    return model['classes'][distances.argmin(axis=1)], distances.min(axis=1)


def scores(truth, predicted):
    classes = np.unique(np.r_[truth, predicted])
    matrix = np.zeros((len(classes), len(classes)), int)
    for a, b in zip(truth, predicted):
        matrix[np.searchsorted(classes, a), np.searchsorted(classes, b)] += 1
    support = matrix.sum(axis=1)
    recall = np.diag(matrix) / np.maximum(support, 1)
    precision = np.diag(matrix) / np.maximum(matrix.sum(axis=0), 1)
    f1 = 2 * recall * precision / np.maximum(recall + precision, 1e-12)
    return dict(accuracy=float(np.mean(truth == predicted)),
                balanced_accuracy=float(recall[support > 0].mean()),
                macro_f1=float(f1.mean()), classes=classes.tolist(),
                confusion_matrix=matrix.tolist(),
                per_class=[dict(label=str(c), support=int(n), recall=float(r), f1=float(f))
                           for c, n, r, f in zip(classes, support, recall, f1)])


def grouped_evaluate(x, y, groups):
    groups = np.asarray(groups)
    if len(np.unique(groups)) < 2:
        return dict(status='not_evaluated', reason='At least two independent groups are required'), []
    predicted = np.empty(len(y), dtype=y.dtype)
    folds, predictions = [], []
    for group in np.unique(groups):
        test = groups == group
        model = fit(x[~test], y[~test])
        predicted[test], distances = predict(model, x[test])
        folds.append(dict(held_out_group=str(group), train_windows=int((~test).sum()),
                          test_windows=int(test.sum()),
                          unseen_test_classes=sorted(set(y[test]) - set(y[~test])),
                          **scores(y[test], predicted[test])))
        predictions.extend(dict(sample_index=int(i), held_out_group=str(group),
                                truth=str(y[i]), predicted=str(p), distance=float(d))
                           for i, p, d in zip(np.flatnonzero(test), predicted[test], distances))
    return dict(status='evaluated', folds=folds, **scores(y, predicted)), predictions


def train(records, labels, output, source='noitom', group='subject', subjects=None):
    records, labels, output = Path(records).resolve(), Path(labels).resolve(), Path(output).resolve()
    if output == records or output.is_relative_to(records) or output == labels or output.is_relative_to(labels):
        raise ValueError('Classifier output must be separate from records and annotation directories')
    mapping = json.loads(Path(subjects).read_text(encoding='utf-8')) if subjects else {}
    if not isinstance(mapping, dict) or any(not isinstance(v, str) or not v for v in mapping.values()):
        raise ValueError('Subject mapping must be a JSON object with non-empty string values')
    all_rows = list(indexed_records(records))
    usable = [r for r in all_rows if source in r['paths'] and (labels / (r['recording_id'] + '.json')).exists()]
    raw_ids = {r['subject_id'] for r in usable}
    if group == 'subject' and any(s.endswith('_mio') and s[:-4] in raw_ids for s in raw_ids) and not subjects:
        raise ValueError('Provide --subjects mapping: IDs with/without _mio may refer to the same participant')
    if group == 'subject' and subjects and not raw_ids <= mapping.keys():
        raise ValueError('Subject mapping must cover all annotated subject IDs')
    samples, values, targets, groups, manifests, skipped = [], [], [], [], [], []
    for row in usable:
        motion = load_bvh(row['paths'][source])
        path = labels / (row['recording_id'] + '.json')
        document = read_document(path, row, {source: motion})
        manifests.append(dict(recording_id=row['recording_id'], source_sha256=motion.sha256,
                              annotation_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        directions = motion.directions()
        for item in document['intervals']:
            if not item['reviewed'] or item['label'] in EXCLUDED:
                continue
            indices = list(windows(motion, item['start_s'] + TRIM_S, item['end_s'] - TRIM_S))
            if not indices:
                skipped.append(dict(recording_id=row['recording_id'], interval_id=item['id'], reason='Too short after edge trim'))
            for a, b in indices:
                values.append(feature(directions[a:b]))
                targets.append(canonical(item['label']))
                group_id = mapping.get(row['subject_id'], row['subject_id']) if group == 'subject' else row['recording_id']
                groups.append(group_id)
                samples.append(dict(recording_id=row['recording_id'], subject_id=row['subject_id'], group=group_id,
                                    interval_id=item['id'], start_s=a * motion.dt, end_s=b * motion.dt,
                                    label=canonical(item['label']), original_label=item['label'], source=source))
    if not values or len(set(targets)) < 2:
        raise ValueError('At least two manually reviewed pose classes are needed. Run annotate first.')
    x, y = np.asarray(values), np.asarray(targets)
    evaluation, predictions = grouped_evaluate(x, y, groups)
    model = fit(x, y)
    output.mkdir(parents=True, exist_ok=True)
    # No object arrays: loading uses allow_pickle=False.
    with (output / 'model.npz').open('wb') as stream:
        np.savez_compressed(stream, **model, feature_version=np.array(FEATURE_VERSION),
                            window_s=np.array(WINDOW_S), source=np.array(source))
    write_csv(output / 'windows.csv', samples)
    write_csv(output / 'validation_predictions.csv', predictions, ['sample_index', 'held_out_group', 'truth', 'predicted', 'distance'])
    atomic_json(output / 'evaluation.json', dict(schema_version=1, feature_version=FEATURE_VERSION,
        model='standardized nearest class centroid', source=source, group=group, subject_mapping=mapping,
        window_s=WINDOW_S, step_s=STEP_S, trim_s=TRIM_S, samples=len(x),
        class_windows={str(c): int(np.sum(y == c)) for c in np.unique(y)},
        skipped_intervals=skipped, manifest=manifests, evaluation=evaluation,
        limitations=['Е and Э merged as Е/Э: identical reference poses',
                     'Pose classifier for motion capture only',
                     'Distance is not a calibrated probability; no unknown-pose rejection',
                     'Window metrics are dependent within holds; not independent trial accuracy',
                     'Recording grouping does not establish generalization to unseen people']))
    print(f"Saved {output / 'model.npz'}; windows={len(x)}; evaluation={evaluation['status']}")
    return evaluation


def predict_file(bvh, model_path, output):
    motion = load_bvh(bvh)
    with np.load(model_path, allow_pickle=False) as archive:
        if str(archive['feature_version']) != FEATURE_VERSION or float(archive['window_s']) != WINDOW_S:
            raise ValueError('Unsupported model feature version/window')
        model = {k: archive[k] for k in ('mean', 'scale', 'classes', 'centers')}
    directions = motion.directions()
    indices = list(windows(motion, 0, motion.duration, WINDOW_S))
    if not indices:
        raise ValueError('BVH is shorter than one model window')
    predicted, distances = predict(model, np.stack([feature(directions[a:b]) for a, b in indices]))
    output = Path(output).resolve()
    if output in (Path(bvh).resolve(), Path(model_path).resolve()):
        raise ValueError('Prediction output cannot overwrite source/model')
    output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(output, [dict(start_s=a * motion.dt, end_s=b * motion.dt, label=str(label),
                           distance=float(distance), reviewed=False)
                       for (a, b), label, distance in zip(indices, predicted, distances)])
