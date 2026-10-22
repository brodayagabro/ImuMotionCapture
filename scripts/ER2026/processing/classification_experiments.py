"""Window features and grouped model comparisons for reviewed mocap holds.

No annotation edits, random frame splits, global scaling, or automatic tuning.
The notebook owns plots and experiment settings; this module owns validation.
"""
from dataclasses import dataclass
import hashlib
from pathlib import Path
from time import perf_counter
import warnings

import numpy as np
import pandas as pd

from .annotations import EXCLUDED, read_document
from .classifier import canonical, feature
from .io import indexed_records
from .motion import ARMS, load_bvh

FEATURE_NAMES = [f'{joint}_{axis}_mean' for joint in ARMS for axis in 'xyz'] + [
    f'{joint}_{axis}_std' for joint in ARMS for axis in 'xyz']
MODEL_NAMES = ('Dummy', 'Nearest centroid', 'kNN', 'Logistic regression',
               'Linear SVM', 'RBF SVM', 'Random forest')
META_COLUMNS = ['sample_id', 'recording_id', 'subject_id', 'word_set_id', 'hold_id',
                'interval_id', 'label', 'original_label', 'source', 'start_s', 'end_s',
                'hold_start_s', 'hold_end_s', 'recording_duration_s']
HOLD_COLUMNS = ['recording_id', 'subject_id', 'hold_id', 'interval_id', 'label',
                'original_label', 'start_s', 'end_s', 'duration_s', 'windows', 'status']


@dataclass
class Dataset:
    features: dict
    samples: pd.DataFrame
    holds: pd.DataFrame
    inventory: pd.DataFrame
    manifest: list


def build_dataset(records_root, labels_root, *, source='noitom', window_s=.25,
                  step_s=.25, trim_s=.1, include_neutral=False, recording_ids=None):
    """Only complete windows inside reviewed intervals; source hashes are checked."""
    if source not in ('noitom', 'custom'):
        raise ValueError('source must be noitom or custom')
    if not np.isfinite([window_s, step_s, trim_s]).all() or min(window_s, step_s) <= 0 or trim_s < 0:
        raise ValueError('Window/step must be positive and trim nonnegative')
    records = list(indexed_records(records_root))
    if recording_ids is not None:
        missing = set(recording_ids) - {r['recording_id'] for r in records}
        if missing:
            raise ValueError(f'Unknown recording IDs: {sorted(missing)}')
        records = [r for r in records if r['recording_id'] in recording_ids]
    values, samples, holds, inventory, manifest = [], [], [], [], []
    for row in records:
        rid = row['recording_id']
        label_path = Path(labels_root) / (rid + '.json')
        info = dict(recording_id=rid, subject_id=row['subject_id'], source=source,
                    intervals=0, usable_holds=0, windows=0)
        if source not in row['paths'] or not label_path.exists():
            info['status'] = 'missing_source' if source not in row['paths'] else 'not_annotated'
            inventory.append(info)
            continue
        try:
            motion = load_bvh(row['paths'][source])
            annotation_bytes = label_path.read_bytes()
            document = read_document(label_path, row, {source: motion})
            if annotation_bytes != label_path.read_bytes():
                raise ValueError('Annotations changed during loading; rerun this cell')
        except (ValueError, OSError, KeyError, TypeError) as error:
            raise ValueError(f'{rid}: {error}') from error
        manifest.append(dict(recording_id=rid, source=source, source_sha256=motion.sha256,
                             annotation_sha256=hashlib.sha256(annotation_bytes).hexdigest()))
        directions = motion.directions()
        width = max(2, int(np.ceil(window_s / motion.dt - 1e-8)))
        step = max(1, round(step_s / motion.dt))
        if step_s >= window_s:
            step = max(step, width)
        for item in document['intervals']:
            label = canonical(item['label'])
            hold_id = rid + '/' + item['id']
            hold = dict(recording_id=rid, subject_id=row['subject_id'], hold_id=hold_id,
                        interval_id=item['id'], label=label, original_label=item['label'],
                        start_s=item['start_s'], end_s=item['end_s'],
                        duration_s=item['end_s'] - item['start_s'], windows=0)
            if not item['reviewed']:
                hold['status'] = 'unreviewed'
            elif label in EXCLUDED:
                hold['status'] = 'excluded_label'
            elif label == 'NEUTRAL' and not include_neutral:
                hold['status'] = 'neutral_disabled'
            else:
                first = int(np.ceil((item['start_s'] + trim_s) / motion.dt - 1e-8))
                stop = min(motion.count, int(np.ceil((item['end_s'] - trim_s) / motion.dt - 1e-8)))
                for a in range(first, stop - width + 1, step):
                    b = a + width
                    values.append(feature(directions[a:b]))
                    samples.append(dict(sample_id=len(samples), recording_id=rid,
                        subject_id=row['subject_id'], word_set_id=row['word_set_id'],
                        hold_id=hold_id, interval_id=item['id'], label=label,
                        original_label=item['label'], source=source,
                        start_s=a * motion.dt, end_s=b * motion.dt,
                        hold_start_s=item['start_s'], hold_end_s=item['end_s'],
                        recording_duration_s=motion.duration))
                    hold['windows'] += 1
                hold['status'] = 'ready' if hold['windows'] else 'too_short_after_trim'
            holds.append(hold)
            info['intervals'] += 1
            info['usable_holds'] += hold['status'] == 'ready'
            info['windows'] += hold['windows']
        info['status'] = 'ready' if info['windows'] else 'no_usable_windows'
        inventory.append(info)
    x = np.asarray(values, dtype=float).reshape(-1, len(FEATURE_NAMES))
    if not np.isfinite(x).all():
        raise ValueError('Non-finite features')
    return Dataset({'mean': x[:, :12], 'mean_std': x},
                   pd.DataFrame(samples, columns=META_COLUMNS),
                   pd.DataFrame(holds, columns=HOLD_COLUMNS), pd.DataFrame(inventory), manifest)


def make_splits(samples, *, mode='auto', n_blocks=5, purge_s=.5, subject_map=None):
    """Leave one subject/recording out, or one contiguous time block out.

    Time blocks use the midpoint of whole holds. Training holds intersecting
    the test envelope plus purge_s in the same recording are removed entirely.
    This is within-session validation, not forecasting or unseen-person testing.
    """
    if mode not in ('auto', 'subject', 'recording', 'blocked_time'):
        raise ValueError('Unknown split mode')
    if not isinstance(n_blocks, int) or n_blocks < 2 or not np.isfinite(purge_s) or purge_s < 0:
        raise ValueError('Use at least two time blocks and a nonnegative purge gap')
    if samples.empty:
        return [], pd.DataFrame(), 'unavailable'
    raw = samples['subject_id']
    if subject_map is not None:
        if not isinstance(subject_map, dict) or not set(raw) <= subject_map.keys() or any(
                not isinstance(v, str) or not v for v in subject_map.values()):
            raise ValueError('subject_map must map every annotated subject ID to a nonempty person ID')
        subjects = raw.map(subject_map)
    else:
        subjects = raw
        aliases = {s for s in raw.unique() if s.endswith('_mio') and s[:-4] in set(raw)}
        if aliases and mode in ('auto', 'subject'):
            raise ValueError('Confirm participant identities in SUBJECT_MAP before subject validation: '
                             + ', '.join(sorted(aliases)))
    if mode == 'auto':
        mode = ('subject' if subjects.nunique() > 1 else
                'recording' if samples['recording_id'].nunique() > 1 else 'blocked_time')
    if mode in ('subject', 'recording'):
        groups = subjects if mode == 'subject' else samples['recording_id']
        masks = [(str(group), (groups == group).to_numpy()) for group in sorted(groups.unique())]
    else:
        midpoint = (samples['hold_start_s'] + samples['hold_end_s']) / 2
        groups = np.minimum(n_blocks - 1, np.floor(midpoint / samples['recording_duration_s'] * n_blocks)).astype(int)
        masks = [(f'time_block_{group + 1}', np.asarray(groups == group)) for group in sorted(set(groups))]
    folds, audit = [], []
    for number, (name, test) in enumerate(masks, 1):
        train = ~test
        before = int(train.sum())
        if mode == 'blocked_time':
            for rid, part in samples.loc[test].groupby('recording_id'):
                lo, hi = part['hold_start_s'].min() - purge_s, part['hold_end_s'].max() + purge_s
                near = ((samples['recording_id'] == rid) & (samples['hold_start_s'] < hi) &
                        (samples['hold_end_s'] > lo)).to_numpy()
                train &= ~near
        train_idx, test_idx = np.flatnonzero(train), np.flatnonzero(test)
        train_labels, test_labels = set(samples.loc[train, 'label']), set(samples.loc[test, 'label'])
        reason = ('No training windows after splitting/purging' if not len(train_idx) else
                  'Fewer than two classes in training' if len(train_labels) < 2 else '')
        assert not set(samples.loc[train, 'hold_id']) & set(samples.loc[test, 'hold_id'])
        audit.append(dict(fold=number, held_out=name, mode=mode, status='skipped' if reason else 'ready',
            reason=reason, train_windows=len(train_idx), test_windows=len(test_idx),
            train_holds=samples.loc[train, 'hold_id'].nunique(), test_holds=samples.loc[test, 'hold_id'].nunique(),
            purged_windows=before - len(train_idx), unseen_test_classes=', '.join(sorted(test_labels - train_labels))))
        if not reason:
            folds.append(dict(fold=number, name=name, train=train_idx, test=test_idx))
    return folds, pd.DataFrame(audit), mode


def make_model(name, n_train, seed=42):
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC, SVC

    if name == 'Dummy':
        return DummyClassifier(strategy='most_frequent')
    if name == 'Random forest':
        return RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
            class_weight='balanced_subsample', random_state=seed, n_jobs=1)
    estimators = {
        'Nearest centroid': NearestCentroid(),
        'kNN': KNeighborsClassifier(n_neighbors=min(5, n_train), weights='distance'),
        'Logistic regression': LogisticRegression(C=1., class_weight='balanced', max_iter=3000, random_state=seed),
        'Linear SVM': LinearSVC(C=1., class_weight='balanced', max_iter=10000, random_state=seed),
        'RBF SVM': SVC(C=10., gamma='scale', class_weight='balanced'),
    }
    if name not in estimators:
        raise ValueError(f'Unknown model: {name}')
    return make_pipeline(StandardScaler(), estimators[name])


def metric_scores(truth, predicted, classes):
    from sklearn.metrics import accuracy_score, f1_score, recall_score
    return dict(accuracy=float(accuracy_score(truth, predicted)),
        macro_f1=float(f1_score(truth, predicted, labels=classes, average='macro', zero_division=0)),
        balanced_accuracy=float(recall_score(truth, predicted, labels=np.unique(truth),
                                            average='macro', zero_division=0)))


def vote_holds(predictions):
    """Every hold receives one vote; ties use lexicographic class order."""
    rows = []
    if predictions.empty:
        return pd.DataFrame()
    for (model, features, fold, hold_id), part in predictions.groupby(['model', 'features', 'fold', 'hold_id']):
        labels, counts = np.unique(part['predicted'], return_counts=True)
        winner = int(np.argmax(counts))
        first = part.iloc[0]
        assert part['label'].nunique() == 1
        rows.append(dict(model=model, features=features, fold=fold, hold_id=hold_id,
            recording_id=first['recording_id'], subject_id=first['subject_id'],
            interval_id=first['interval_id'], label=first['label'], predicted=labels[winner],
            start_s=float(first['hold_start_s']), end_s=float(first['hold_end_s']),
            windows=len(part), vote_fraction=float(counts[winner] / counts.sum()),
            unseen_in_train=bool(first['unseen_in_train'])))
    return pd.DataFrame(rows)


def benchmark(dataset, folds, *, models=MODEL_NAMES, feature_sets=('mean', 'mean_std'), seed=42, progress=None):
    """Same folds for every model; transformers fit inside each training fold."""
    if not folds:
        return dict(summary=pd.DataFrame(), folds=pd.DataFrame(), windows=pd.DataFrame(), holds=pd.DataFrame())
    if not models or not feature_sets:
        raise ValueError('Choose at least one model and feature set')
    classes = np.sort(dataset.samples['label'].unique())
    y = dataset.samples['label'].to_numpy()
    predictions, fold_rows = [], []
    for features in feature_sets:
        x = dataset.features[features]
        for name in models:
            if progress:
                progress(f'{features}: {name}')
            for fold in folds:
                train, test = fold['train'], fold['test']
                if set(dataset.samples.iloc[train]['hold_id']) & set(dataset.samples.iloc[test]['hold_id']):
                    raise ValueError('A hold leaks between training and validation')
                estimator = make_model(name, len(train), seed)
                start = perf_counter()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    estimator.fit(x[train], y[train])
                    predicted = estimator.predict(x[test])
                elapsed = perf_counter() - start
                part = dataset.samples.iloc[test].copy()
                part['predicted'], part['model'], part['features'], part['fold'] = predicted, name, features, fold['fold']
                part['unseen_in_train'] = ~part['label'].isin(set(y[train]))
                predictions.append(part)
                hold_predictions = vote_holds(part)
                fold_rows.append(dict(model=name, features=features, fold=fold['fold'], held_out=fold['name'],
                    fit_predict_s=elapsed, warnings='; '.join(sorted({str(w.message) for w in caught})),
                    **{'window_' + k: v for k, v in metric_scores(y[test], predicted, classes).items()},
                    **{'hold_' + k: v for k, v in metric_scores(hold_predictions['label'], hold_predictions['predicted'], classes).items()}))
    windows = pd.concat(predictions, ignore_index=True)
    if windows.duplicated(['model', 'features', 'sample_id']).any():
        raise ValueError('A validation sample occurs in more than one fold')
    holds = vote_holds(windows)
    fold_table = pd.DataFrame(fold_rows)
    summary = []
    for (name, features), part in windows.groupby(['model', 'features']):
        hp = holds[(holds['model'] == name) & (holds['features'] == features)]
        ft = fold_table[(fold_table['model'] == name) & (fold_table['features'] == features)]
        summary.append(dict(model=name, features=features, folds=len(ft), windows=len(part), holds=len(hp),
            hold_coverage=len(hp) / dataset.samples['hold_id'].nunique(),
            unseen_test_holds=int(hp['unseen_in_train'].sum()), fit_predict_s=float(ft['fit_predict_s'].sum()),
            **{'window_' + k: v for k, v in metric_scores(part['label'], part['predicted'], classes).items()},
            **{'hold_' + k: v for k, v in metric_scores(hp['label'], hp['predicted'], classes).items()}))
    summary = pd.DataFrame(summary).sort_values(['hold_macro_f1', 'hold_balanced_accuracy'], ascending=False).reset_index(drop=True)
    return dict(summary=summary, folds=fold_table, windows=windows, holds=holds)


def split_membership(samples, folds):
    """Export held-out, training and purged membership for reproducibility."""
    parts = []
    for fold in folds:
        part = samples[['sample_id', 'recording_id', 'hold_id', 'start_s', 'end_s', 'label']].copy()
        part['fold'], part['role'] = fold['fold'], 'purged'
        part.loc[fold['train'], 'role'] = 'train'
        part.loc[fold['test'], 'role'] = 'validation'
        parts.append(part)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
