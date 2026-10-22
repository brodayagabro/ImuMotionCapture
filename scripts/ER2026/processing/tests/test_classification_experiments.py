import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip('sklearn')
from scripts.ER2026.processing.classification_experiments import (
    Dataset, benchmark, build_dataset, make_splits, split_membership)
from .test_processing import make_dataset


@pytest.mark.parametrize('beside_bvh', [False, True])
def test_features_use_reviewed_windows_and_preserve_inputs(tmp_path, beside_bvh):
    records, labels = make_dataset(tmp_path, beside_bvh=beside_bvh)
    paths = records.glob('*/*.json') if beside_bvh else labels.glob('*.json')
    before = {p: p.read_bytes() for p in paths}
    data = build_dataset(records) if beside_bvh else build_dataset(records, labels)
    assert set(data.samples['label']) == {'А', 'Б'}
    assert len(data.holds.query("status == 'excluded_label'")) == 2
    assert data.features['mean_std'].shape == (len(data.samples), 24)
    np.testing.assert_equal(data.features['mean'], data.features['mean_std'][:, :12])
    assert (data.samples['start_s'] >= data.samples['hold_start_s'] + .1 - 1e-8).all()
    assert (data.samples['end_s'] <= data.samples['hold_end_s']).all()
    assert data.samples['sample_id'].is_unique
    assert all(p.read_bytes() == b for p, b in before.items())
    path = (records / 'P1' if beside_bvh else labels) / 'P1.json'
    document = json.loads(path.read_text(encoding='utf-8'))
    document['sources']['noitom']['sha256'] = 'changed'
    path.write_text(json.dumps(document), encoding='utf-8')
    with pytest.raises(ValueError, match='Source changed'):
        build_dataset(records, labels)


def test_no_labels_and_one_subject_validation_are_explicit(tmp_path):
    records, _ = make_dataset(tmp_path)
    data = build_dataset(records, tmp_path / 'no_labels')
    assert data.samples.empty and data.features['mean_std'].shape == (0, 24)
    folds, _, mode = make_splits(data.samples)
    assert not folds and mode == 'unavailable'
    assert benchmark(data, folds)['summary'].empty


def test_scaling_is_fitted_only_on_training_subjects(tmp_path, monkeypatch):
    from sklearn.preprocessing import StandardScaler
    records, labels = make_dataset(tmp_path)
    data = build_dataset(records, labels)
    data.features['mean'][data.samples['subject_id'] == 'P2'] += 100
    folds, audit, mode = make_splits(data.samples)
    assert mode == 'subject' and len(folds) == 2
    observed = []
    original = StandardScaler.fit

    def fit(self, x, *args, **kwargs):
        observed.append(np.asarray(x).mean(axis=0))
        return original(self, x, *args, **kwargs)

    monkeypatch.setattr(StandardScaler, 'fit', fit)
    result = benchmark(data, folds, models=('Logistic regression',), feature_sets=('mean',))
    for fold, mean in zip(folds, observed):
        train, test = data.samples.iloc[fold['train']], data.samples.iloc[fold['test']]
        assert not set(train['subject_id']) & set(test['subject_id'])
        assert not set(train['hold_id']) & set(test['hold_id'])
        np.testing.assert_allclose(mean, data.features['mean'][fold['train']].mean(axis=0))
        assert not np.allclose(mean, data.features['mean'].mean(axis=0))
    assert result['windows']['sample_id'].is_unique
    assert result['holds']['hold_id'].is_unique
    assert result['summary'].iloc[0]['hold_coverage'] == 1
    assert set(audit['status']) == {'ready'}


def temporal_samples():
    rows = []
    for hold in range(12):
        for offset in (.1, .4):
            rows.append(dict(sample_id=len(rows), recording_id='session', subject_id='P1',
                hold_id=f'hold{hold}', interval_id=str(hold), label='Z' if hold == 11 else 'AB'[hold % 2],
                start_s=hold + offset, end_s=hold + offset + .25,
                hold_start_s=float(hold), hold_end_s=hold + .8, recording_duration_s=12.))
    return pd.DataFrame(rows)


def test_time_blocks_keep_holds_together_and_purge_whole_neighbours():
    samples = temporal_samples()
    folds, audit, mode = make_splits(samples, n_blocks=3, purge_s=.6)
    assert mode == 'blocked_time' and len(folds) == 3
    assert (audit['purged_windows'] > 0).all()
    assert 'Z' in audit['unseen_test_classes'].tolist()
    for fold in folds:
        train, test = samples.iloc[fold['train']], samples.iloc[fold['test']]
        assert not set(train['hold_id']) & set(test['hold_id'])
        lo, hi = test['hold_start_s'].min() - .6, test['hold_end_s'].max() + .6
        assert not ((train['hold_start_s'] < hi) & (train['hold_end_s'] > lo)).any()
        assert test.groupby('hold_id').size().eq(2).all()
    membership = split_membership(samples, folds)
    assert membership.groupby(['fold', 'hold_id'])['role'].nunique().eq(1).all()
    assert membership.query("role == 'validation'")['sample_id'].is_unique


def test_unseen_letters_are_scored_and_leaky_splits_are_rejected():
    samples = temporal_samples()
    features = np.arange(len(samples), dtype=float)[:, None]
    data = Dataset({'test': features}, samples, pd.DataFrame(), pd.DataFrame(), [])
    folds, _, _ = make_splits(samples, n_blocks=3, purge_s=.6)
    result = benchmark(data, folds, models=('Dummy',), feature_sets=('test',))
    unseen = result['holds'].query('unseen_in_train')
    assert unseen['label'].tolist() == ['Z']
    assert len(result['holds']) == 12
    assert result['summary'].iloc[0]['unseen_test_holds'] == 1
    bad = [dict(fold=1, name='bad', train=np.array([0, 2]), test=np.array([1, 3]))]
    with pytest.raises(ValueError, match='leaks'):
        benchmark(data, bad, models=('Dummy',), feature_sets=('test',))


def test_aliases_require_confirmed_participant_mapping(tmp_path):
    records, labels = make_dataset(tmp_path)
    data = build_dataset(records, labels)
    data.samples['subject_id'] = data.samples['subject_id'].replace({'P2': 'P1_mio'})
    with pytest.raises(ValueError, match='participant identities'):
        make_splits(data.samples)
    _, _, mode = make_splits(data.samples, subject_map={'P1': 'person_1', 'P1_mio': 'person_1'})
    assert mode == 'recording'
    folds, audit, _ = make_splits(data.samples, mode='subject',
                                 subject_map={'P1': 'person_1', 'P1_mio': 'person_1'})
    assert not folds and audit.iloc[0]['status'] == 'skipped'
