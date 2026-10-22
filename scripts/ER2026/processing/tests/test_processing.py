import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from scripts.ER2026.noitom_comparison.bvh_skeleton import create_comparison_recording
from scripts.ER2026.processing.annotations import (candidates, interval, new_document,
    read_document, save_document, validate_intervals)
from scripts.ER2026.processing.classifier import (feature, fit, grouped_evaluate,
    predict, predict_file, train, windows)
from scripts.ER2026.processing.comparison import angle_degrees, compare, correlation
from scripts.ER2026.processing.io import indexed_records, write_csv
from scripts.ER2026.processing.motion import load_bvh, require_pair
from scripts.ER2026.processing.pipeline import run


def write_motion(path, global_rotation=None, frames=101):
    recording = create_comparison_recording(50)
    times = np.arange(frames) / 50
    q = np.eye(3) if global_rotation is None else global_rotation
    orientations = {}
    for i, name in enumerate(('spine', 'shoulder.L', 'forearm.L', 'shoulder.R', 'forearm.R')):
        angles = np.column_stack((15 * np.sin(times + i), 10 * np.cos(times + i), times * (i + 1) * 10))
        orientations[name] = q @ Rotation.from_euler('ZXY', angles, degrees=True).as_matrix()
    recording.append_batch(times, orientations)
    recording.save(path, uniform_grid=True)
    return orientations


@pytest.fixture
def pair(tmp_path):
    a, b = tmp_path / 'a.bvh', tmp_path / 'b.bvh'
    write_motion(a)
    write_motion(b, Rotation.from_euler('XYZ', [35, 40, 75], degrees=True).as_matrix())
    return load_bvh(a), load_bvh(b)


def test_roundtrip_noncommuting_rotations(tmp_path):
    from storage.bvh import TO_BVH
    path = tmp_path / 'motion.bvh'
    expected = write_motion(path)
    motion = load_bvh(path)
    for name, source in (('Spine', 'spine'), ('LeftArm', 'shoulder.L'), ('LeftForeArm', 'forearm.L')):
        np.testing.assert_allclose(motion.rotations[:, motion.index(name)],
                                   TO_BVH @ expected[source] @ TO_BVH.T, atol=1e-8)
    assert motion.count == 101 and motion.duration == 2
    assert motion.points().shape == (101, 4, 3)
    np.testing.assert_allclose(np.linalg.norm(motion.directions(), axis=-1), 1, atol=1e-9)


def test_spine_alignment_removes_global_rigid_rotation(pair):
    a, b = pair
    rows, lag = compare(a, b, 'pair')
    for row in rows:
        if row['frame'] == 'spine':
            assert row['rms'] < 1e-4
    assert max(r['rms'] for r in rows if r['frame'] == 'world') > 10
    assert lag['applied'] is False
    upper = [a.index(n) for n in ('Spine', 'Chest', 'Head', 'LeftArm', 'LeftForeArm', 'RightArm', 'RightForeArm')]
    np.testing.assert_allclose(a.body_positions()[:, upper], b.body_positions()[:, upper], atol=1e-8)


def test_geodesic_wrap_and_constant_correlation():
    a = Rotation.from_euler('Y', [179], degrees=True).as_matrix()
    b = Rotation.from_euler('Y', [-179], degrees=True).as_matrix()
    np.testing.assert_allclose(angle_degrees(a, b), 2, atol=1e-9)
    assert correlation(np.ones(20), np.ones(20)) is None


def test_pair_time_grid_rejected(pair):
    a, b = pair
    b.dt = .03
    with pytest.raises(ValueError, match='time grids'):
        require_pair(a, b)


def test_annotation_overlap_hash_and_atomic_backup(pair, tmp_path):
    a, _ = pair
    row = dict(recording_id='session', subject_id='person', timestamp='date', word_set_id='SET01')
    document = new_document(row, {'noitom': a})
    document['intervals'] = [interval(.1, .9, 'А'), interval(.9, 1.8, 'NEUTRAL')]
    path = tmp_path / 'labels.json'
    save_document(path, document, a.duration)
    save_document(path, document, a.duration)
    assert path.with_suffix('.json.bak').exists()
    assert read_document(path, row, {'noitom': a})['intervals'] == document['intervals']
    with pytest.raises(ValueError, match='overlap'):
        validate_intervals([interval(0, 1, 'А'), interval(.5, 1.5, 'Б')], 2)
    a.sha256 = 'changed'
    with pytest.raises(ValueError, match='changed'):
        read_document(path, row, {'noitom': a})


def test_unreviewed_and_candidates(pair):
    a, _ = pair
    a.rotations[:] = np.eye(3)
    result = candidates(a)
    assert result == [(.1, 1.9)]
    assert feature(a.directions()[:10]).shape == (24,)


def test_grouped_validation_has_no_training_group_leak():
    x = np.array([[0., 0.], [10., 10.], [.1, .1], [10.1, 10.1], [-.1, -.1], [9.9, 9.9]])
    y = np.array(['А', 'Б', 'А', 'Б', 'А', 'Б'])
    groups = np.array(['P1', 'P1', 'P2', 'P2', 'P3', 'P3'])
    report, predictions = grouped_evaluate(x, y, groups)
    assert report['accuracy'] == 1 and len(report['folds']) == 3
    assert len(predictions) == len(x)
    for p in predictions:
        i = p['sample_index']
        assert groups[i] == p['held_out_group']
    report, _ = grouped_evaluate(x, y, ['P1'] * 6)
    assert report['status'] == 'not_evaluated'


def test_unseen_test_class_is_reported():
    report, _ = grouped_evaluate(np.array([[0.], [1.]]), np.array(['А', 'Б']), ['P1', 'P2'])
    assert report['accuracy'] == 0
    assert report['folds'][0]['unseen_test_classes'] == ['А']


def make_dataset(tmp_path, *, beside_bvh=False):
    records = tmp_path / 'records'
    labels = tmp_path / 'labels'
    records.mkdir()
    labels.mkdir()
    rows = []
    for person in ('P1', 'P2'):
        folder = records / person
        folder.mkdir()
        write_motion(folder / 'noitom.bvh', frames=201)
        row = dict(timestamp='2026-10-21T13:00:00+03:00', subject_id=person, word_set_id='SET01',
                   custom_bvh_path='', noitom_bvh_path=f'{person}/noitom.bvh')
        rows.append(row)
    write_csv(records / 'records.csv', rows)
    for row in indexed_records(records):
        motion = load_bvh(row['paths']['noitom'])
        document = new_document(row, {'noitom': motion})
        document['intervals'] = [interval(.1, 1.5, 'А'), interval(2., 3.5, 'Б'), interval(3.5, 4., 'BAD')]
        folder = records / row['recording_id'] if beside_bvh else labels
        save_document(folder / (row['recording_id'] + '.json'), document, motion.duration)
    return records, None if beside_bvh else labels


@pytest.mark.parametrize('beside_bvh', [False, True])
def test_training_and_prediction_end_to_end(tmp_path, beside_bvh):
    records, labels = make_dataset(tmp_path, beside_bvh=beside_bvh)
    output = tmp_path / 'model'
    report = train(records, labels, output)
    assert report['status'] == 'evaluated'
    assert {p['label'] for p in report['per_class']} == {'А', 'Б'}
    predict_file(records / 'P1/noitom.bvh', output / 'model.npz', output / 'predictions.csv')
    assert (output / 'predictions.csv').exists()
    with np.load(output / 'model.npz', allow_pickle=False) as model:
        assert model['centers'].shape == (2, 24)


def test_empty_annotations_do_not_create_fake_model(tmp_path):
    records, labels = make_dataset(tmp_path)
    for p in labels.glob('*.json'):
        d = json.loads(p.read_text(encoding='utf-8'))
        d['intervals'] = []
        p.write_text(json.dumps(d), encoding='utf-8')
    with pytest.raises(ValueError, match='manually reviewed'):
        train(records, labels, tmp_path / 'model')
    assert not (tmp_path / 'model/model.npz').exists()


def test_audit_tracks_bad_file_and_orphan(tmp_path):
    records, _ = make_dataset(tmp_path)
    (records / 'P1/noitom.bvh').write_text('invalid', encoding='ascii')
    (records / 'orphan').mkdir()
    (records / 'orphan/custom.bvh').write_text('not indexed', encoding='ascii')
    report = run(records, tmp_path / 'analysis')
    assert report['indexed_recordings'] == 2
    assert not report['inventory'][0]['valid']
    assert report['unindexed_bvh'] == ['orphan/custom.bvh']


def test_index_rejects_path_escape(tmp_path):
    write_csv(tmp_path / 'records.csv', [dict(timestamp='x', subject_id='x', word_set_id='x',
                                            custom_bvh_path='../out.bvh', noitom_bvh_path='')])
    with pytest.raises(ValueError, match='outside records'):
        list(indexed_records(tmp_path))


@pytest.mark.parametrize('beside_bvh', [False, True])
def test_reviewed_label_export_preserves_bvh_time_basis(tmp_path, beside_bvh):
    from scripts.ER2026.processing.export import export_labels
    records, labels = make_dataset(tmp_path, beside_bvh=beside_bvh)
    rows = export_labels(records, labels, tmp_path / 'export.csv')
    assert len(rows) == 6
    assert all(r['time_basis'] == 'BVH seconds; half-open [start_s,end_s)' for r in rows)
    assert rows[0]['wall_start_hint'] == '2026-10-21T13:00:00.100000+03:00'
    assert sum(r['usable_pose'] for r in rows) == 4


def test_prediction_windows_do_not_overlap_at_50_hz(pair):
    motion, _ = pair
    spans = list(windows(motion, 0, motion.duration, .25))
    assert all(a1 <= a2 and b1 <= a2 for (a1, b1), (a2, b2) in zip(spans, spans[1:]))


def test_static_pair_has_no_invented_lag(tmp_path):
    records, _ = make_dataset(tmp_path)
    folder = records / 'P1'
    motion = create_comparison_recording(50)
    times = np.arange(51) / 50
    orientations = {s: np.broadcast_to(np.eye(3), (len(times), 3, 3))
                    for s in ('spine', 'shoulder.L', 'forearm.L', 'shoulder.R', 'forearm.R')}
    motion.append_batch(times, orientations)
    motion.save(folder / 'custom.bvh', uniform_grid=True)
    motion.save(folder / 'noitom.bvh', uniform_grid=True)
    write_csv(records / 'records.csv', [dict(timestamp='2026-10-21T13:00:00+03:00', subject_id='P1',
        word_set_id='SET01', custom_bvh_path='P1/custom.bvh', noitom_bvh_path='P1/noitom.bvh')])
    report = run(records, tmp_path / 'static_analysis')
    assert report['successful_comparisons'] == 1
    assert report['lag_diagnostics']['P1']['best_lag_s'] is None


@pytest.mark.parametrize('beside_bvh', [False, True])
def test_comparison_of_reviewed_intervals(tmp_path, beside_bvh):
    records, labels = make_dataset(tmp_path, beside_bvh=beside_bvh)
    folder = records / 'P1'
    write_motion(folder / 'custom.bvh', frames=201)
    rows = list(indexed_records(records))
    index_rows = [{k: r[k] for k in ('timestamp', 'subject_id', 'word_set_id', 'custom_bvh_path', 'noitom_bvh_path')}
                  for r in rows]
    index_rows[0]['custom_bvh_path'] = 'P1/custom.bvh'
    write_csv(records / 'records.csv', index_rows)
    row = list(indexed_records(records))[0]
    motions = {s: load_bvh(p) for s, p in row['paths'].items()}
    document = new_document(row, motions)
    document['intervals'] = [interval(.1, 1.5, 'А'), interval(2, 3, 'BAD')]
    save_document((folder if beside_bvh else labels) / 'P1.json', document, 4.)
    report = run(records, tmp_path / 'interval_analysis', labels, use_labels=True)
    assert report['annotation_metrics'] == dict(requested=True, errors={}, intervals=1)


def test_motion_audit_ignores_auxiliary_txt(tmp_path, monkeypatch):
    records, _ = make_dataset(tmp_path)
    (records / 'P1/unrelated.txt').write_text('not motion data', encoding='utf-8')
    original = Path.open

    def motion_only_open(path, *args, **kwargs):
        if path.suffix.lower() == '.txt':
            raise AssertionError('Motion audit must not read auxiliary TXT')
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', motion_only_open)
    report = run(records, tmp_path / 'motion_only_analysis')
    assert report['schema_version'] == 2
    assert 'emg' not in report
