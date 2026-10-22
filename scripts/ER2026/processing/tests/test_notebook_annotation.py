"""Widget actions use synthetic motion; no hardware or browser is required."""
import pytest

from .test_processing import make_dataset, write_motion


@pytest.fixture
def editor(tmp_path, monkeypatch):
    pytest.importorskip('ipywidgets')
    from scripts.ER2026.processing.notebook_annotation import NoitomAnnotator
    records, labels = make_dataset(tmp_path)
    monkeypatch.setattr(NoitomAnnotator, '_draw', lambda *args: None)
    return NoitomAnnotator('P1', records, labels)


def test_noitom_notebook_save_edit_delete_and_undo(editor):
    assert editor.motion.path.name == 'noitom.bvh'
    assert editor.end.value == 1.
    editor.start.value, editor.end.value = 1.6, 1.9
    editor.label.value = 'NEUTRAL'
    assert editor.save()
    saved = next(i for i in editor.document['intervals'] if i['label'] == 'NEUTRAL')
    assert saved['annotation_source'] == 'noitom'
    assert len(editor.document['intervals']) == 4
    assert editor.path.with_suffix('.json.bak').exists()
    assert editor.undo()
    assert len(editor.document['intervals']) == 3
    first = editor.document['intervals'][0]
    editor.saved.value = first['id']
    editor.label.value = 'В'
    assert editor.save()
    assert editor.document['intervals'][0]['label'] == 'В'
    assert editor.delete()
    assert len(editor.document['intervals']) == 2
    assert editor.undo()
    assert len(editor.document['intervals']) == 3
    assert editor.load('P2')
    assert editor.load('P1')
    assert editor.document['intervals'][0]['label'] == 'В'


def test_notebook_rejects_overlapping_interval(editor):
    editor.start.value, editor.end.value = .2, 1.
    editor.label.value = 'Б'
    assert not editor.save()
    assert len(editor.document['intervals']) == 3
    assert 'overlap' in editor.status.value


def test_noitom_only_view_retains_both_hashes(tmp_path, monkeypatch):
    pytest.importorskip('ipywidgets')
    from scripts.ER2026.processing.notebook_annotation import NoitomAnnotator
    from scripts.ER2026.processing.io import indexed_records, write_csv
    from scripts.ER2026.processing.annotations import read_document
    from scripts.ER2026.processing.motion import load_bvh

    records, _ = make_dataset(tmp_path)
    write_motion(records / 'P1/custom.bvh', frames=201)
    rows = list(indexed_records(records))
    fields = ('timestamp', 'subject_id', 'word_set_id', 'custom_bvh_path', 'noitom_bvh_path')
    rows = [{k: r[k] for k in fields} for r in rows]
    rows[0]['custom_bvh_path'] = 'P1/custom.bvh'
    write_csv(records / 'records.csv', rows)
    labels = tmp_path / 'notebook_labels'
    monkeypatch.setattr(NoitomAnnotator, '_draw', lambda *args: None)
    widget = NoitomAnnotator('P1', records, labels)
    assert set(widget.document['sources']) == {'noitom', 'custom'}
    widget.start.value, widget.end.value = .1, .9
    widget.label.value = 'А'
    assert widget.save()
    row = list(indexed_records(records))[0]
    motions = {s: load_bvh(p) for s, p in row['paths'].items()}
    assert read_document(widget.path, row, motions)['intervals'][0]['label'] == 'А'
