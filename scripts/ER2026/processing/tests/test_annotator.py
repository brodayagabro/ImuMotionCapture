import pytest

pytestmark = pytest.mark.gui


@pytest.mark.parametrize('paired', [False, True])
@pytest.mark.parametrize('beside_bvh', [False, True])
def test_manual_review_save_reload_edit_undo(tmp_path, paired, beside_bvh, monkeypatch):
    from PyQt6.QtWidgets import QApplication
    from scripts.ER2026.processing.annotator import Annotator
    from scripts.ER2026.processing.annotations import new_document, save_document
    from scripts.ER2026.processing.io import indexed_records, write_csv
    from scripts.ER2026.processing.motion import load_bvh
    from .test_processing import make_dataset, write_motion

    app = QApplication.instance() or QApplication([])
    records, labels = make_dataset(tmp_path, beside_bvh=beside_bvh)
    if paired:
        write_motion(records / 'P1/custom.bvh', frames=201)
        fields = ('timestamp', 'subject_id', 'word_set_id', 'custom_bvh_path', 'noitom_bvh_path')
        rows = [{key: row[key] for key in fields} for row in indexed_records(records)]
        rows[0]['custom_bvh_path'] = 'P1/custom.bvh'
        # Custom-only recordings must not be offered for Noitom review.
        rows.append(dict(rows[0], subject_id='P3', custom_bvh_path='P3/custom.bvh', noitom_bvh_path=''))
        write_csv(records / 'records.csv', rows)
        row = next(indexed_records(records))
        motions = {source: load_bvh(path) for source, path in row['paths'].items()}
        import json
        path = (records / 'P1' if beside_bvh else labels) / 'P1.json'
        document = new_document(row, motions)
        document['intervals'] = json.loads(path.read_text(encoding='utf-8'))['intervals']
        save_document(path, document, motions['noitom'].duration)
    before = {p: p.read_bytes() for p in records.glob('*/*.bvh')}
    monkeypatch.chdir(tmp_path)
    window = Annotator(records) if beside_bvh else Annotator(records, labels)
    window.show()
    app.processEvents()
    assert window.motion.count == 201
    assert window.end.value() == 1.
    assert window.motion is window.motions['noitom']
    assert len(window.records) == 2
    assert set(window.lines) == {'noitom'}
    assert all(len(view.listDataItems()) == 1 for view in window.views)
    assert all(len(curve.getData()[0]) > 0 for curve in window.lines['noitom'])
    assert set(window.document['sources']) == ({'custom', 'noitom'} if paired else {'noitom'})
    window.new_interval()
    window.region.setRegion((1.6, 1.9))
    window.label.setCurrentText('NEUTRAL')
    window.commit()
    assert len(window.document['intervals']) == 4
    saved = next(item for item in window.document['intervals'] if item['label'] == 'NEUTRAL')
    assert saved['annotation_source'] == 'noitom'
    assert saved['annotation_tool'] == 'qt'
    assert window.path.exists()
    assert window.path == (records / 'P1' if beside_bvh else labels) / 'P1.json'
    window.undo_change()
    assert len(window.document['intervals']) == 3
    window.select_interval(0)
    window.label.setCurrentText('В')
    window.commit()
    assert window.document['intervals'][0]['label'] == 'В'
    window.record.setCurrentIndex(1)
    window.record.setCurrentIndex(0)
    assert window.document['intervals'][0]['label'] == 'В'
    assert window.document['intervals'][0]['annotation_source'] == 'noitom'
    assert window.motion is window.motions['noitom']
    assert all(len(view.listDataItems()) == 1 for view in window.views)
    assert window.path.with_suffix('.json.bak').exists()
    window.close()
    app.processEvents()
    assert all(p.read_bytes() == original for p, original in before.items())
