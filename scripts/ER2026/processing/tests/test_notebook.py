"""Regression checks execute notebook code against representative audit tables."""
import json
from pathlib import Path

import pytest

from scripts.ER2026.processing.io import indexed_records, write_csv
from scripts.ER2026.processing.motion import ARMS, POINTS
from scripts.ER2026.processing.pipeline import run
from .test_processing import make_dataset, write_motion


@pytest.fixture
def notebook_session(tmp_path):
    pd = pytest.importorskip('pandas')
    records_root, labels_root = make_dataset(tmp_path)
    write_motion(records_root / 'P1/custom.bvh', frames=201)
    fields = ('timestamp', 'subject_id', 'word_set_id', 'custom_bvh_path', 'noitom_bvh_path')
    rows = [{key: row[key] for key in fields} for row in indexed_records(records_root)]
    rows[0]['custom_bvh_path'] = 'P1/custom.bvh'
    write_csv(records_root / 'records.csv', rows)
    output = tmp_path / 'analysis'
    notebook_path = Path(__file__).resolve().parents[1] / 'motion_capture_analysis.ipynb'
    notebook = json.loads(notebook_path.read_text(encoding='utf-8'))
    sources = [''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type'] == 'code']
    namespace = dict(pd=pd, Markdown=lambda value: value,
                     display=lambda value: None, print=lambda *args, **kwargs: None,
                     RECORDS_ROOT=records_root, LABELS_ROOT=labels_root, OUTPUT_DIR=output,
                     ARMS=ARMS, POINTS=POINTS, run=run, indexed_records=indexed_records)

    def execute(prefix):
        source = next(source for source in sources if source.startswith(prefix))
        exec(compile(source, str(notebook_path), 'exec'), namespace)

    return namespace, execute


@pytest.mark.parametrize('case', ['all_valid', 'mixed', 'legacy_cache'])
def test_notebook_audit_with_and_without_file_errors(notebook_session, case):
    namespace, execute = notebook_session
    if case == 'mixed':
        (namespace['RECORDS_ROOT'] / 'P2/noitom.bvh').write_text('invalid', encoding='ascii')
    execute('RECOMPUTE =')
    assert 'error' in namespace['inventory'].columns
    if case == 'legacy_cache':
        # Previously written, successful audits omitted this optional column.
        namespace['inventory'] = namespace['inventory'].drop(columns='error')
    execute('invalid =')
    assert namespace['pair_ids'] == ['P1']
    assert namespace['SELECTED_RECORDING'] == 'P1'
    assert len(namespace['invalid']) == (1 if case == 'mixed' else 0)


@pytest.mark.parametrize('prefix, image', [
    ('rotation =', '01_rotation_rms.png'),
    ('position =', '02_position_rms.png'),
    ('direction_rows =', '03_direction_correlations.png'),
    ('temporal_ids =', '04_error_over_time.png'),
])
def test_plots_do_not_require_diagnostic_cell_or_pair_ids(notebook_session, prefix, image):
    import numpy as np
    matplotlib = pytest.importorskip('matplotlib')
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    namespace, execute = notebook_session
    namespace.update(np=np, plt=plt)
    execute('RECOMPUTE =')
    # Reproduce an existing kernel where tables exist but pair_ids was never set.
    assert 'invalid' not in namespace
    assert namespace['pair_ids'] == ['P1']
    assert namespace['SELECTED_RECORDING'] == 'P1'
    del namespace['pair_ids']
    try:
        execute(prefix)
        assert (namespace['OUTPUT_DIR'] / image).is_file()
        assert 'pair_ids' not in namespace
    finally:
        plt.close('all')
