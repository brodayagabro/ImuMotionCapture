"""Portable reviewed mocap intervals with explicit time coordinates."""
from datetime import datetime, timedelta
from pathlib import Path

from .annotations import EXCLUDED, annotation_path, read_document
from .io import indexed_records, write_csv
from .motion import load_bvh


def export_labels(records, labels, output):
    """Export reviewed labels and wall time hints, preserving half-open intervals.

    Wall time is a lookup hint for external recordings, never a claim that
    sensor clocks/latencies are synchronized.
    """
    root, output = Path(records).resolve(), Path(output).resolve()
    if output == root or output.is_relative_to(root):
        raise ValueError('Label export must be outside source records')
    rows = []
    for row in indexed_records(root):
        path = annotation_path(row, labels)
        if not path.exists():
            continue
        # Check hashes before exporting labels for downstream work.
        motions = {name: load_bvh(p) for name, p in row['paths'].items()}
        document = read_document(path, row, motions)
        start = datetime.fromisoformat(row['timestamp'])
        for item in document['intervals']:
            if not item['reviewed']:
                continue
            rows.append(dict(recording_id=row['recording_id'], subject_id=row['subject_id'],
                word_set_id=row['word_set_id'], interval_id=item['id'],
                label=item['label'], usable_pose=item['label'] not in EXCLUDED,
                start_s=item['start_s'], end_s=item['end_s'],
                wall_start_hint=(start + timedelta(seconds=item['start_s'])).isoformat(),
                wall_end_hint=(start + timedelta(seconds=item['end_s'])).isoformat(),
                time_basis='BVH seconds; half-open [start_s,end_s)', annotation_file=path.name,
                note=item.get('annotator_note', '')))
    if not rows:
        raise ValueError('No reviewed labels to export')
    output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(output, rows)
    return rows
