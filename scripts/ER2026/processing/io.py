"""Small atomic outputs and safe, index-relative recording paths."""
import csv
import json
import os
from pathlib import Path
import tempfile


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent, delete=False) as stream:
            name = stream.name
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
        os.replace(name, path)
    finally:
        if name and Path(name).exists():
            Path(name).unlink()


def write_csv(path, rows, fields=None):
    rows = list(rows)
    fields = fields or list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def indexed_records(root):
    root = Path(root).resolve()
    with (root / 'records.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    seen = set()
    for row in rows:
        paths = {}
        for source in ('custom', 'noitom'):
            relative = row[f'{source}_bvh_path']
            if not relative:
                continue
            path = (root / relative).resolve()
            if not path.is_relative_to(root) or Path(relative).is_absolute():
                raise ValueError(f'Path outside records: {relative}')
            paths[source] = path
        ids = {p.parent.name for p in paths.values()}
        if len(ids) != 1:
            raise ValueError('An index row must reference exactly one recording directory')
        recording_id = ids.pop()
        if recording_id in seen:
            raise ValueError(f'Duplicate recording ID: {recording_id}')
        seen.add(recording_id)
        yield dict(row, recording_id=recording_id, paths=paths)
