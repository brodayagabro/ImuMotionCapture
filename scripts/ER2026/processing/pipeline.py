"""Read-only dataset audit and unshifted pair comparison."""
import json
from dataclasses import replace
from pathlib import Path
import numpy as np

from .comparison import angle_degrees, compare, time_metrics
from .io import atomic_json, indexed_records, write_csv
from .motion import ARMS, load_bvh
from .annotations import EXCLUDED, read_document


def run(root, output, labels=None):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output == root or output.is_relative_to(root):
        raise ValueError('Analysis output must be outside source records')
    output.mkdir(parents=True, exist_ok=True)
    if labels is not None and not Path(labels).is_dir():
        raise ValueError('Annotation directory does not exist; run annotate first')
    records = list(indexed_records(root))
    indexed = {p for row in records for p in row['paths'].values()}
    logs, log_errors = {}, []
    if (root / 'application.log').exists():
        for number, line in enumerate((root / 'application.log').read_text(encoding='utf-8-sig').splitlines(), 1):
            try:
                entry = json.loads(line)
                if entry.get('status') in ('complete', 'recording', 'failed', 'incomplete'):
                    # Historical absolute paths are used only for their directory name.
                    name = entry.get('path', '').replace('\\', '/').rstrip('/').split('/')[-1]
                    logs.setdefault(name, []).append(entry)
            except (ValueError, AttributeError):
                log_errors.append(number)
    inventory, metrics, temporal, diagnostics = [], [], [], {}
    interval_metrics, annotation_errors = [], {}
    for row in records:
        rid, motions = row['recording_id'], {}
        for source, path in row['paths'].items():
            item = dict(recording_id=rid, subject_id=row['subject_id'], word_set_id=row['word_set_id'],
                        source=source, path=path.relative_to(root).as_posix(), valid=False,
                        error='', warnings=[])
            try:
                motion = load_bvh(path)
                motions[source] = motion
                step = angle_degrees(motion.rotations[:-1], motion.rotations[1:])
                item.update(valid=True, frames=motion.count, fps=1 / motion.dt,
                            duration_s=motion.duration, sha256=motion.sha256,
                            exact_static_step_fraction=float(np.mean(np.max(step, axis=1) < 1e-5)),
                            max_rotation_step_deg=float(step.max()),
                            body_direction_std=float(motion.directions().std(axis=0).mean()))
                if item['exact_static_step_fraction'] > .2:
                    item['warnings'].append('Many identical adjacent poses; inspect source cadence/freeze')
                if item['body_direction_std'] < .02:
                    item['warnings'].append('Very little arm movement')
                if item['max_rotation_step_deg'] > 90:
                    item['warnings'].append('Large rotation jump; inspect calibration or tracking')
            except (OSError, ValueError, IndexError) as error:
                item['error'] = str(error)
            inventory.append(item)
        if set(motions) == {'custom', 'noitom'}:
            try:
                pair_rows, lag = compare(motions['custom'], motions['noitom'], rid)
                metrics.extend(pair_rows)
                temporal.extend(time_metrics(motions['custom'], motions['noitom'], rid))
                diagnostics[rid] = lag
                label_path = Path(labels) / (rid + '.json') if labels is not None else None
                if label_path is not None and label_path.exists():
                    try:
                        document = read_document(label_path, row, motions)
                        for item in document['intervals']:
                            if not item['reviewed'] or item['label'] in EXCLUDED:
                                continue
                            a = motions['custom']
                            first = int(np.ceil(item['start_s'] / a.dt - 1e-8))
                            end = int(np.ceil(item['end_s'] / a.dt - 1e-8))
                            if end - first < 3:
                                continue
                            parts = [replace(motions[s], rotations=motions[s].rotations[first:end],
                                             positions=motions[s].positions[first:end])
                                     for s in ('custom', 'noitom')]
                            entries, _ = compare(*parts, rid)
                            interval_metrics.extend(dict(entry, interval_id=item['id'], label=item['label'],
                                start_s=item['start_s'], end_s=item['end_s'], frames=end - first) for entry in entries)
                    except (OSError, ValueError, KeyError, TypeError) as error:
                        annotation_errors[rid] = str(error)
            except ValueError as error:
                diagnostics[rid] = {'error': str(error)}
    orphans = [p.relative_to(root).as_posix() for p in root.glob('*/*.bvh') if p.resolve() not in indexed]
    legacy = dict(index_present=False)
    legacy_index = root / 'semaphore/dataset/dataset_index.csv'
    if legacy_index.exists():
        import csv
        with legacy_index.open(encoding='utf-8-sig') as stream:
            legacy_rows = list(csv.DictReader(stream))
        paths = {str((legacy_index.parent / row[field]).resolve())
                 for row in legacy_rows for field in ('own_frames_file', 'noitom_frames_file') if row.get(field)}
        legacy = dict(index_present=True, indexed_presenter_sessions=len(legacy_rows),
                      referenced_motion_files=len(paths), existing_motion_files=sum(Path(p).exists() for p in paths),
                      note='Historical index presence flags may be stale; not merged into BVH dataset')
    report = dict(schema_version=2, indexed_recordings=len(records),
                  indexed_pairs=sum(len(r['paths']) == 2 for r in records),
                  successful_comparisons=sum('error' not in v for v in diagnostics.values()),
                  inventory=inventory, lag_diagnostics=diagnostics, unindexed_bvh=orphans,
                  empty_recording_dirs=[p.name for p in root.iterdir() if p.is_dir() and
                                        p.name != 'semaphore' and not any(p.glob('*.bvh'))],
                  log_errors=log_errors,
                  legacy_archive=legacy,
                  annotation_metrics=dict(requested=labels is not None, errors=annotation_errors,
                      intervals=len({(r['recording_id'], r['interval_id']) for r in interval_metrics})),
                  acquisition={rid: [dict(status=e['status'], export=e.get('export'),
                                         error=e.get('error'), synthetic=e.get('noitom', {}).get('configuration', {}).get('fake'))
                                    for e in entries] for rid, entries in logs.items()},
                  scope='motion capture only; records.csv; historical semaphore archive is not auto-joined',
                  metrics='all frames, no latency correction; spine alignment per frame; units metres/degrees')
    atomic_json(output / 'audit.json', report)
    write_csv(output / 'inventory.csv', inventory)
    write_csv(output / 'comparison.csv', metrics)
    write_csv(output / 'comparison_10s.csv', temporal)
    interval_fields = list(dict.fromkeys(['recording_id', 'interval_id', 'label', 'start_s', 'end_s', 'frames'] +
                                        [k for r in interval_metrics for k in r]))
    write_csv(output / 'comparison_intervals.csv', interval_metrics, interval_fields)
    lines = ['# ER2026: аудит и сравнение', '',
             f"Записей: {len(records)}; пар: {report['indexed_pairs']}; рассчитано пар: {report['successful_comparisons']}.", '',
             'Сравнение на исходной общей сетке, без подбора задержки. Все кадры, включая переходы.',
             'Спина выравнивается заново в каждом кадре: R_spine.T @ R_segment; начало — Spine.',
             'Это измеряет руки относительно туловища и исключает ошибку самой спины из aligned-метрик.',
             'Исходная ошибка спины сохранена отдельно в world. Ноги/голова не оцениваются.', '',
             '| Запись | Сегмент | RMS вращения, ° | P95, ° |',
             '|---|---|---:|---:|']
    for item in metrics:
        if item['kind'] == 'rotation_deg' and item['frame'] == 'spine' and item['joint'] in ARMS:
            lines.append(f"| {item['recording_id']} | {item['joint']} | {item['rms']:.2f} | {item['p95']:.2f} |")
    lines += ['', 'Полные world/spine, position_mm, direction_deg и Pearson по осям: `comparison.csv`.',
              'RMS положения = sqrt(mean(||p_custom − p_noitom||²)); Pearson постоянного сигнала пустой.',
              'Угловая ошибка = acos((trace(R_custom.T @ R_noitom) − 1)/2), диапазон 0…180°.', '',
              'frame=parent отдельно оценивает локальные вращения плеча/локтя. elbow_flexion_deg —',
              'угол между плечом и предплечьем (0° = выпрямленная рука), RMS разности, bias Custom−Noitom и Pearson.', '',
              f'Неиндексированные BVH: {orphans}. Они не включены в основные результаты.',
              f'Архив старого протокола: {legacy}.',
              'Не усредняйте записи как независимых участников: ID с суффиксом _mio могут относиться к тому же человеку.',
              'Noitom — система сравнения; эти ошибки сами по себе не доказывают абсолютную точность.', '']
    for rid, item in diagnostics.items():
        if 'error' not in item:
            score = item['speed_correlation']
            text = f'{score:.3f}' if score is not None else 'не определена (недостаточно движения)'
            lines.append(f"Диагностика задержки {rid}: {item['best_lag_s']} с, корреляция скоростей {text}; сдвиг не применён.")
    lines += ['', f"Пустые каталоги без BVH: {report['empty_recording_dirs']}.", '',
              '## Проверка формата', '']
    lines += [f"- {i['recording_id']} / {i['source']}: {i['error']}" for i in inventory if not i['valid']]
    lines += ['', f"Метрики ручных интервалов: {report['annotation_metrics']}; comparison_intervals.csv."]
    (output / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    return report
