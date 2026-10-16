"""Verify A-return discrepancies directly from saved UDP packets, using SciPy.

Window times are seconds from REC, manually selected unless wizard events exist.
Only within-stream changes are compared; these are not cross-suit pose errors.
"""
import argparse
import base64
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from pyqt_mocap.mocap_core import RAW_SENSOR_ID_PROFILES
from .calibration_analyzer import read_json


def inspect_return(session, a_start, return_start, duration=5.):
    session = Path(session)
    if not all(math.isfinite(v) and v >= 0 for v in (a_start, return_start)) or not math.isfinite(duration) or duration <= 0:
        raise ValueError('Window starts must be nonnegative; duration must be positive')
    windows = [(a_start, a_start + duration), (return_start, return_start + duration)]
    if windows[0][1] > windows[1][0]:
        raise ValueError('A-return windows must be ordered and non-overlapping')
    metadata = read_json(session / 'metadata.json')
    t0 = metadata['recording_t0_ns']

    def window(row):
        seconds = (row['host_timestamp_ns'] - t0) / 1e9
        return next((i for i, (start, end) in enumerate(windows) if start <= seconds < end), None)

    frames, config = {}, None
    with (session / 'own_frames.jsonl').open(encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            if window(row) is None:
                continue
            current = (row['sensor_id_mode'], row['sensor_mapping'])
            if config is not None and config != current:
                raise ValueError('Sensor mapping changed across the selected windows')
            config = current
            frames[row['host_timestamp_ns'], row.get('frame_header')] = row['raw_sensor_quaternions']
    if config is None:
        raise ValueError('No own frames in the selected windows')
    mode, mapping = config
    id_map = RAW_SENSOR_ID_PROFILES[mode] if mode != 'raw' else None
    arms = [s for s in mapping if s.startswith(('shoulder.', 'forearm.'))]
    packet_values = [{s: [] for s in arms} for _ in windows]
    parse_differences = []
    with (session / 'own_packets.jsonl').open(encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            i = window(row)
            if i is None:
                continue
            payload = base64.b64decode(row['payload_base64']).decode('utf-8')
            header = next((line.strip() for line in payload.splitlines() if line.startswith('FRAME ')), None)
            for record in payload.splitlines():
                parts = record.split()
                if len(parts) != 6 or parts[0].upper() not in ('Q', 'QUAT'):
                    continue
                sid = int(parts[1])
                canonical_id = sid if id_map is None else id_map.get(sid)
                q = np.array([float(v) for v in parts[2:]])
                rotation = Rotation.from_quat(q[[1, 2, 3, 0]])
                for s in arms:
                    if mapping[s] == canonical_id:
                        packet_values[i][s].append(q)
                parsed = frames.get((row['host_timestamp_ns'], header), {}).get(str(canonical_id))
                if parsed is not None:
                    parsed_rotation = Rotation.from_quat(np.array(parsed)[[1, 2, 3, 0]])
                    parse_differences.append(math.degrees((rotation.inv() * parsed_rotation).magnitude()))

    noitom_values = [{s: [] for s in arms} for _ in windows]
    headings = set()
    with (session / 'noitom_frames.jsonl').open(encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            i = window(row)
            if i is None or row.get('mapping_error'):
                continue
            headings.add(row.get('heading_offset_rad', 0.))
            for s in arms:
                if s in row.get('segment_orientations', {}) and s in row.get('enabled_segments', arms):
                    noitom_values[i][s].append(row['segment_orientations'][s])

    def direction_change(values, axis):
        if min(map(len, values)) < 15:
            return None
        means = [Rotation.from_quat(np.array(q)[:, [1, 2, 3, 0]]).mean() for q in values]
        directions = [r.apply(axis) for r in means]
        angle = math.degrees(math.atan2(np.linalg.norm(np.cross(*directions)), np.dot(*directions)))
        return {'angle_deg': angle, 'sample_counts': list(map(len, values)),
                'mean_directions': [v.tolist() for v in directions],
                'orientation_change_deg': math.degrees((means[0].inv() * means[1]).magnitude())}

    return {
        'session': str(session), 'windows_seconds_from_rec': windows,
        'windows_source': 'manually selected; exact wizard intervals unavailable in older recordings',
        'meaning': 'within-stream direction changes, not aligned cross-suit accuracy',
        'packet_to_parsed_comparisons': len(parse_differences),
        'packet_to_parsed_max_difference_deg': max(parse_differences) if parse_differences else None,
        'segments': {s: {
            'canonical_sensor_id': mapping[s],
            'own_raw_y': direction_change([v[s] for v in packet_values], (0., 1. if s.endswith('.L') else -1., 0.)),
            'noitom_bone': direction_change([v[s] for v in noitom_values], (0., 0., -1.)) if len(headings) <= 1 else None,
        } for s in arms},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--a-start', type=float, required=True)
    parser.add_argument('--return-start', type=float, required=True)
    parser.add_argument('--duration', type=float, default=5.)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() == args.session.resolve() or args.session.resolve() in args.output.resolve().parents:
        parser.error('Output must be outside the source recording')
    report = inspect_return(args.session, args.a_start, args.return_start, args.duration)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
