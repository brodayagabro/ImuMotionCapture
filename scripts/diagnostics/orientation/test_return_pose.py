import base64
import json

import numpy as np
import pytest

pytest.importorskip('scipy')
from scipy.spatial.transform import Rotation
from scripts.diagnostics.orientation.return_pose_check import inspect_return


def test_raw_packet_check_handles_quaternion_sign_and_duplicate_timestamps(tmp_path):
    session = tmp_path / 'session'
    session.mkdir()
    (session / 'metadata.json').write_text(json.dumps({'recording_t0_ns': 0}))
    mapping = {'shoulder.R': 0, 'forearm.R': 1, 'forearm.L': 6, 'shoulder.L': 7}
    frames, packets, noitom = [], [], []
    for phase in range(2):
        for i in range(20):
            # Two different frames arrive at exactly the same host timestamp.
            stamp = (phase * 2_000 + (i // 2) * 50) * 1_000_000
            header = f'FRAME {phase * 20 + i} {stamp} 4'
            raw = {}
            for segment, sid in mapping.items():
                swing = Rotation.from_euler('x', 40.3 * phase if segment == 'forearm.R' else 0., degrees=True)
                twist = Rotation.from_euler('y', 2. * i + phase * 60., degrees=True)
                q = (swing * twist).as_quat()[[3, 0, 1, 2]]
                raw[str(sid)] = (q * (-1 if i % 2 else 1)).tolist()
            payload = header + '\n' + '\n'.join('Q ' + sid + ' ' + ' '.join(map(str, q)) for sid, q in raw.items())
            frames.append({'host_timestamp_ns': stamp, 'frame_header': header,
                           'sensor_id_mode': 'tca_channel', 'sensor_mapping': mapping,
                           'raw_sensor_quaternions': raw})
            packets.append({'host_timestamp_ns': stamp, 'payload_base64': base64.b64encode(payload.encode()).decode()})
            noitom.append({'host_timestamp_ns': stamp,
                           'segment_orientations': {s: [1., 0., 0., 0.] for s in mapping}})
    for filename, rows in [('own_packets', packets), ('own_frames', frames), ('noitom_frames', noitom)]:
        (session / (filename + '.jsonl')).write_text('\n'.join(json.dumps(r) for r in rows), encoding='utf-8')
    result = inspect_return(session, 0., 2., 1.)
    assert result['packet_to_parsed_comparisons'] == 160
    assert result['packet_to_parsed_max_difference_deg'] < 1e-10
    assert result['segments']['forearm.R']['own_raw_y']['angle_deg'] == pytest.approx(40.3)
    assert result['segments']['forearm.R']['noitom_bone']['angle_deg'] == pytest.approx(0.)
    assert result['segments']['forearm.L']['own_raw_y']['angle_deg'] < 1e-10
    assert result['segments']['forearm.R']['own_raw_y']['sample_counts'] == [20, 20]
    with pytest.raises(ValueError, match='non-overlapping'):
        inspect_return(session, 0., .5, 1.)
