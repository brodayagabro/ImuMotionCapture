from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import pytest
from pyqt_mocap.mocap_core import compute_body_pose, quaternion_to_matrix
from scripts.Visual_IMU_compare.src.visual_protocol import BvhDecoder, BVH_TO_PROJECT, MAX_MESSAGE_BYTES
from scripts.Visual_IMU_compare.src.visual_adapter import VisualAdapter, world_transforms


def decode(pair):
    decoder = BvhDecoder()
    assert decoder.decode(json.dumps(pair['skeleton']).encode(), 100) is None
    return decoder, decoder.decode(json.dumps(pair['frame']).encode(), 123)


@pytest.mark.parametrize('units', ['m', 'cm'])
def test_real_server_fixture_axes_units_and_all_five_bone_directions(units):
    pair = json.loads((Path(__file__).parent / f'fixtures/visual_{units}.json').read_text())
    decoder, frame = decode(pair)
    assert len(frame.joints) == 16
    assert frame.host_timestamp_ns == 123 and frame.source_timestamp == 10.
    points = np.asarray(pair['points']) @ np.diag([1., -1., -1.]) @ BVH_TO_PROJECT.T
    worlds = world_transforms(frame)
    np.testing.assert_allclose(worlds['Hips'][1], (points[11] + points[12]) / 2, atol=2e-6)
    orientations = VisualAdapter().orientations(frame)
    rest = compute_body_pose({}).tracked_segments
    directions = {'spine': (points[5]+points[6]-points[11]-points[12])/2,
                  'shoulder.L': points[7]-points[5], 'forearm.L': points[9]-points[7],
                  'shoulder.R': points[8]-points[6], 'forearm.R': points[10]-points[8]}
    for segment, expected in directions.items():
        start, end = rest[segment]
        actual = quaternion_to_matrix(orientations[segment]) @ (end-start)
        np.testing.assert_allclose(actual/np.linalg.norm(actual), expected/np.linalg.norm(expected), atol=2e-6)


def test_duplicate_reorder_missing_header_and_stream_restart(packet_pair):
    pair = packet_pair
    decoder = BvhDecoder()
    assert decoder.decode(json.dumps(pair['frame']).encode(), 1) is None
    decoder, first = decode(pair)
    assert decoder.decode(json.dumps(pair['skeleton']).encode(), 124) is None
    assert decoder.decode(json.dumps(pair['frame']).encode(), 125) is None
    pair['frame']['frame_id'] = 0
    assert decoder.decode(json.dumps(pair['frame']).encode(), 126) is None
    pair['skeleton']['stream_id'] = pair['frame']['stream_id'] = 'restarted'
    decoder.decode(json.dumps(pair['skeleton']).encode(), 127)
    assert decoder.decode(json.dumps(pair['frame']).encode(), 128).stream_id == 'restarted'


@pytest.mark.parametrize('mutation', ['hash', 'channels', 'units', 'axes', 'hierarchy', 'nonfinite', 'mask', 'time'])
def test_invalid_payloads_rejected(packet_pair, mutation):
    decoder, _ = decode(packet_pair)
    if mutation in {'hash','channels','units','axes','hierarchy'}:
        header = deepcopy(packet_pair['skeleton'])
        key, value = {'hash': ('skeleton_id','bad'), 'channels': ('channel_names',[]),
                      'units': ('units','inch'), 'axes': ('axes','left-handed'),
                      'hierarchy': ('hierarchy','HIERARCHY ROOT Hips {')}[mutation]
        header[key] = value
        payload = header
    else:
        payload = deepcopy(packet_pair['frame'])
        payload['frame_id'] += 1
        payload['timestamp_monotonic_s'] += .1
        if mutation == 'nonfinite': payload['channels'][0] = float('nan')
        if mutation == 'mask': payload['tracked'] = [1] * 16
        if mutation == 'time': payload['timestamp_monotonic_s'] = 1
    with pytest.raises(ValueError):
        decoder.decode(json.dumps(payload).encode(), 200)


def test_oversized_and_untracked_input(packet_pair):
    with pytest.raises(ValueError, match='too large'):
        BvhDecoder().decode(b' ' * (MAX_MESSAGE_BYTES+1), 0)
    index = packet_pair['skeleton']['joint_names'].index('LeftForeArm')
    packet_pair['frame']['tracked'][index] = False
    _, frame = decode(packet_pair)
    with pytest.raises(ValueError, match='untracked'):
        VisualAdapter().orientations(frame)
