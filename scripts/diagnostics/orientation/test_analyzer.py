import json
import math
import numpy as np
import pytest
from pyqt_mocap.calibration import profile_document
from pyqt_mocap.t_pose_calibration import calibrate_t_pose
from pyqt_mocap.tests.test_t_pose_calibration import physical_captures
from pyqt_mocap.mocap_core import SEGMENT_NAMES, DEFAULT_SENSOR_MAPPING
from pyqt_mocap.quaternion_utils import compose,inverse,quaternion_from_rotation_vector,angular_distance
from scripts.diagnostics.orientation.calibration_analyzer import (
    analyze_profile, repeatability, compare_streams, compare_fixed_interval, rejected_calibrations, run,
)


def test_residuals_and_five_repeatability_do_not_confuse_quaternion_sign():
    profile=profile_document({'enabled_segments':list(SEGMENT_NAMES)},calibrate_t_pose(physical_captures()))
    result=analyze_profile(profile)
    assert max(r['axis_direction_error_deg'] for r in result['residuals']) < 2e-6
    entries=[]
    for i in range(5):
        copy=json.loads(json.dumps(profile))
        if i%2:
            copy['calibration']['axis_alignment_quaternions_wxyz']={s:[-x for x in q] for s,q in copy['calibration']['axis_alignment_quaternions_wxyz'].items()}
        entries.append({'profile':copy})
    for row in repeatability(entries):
        assert row['minimum_five_met']
        assert row['world_alignment_max_deviation_deg'] < 1e-10
        assert row['mounting_max_deviation_deg'] < 1e-10


def test_fixed_world_and_segment_alignment_recovers_global_and_elbow_rotations():
    world=quaternion_from_rotation_vector((0,0,.7))
    offsets={s:quaternion_from_rotation_vector((.1*i,.2,.3)) for i,s in enumerate(SEGMENT_NAMES)}
    own,noitom=[],[]
    for i in range(80):
        oq={s:quaternion_from_rotation_vector((0,0,0) if i<20 else (.2+.02*i,.1*j,-.1)) for j,s in enumerate(SEGMENT_NAMES)}
        nq={s:compose(world,compose(q,inverse(offsets[s]))) for s,q in oq.items()}
        stamp=i*50_000_000
        own.append({'t':stamp,'q':oq,'enabled':SEGMENT_NAMES})
        noitom.append({'t':stamp+1_000_000,'q':nq,'enabled':SEGMENT_NAMES,'torso_heading_rad':.7,'heading_offset_rad':0.})
    report,errors=compare_streams(own,noitom,0,1.)
    assert report['status']=='aligned'
    assert max(r['angular_error_deg'] for r in errors)<1e-10
    assert {r['segment'] for r in errors if r['kind']=='joint'}=={'elbow.L','elbow.R'}
    assert all(r['timestamp_ns']>=1e9 for r in errors)
    assert angular_distance(report['q_world_alignment'],inverse(world))<1e-10


def test_profile_cli_artifacts_and_multipose_candidate(tmp_path):
    pytest.importorskip('scipy')
    profile=profile_document({},calibrate_t_pose(physical_captures()))
    source=tmp_path/'profile.json'
    source.write_text(json.dumps(profile),encoding='utf-8')
    before=source.read_bytes()
    report=run(profiles=[source],output=tmp_path/'results',optimize=True)
    assert source.read_bytes()==before
    assert report['profile_count']==1
    assert report['profiles'][0]['multipose_candidates']['forearm.L']['success']
    assert (tmp_path/'results'/'pose_residuals.csv').exists()
    assert not any(r['minimum_five_met'] for r in report['repeatability'])


def test_comparison_does_not_cross_reference_changes():
    q = {s: np.array([1., 0., 0., 0.]) for s in SEGMENT_NAMES}
    own = [{'t': i * 50_000_000, 'q': q, 'enabled': SEGMENT_NAMES,
            'raw': {}, 'mapping': DEFAULT_SENSOR_MAPPING} for i in range(80)]
    noitom = [{**r, 'torso_heading_rad': 0., 'heading_offset_rad': 0.} for r in own]
    events = [{'event': 'own_calibration_state', 'host_timestamp_ns': 2_000_000_000}]
    report, errors, _ = compare_fixed_interval(own, noitom, events, 0, 1., 3.)
    assert report['status'] == 'aligned'
    assert report['valid_until_ns'] == 2_000_000_000
    assert errors and all(1e9 <= r['timestamp_ns'] < 2e9 for r in errors)
    events[0]['host_timestamp_ns'] = 500_000_000
    report, errors, drift = compare_fixed_interval(own, noitom, events, 0, 1., 3.)
    assert report['status'] == 'reference_changed_during_capture'
    assert not errors and not drift


def test_failed_attempt_is_reported_without_counting_it_as_applied_profile(tmp_path):
    import csv
    event = {'event': 'own_calibration_fit_rejected', 'host_timestamp_ns': 200_000_000,
             'details': {'error': 'return mismatch', 'closure_error_deg': {'forearm.R': 40.3},
                         'sensor_mapping': {'forearm.R': 1},
                         'poses': {'a_pose': {'started_s': .1, 'ended_s': .15},
                                   'return_a_pose': {'started_s': .17, 'ended_s': .2}}}}
    row, = rejected_calibrations([event])
    assert not row['applied'] and row['sensor_id'] == 1 and row['closure_error_deg'] == 40.3
    session = tmp_path / 'session'
    session.mkdir()
    (session / 'metadata.json').write_text(json.dumps({'recording_t0_ns': 0}), encoding='utf-8')
    with (session / 'events.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['event', 'host_timestamp_ns', 'details_json'])
        writer.writerow([event['event'], event['host_timestamp_ns'], json.dumps(event['details'])])
    report = run(session=session, output=tmp_path/'analysis')
    assert report['profile_count'] == 0 and not report['repeatability']
    assert report['rejected_calibrations'] == [row]
    assert (tmp_path/'analysis'/'rejected_calibrations.csv').exists()
