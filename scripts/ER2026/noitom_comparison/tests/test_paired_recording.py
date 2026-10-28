from dataclasses import replace
from datetime import datetime
import csv
import json
import math

import numpy as np
import pytest
from scripts.ER2026.noitom_comparison.paired_recording import PairedRecorder
from scripts.ER2026.noitom_comparison.paired_bvh import INDEX_FIELDS, SEGMENTS, append_index, common_grid, slerp
from scripts.ER2026.noitom_comparison.noitom_client import FakeNoitomClient
from scripts.ER2026.noitom_comparison.synchronization import JointTransform
from pyqt_mocap.mocap_core import quaternion_from_rotation_vector
from pyqt_mocap.tests.test_bvh import read_bvh, rotation


class Clock:
    def __init__(self): self.now = 10_000_000_000
    def __call__(self): return self.now


def own(recorder, clock, index, *, complete=True):
    _, claim = recorder.stamp_packet()
    angle = (clock.now - 10_000_000_000) / 1e9 * .2
    q = quaternion_from_rotation_vector((0, angle, 0))
    recorder.complete_packet(claim, {}, {'measurement_complete': complete,
        'measurement_key': (index, index * 100), 'bvh_orientations': dict.fromkeys(SEGMENTS, q)})


def noitom(recorder, clock, index, *, invalid=False, heading=0.):
    ns, claim = recorder.stamp_noitom_packet()
    frame = FakeNoitomClient.frame(index, ns)
    # Native hierarchy contains joints which do not appear in the five-segment view.
    joints = dict(frame.joints)
    joints['LeftFinger'] = JointTransform(np.array([.08, 0, 0]), np.array([1., 0, 0, 0]),
                                         'LeftHand', np.array([.08, 0, 0]))
    if invalid: joints['Hips'] = replace(joints['Hips'], rotation=np.zeros(4))
    recorder.noitom_frame(replace(frame, joints=joints, capture_context=claim), {}, heading_offset_rad=heading)


def run_pair(tmp_path, *, duration=1.2, max_fps=100):
    clock = Clock()
    recorder = PairedRecorder(clock=clock, watchdog=False)
    session = recorder.start(tmp_path, {'subject_id': 'P001', 'word_set_id': 'SET01',
        'duration_s': duration, 'max_fps': max_fps})
    # custom 10 Hz, Noitom 20 Hz, staggered starts by 50 ms.
    for i in range(math.ceil((duration + .2) * 20) + 1):
        clock.now = 10_000_000_000 + i * 50_000_000
        if i % 2 == 0: own(recorder, clock, i)
        if i >= 1: noitom(recorder, clock, i)
        recorder.tick()
        if recorder.active is None: break
    assert session.done.wait(20), 'export did not complete'
    return session


def test_pair_common_timeline_identical_hierarchy_index_and_repeat(tmp_path):
    s = run_pair(tmp_path)
    assert s.error is None, s.error
    assert s.metadata['status'] == 'complete'
    assert {p.name for p in s.path.iterdir()} == {'custom.bvh', 'noitom.bvh'}
    a, fa, dta = read_bvh(s.path / 'custom.bvh')
    b, fb, dtb = read_bvh(s.path / 'noitom.bvh')
    assert dta == dtb and len(fa) == len(fb) == 13
    assert (len(fa)-1)*dta == pytest.approx(1.2, abs=1e-8)
    assert s.metadata['t0_ns'] == 10_050_000_000
    assert s.metadata['t1_ns'] == 11_250_000_000
    assert s.metadata['fps'] <= 10.000001
    assert len(a) == len(b) == 14 and fa.shape[1] == fb.shape[1] == 45
    assert not any('Hand' in j[0] or 'Finger' in j[0] for j in b)
    assert ((s.path/'custom.bvh').read_text().split('MOTION')[0] ==
            (s.path/'noitom.bvh').read_text().split('MOTION')[0])
    assert s.metadata['skeleton_height_m'] == 1.70
    assert s.export_progress['percent'] == 100
    # First and last custom spine are interpolated at shared T0/T1, not held.
    offset = sum(len(j[3]) for j in a[:next(i for i,j in enumerate(a) if j[0]=='Spine')])
    for row, angle in ((fa[0], .05*.2), (fa[-1], 1.25*.2)):
        z,x,y = np.radians(row[offset:offset+3])
        from storage.bvh import TO_BVH
        expected = TO_BVH @ rotation('Y', angle) @ TO_BVH.T
        np.testing.assert_allclose(rotation('Z', z) @ rotation('X', x) @ rotation('Y', y), expected, atol=1e-8)
    with (tmp_path/'records.csv').open(encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert reader.fieldnames == list(INDEX_FIELDS)
    assert len(rows) == 1 and datetime.fromisoformat(rows[0]['timestamp']).tzinfo is not None
    assert (tmp_path / rows[0]['custom_bvh_path']).is_file()
    append_index(tmp_path, rows[0])
    second = run_pair(tmp_path)
    assert second.path != s.path and (s.path/'custom.bvh').is_file()
    with (tmp_path/'records.csv').open(encoding='utf-8') as f:
        assert len(list(csv.DictReader(f))) == 2


def test_default_300_seconds_stops_at_exact_shared_deadline(tmp_path):
    clock = Clock()
    recorder = PairedRecorder(clock=clock, watchdog=False, max_gap_ms=600)
    s = recorder.start(tmp_path, {'subject_id':'P001','word_set_id':'SET02','max_fps':2})
    for i in range(601):
        clock.now = 10_000_000_000 + i*500_000_000
        own(recorder, clock, i)
        noitom(recorder, clock, i)
        recorder.tick()
        if i < 600: assert recorder.active is s
    assert recorder.active is None and s.done.wait(20)
    assert s.error is None, s.error
    assert s.stop_ns - s.t0_ns == 300_000_000_000
    assert s.metadata['frames'] == 601 and s.metadata['fps'] == 2


@pytest.mark.parametrize('failure', ['stop', 'missing', 'gap', 'invalid', 'heartbeat'])
def test_incomplete_never_creates_success_row(tmp_path, failure):
    clock = Clock()
    r = PairedRecorder(clock=clock, watchdog=False)
    s = r.start(tmp_path, {'subject_id':'P001','word_set_id':'SET01','require_heartbeat': failure=='heartbeat'})
    if failure != 'missing':
        own(r, clock, 0)
        noitom(r, clock, 0, invalid=failure=='invalid')
    if failure == 'stop': r.stop()
    else:
        clock.now += 6_000_000_000
        r.tick()
    assert s.done.wait(3)
    assert s.error and s.metadata['status'] != 'complete'
    assert not (tmp_path/'records.csv').exists()
    journal = [json.loads(l) for l in (tmp_path/'application.log').read_text(encoding='utf-8').splitlines()]
    assert journal[-1]['status'] in ('incomplete','error')


def test_export_error_does_not_append_index(tmp_path, monkeypatch):
    from scripts.ER2026.noitom_comparison import paired_bvh
    original = paired_bvh.validate_bvh
    def fail(path, *args, **kwargs):
        if path.name.startswith('noitom'): raise OSError('test disk failure')
        original(path, *args, **kwargs)
    monkeypatch.setattr(paired_bvh, 'validate_bvh', fail)
    s = run_pair(tmp_path)
    assert s.metadata['status'] == 'error'
    assert not (tmp_path/'records.csv').exists()
    assert (s.path/'custom.bvh.partial').exists()
    assert s.export_progress['percent'] < 100
    assert 'test disk failure' in s.error


def test_slerp_signs_and_midpoint_and_gap_rejection():
    a = np.array([[1.,0,0,0]])
    b = np.array([quaternion_from_rotation_vector((0,0,math.pi/2))])
    expected = quaternion_from_rotation_vector((0,0,math.pi/4))
    np.testing.assert_allclose(slerp(a,-b,.5)[0], expected, atol=1e-10)
    np.testing.assert_allclose(slerp(a,-a,.4), a)
    samples = [(t, a, np.zeros(3)) for t in (0, 100_000_000, 1_000_000_000)]
    with pytest.raises(ValueError, match='разрыв'):
        common_grid((samples,samples), 0, 1_000_000_000, 500_000_000)


def test_incomplete_duplicate_and_absent_five_sensor_frames_cannot_start(tmp_path):
    clock = Clock()
    r = PairedRecorder(clock=clock, watchdog=False)
    s = r.start(tmp_path, {'subject_id':'P001','word_set_id':'SET01'})
    own(r, clock, 0, complete=False)
    noitom(r, clock, 0)
    assert not s.ready.is_set()
    clock.now += 20_000_000
    own(r, clock, 1)
    assert s.ready.is_set()
    clock.now += 20_000_000
    own(r, clock, 1)
    assert len(s.own) == 1 and s.rejected['custom'] == 2
    r.stop()
    assert s.done.wait(3)


def test_bad_frames_after_complete_interval_do_not_change_result(tmp_path):
    clock = Clock()
    r = PairedRecorder(clock=clock, watchdog=False)
    s = r.start(tmp_path, {'subject_id':'P001','word_set_id':'SET01','duration_s':1.})
    for i in range(11):
        clock.now = 10_000_000_000 + i*100_000_000
        own(r, clock, i)
        noitom(r, clock, i)
    clock.now += 10_000_000
    noitom(r, clock, 11, invalid=True)
    own(r, clock, 0)  # reset beyond T1 is outside the recorded interval
    r.event('noitom_disconnect', incomplete=True)
    assert s.done.wait(3) and s.error is None
    assert s.metadata['status']=='complete'


def test_waiting_packet_claim_has_bounded_shutdown(tmp_path):
    clock = Clock()
    r = PairedRecorder(clock=clock, watchdog=False)
    s = r.start(tmp_path, {'subject_id':'P001','word_set_id':'SET01'})
    _, claim = r.stamp_packet()
    r.stop()
    assert not s.done.is_set()
    clock.now += 6_000_000_000
    r.tick()
    assert s.done.wait(3) and s.metadata['status']=='incomplete'
    assert not (tmp_path/'records.csv').exists()
    r.release_claim(claim)
    assert s.pending==0


def test_50_and_100_hz_frames_with_equal_windows_ticks_export_without_invented_times(tmp_path):
    clock = Clock()
    recorder = PairedRecorder(clock=clock, watchdog=False)
    session = recorder.start(tmp_path, {'subject_id': 'PTEST', 'word_set_id': 'SET01', 'duration_s': 1.})
    # Reproduce GetTickCount64: different 100 Hz frames share 15.625 ms ticks.
    for i in range(101):
        clock.now = 10_000_000_000 + (i * 10_000_000 // 15_625_000) * 15_625_000
        if i % 2 == 0:
            own(recorder, clock, i)
        noitom(recorder, clock, i)
        assert session.error is None, session.error
        assert all(b[0] > a[0] for a, b in zip(session.noitom, session.noitom[1:]))
        recorder.tick()
    assert session.done.wait(10) and session.metadata['status'] == 'complete'
    assert session.coalesced_timestamps['noitom'] > 0
    _, custom, dt = read_bvh(session.path / 'custom.bvh')
    _, native, native_dt = read_bvh(session.path / 'noitom.bvh')
    assert len(custom) == len(native) and dt == native_dt
    assert (len(custom) - 1) * dt == pytest.approx(1., abs=1e-8)
    assert session.metadata['fps'] <= 50
    assert len(list(csv.DictReader((tmp_path / 'records.csv').open(encoding='utf-8')))) == 1


@pytest.mark.parametrize('source', ['custom', 'noitom'])
def test_actual_backward_acquisition_time_still_stops_recording(tmp_path, source):
    clock = Clock()
    recorder = PairedRecorder(clock=clock, watchdog=False)
    session = recorder.start(tmp_path, {'subject_id': 'PTEST', 'word_set_id': 'SET01'})
    own(recorder, clock, 0)
    noitom(recorder, clock, 0)
    clock.now += 20_000_000
    own(recorder, clock, 1)
    noitom(recorder, clock, 1)
    clock.now -= 1_000_000
    (own if source == 'custom' else noitom)(recorder, clock, 2)
    assert session.done.wait(3) and source in session.error
    assert '1000000' in session.error
    assert not (tmp_path / 'records.csv').exists()


def test_noitom_matches_own_geometry_ignores_hands_root_translation_and_display_selection(tmp_path):
    from scripts.ER2026.noitom_comparison.noitom_adapter import NoitomAdapter
    clock = Clock()
    r = PairedRecorder(clock=clock, watchdog=False)
    s = r.start(tmp_path, {'subject_id':'PTEST','word_set_id':'SET01','duration_s':1.,'export_fps':20})
    native = FakeNoitomClient.frame(17, clock.now)
    mapped = NoitomAdapter().orientations(native, heading_offset_rad=.3)
    # A physically larger Noitom skeleton, moving wrists and displaced pelvis
    # must not change the shared skeleton or its five measured orientations.
    joints = {name: replace(j, position=j.position*2.5, default_position=j.default_position*2.5)
              for name,j in native.joints.items()}
    joints['Hips'] = replace(joints['Hips'], position=np.array([5., -9., 3.]))
    for name in ('LeftHand','RightHand'):
        joints[name] = replace(joints[name], rotation=np.full(4, np.nan))
    for i in range(101):
        clock.now = 10_000_000_000 + i*10_000_000
        if i % 2 == 0:
            _, claim = r.stamp_packet()
            r.complete_packet(claim, {}, {'measurement_complete':True, 'measurement_key':(i,i*10),
                                         'bvh_orientations':mapped})
        ns, claim = r.stamp_packet()
        r.noitom_frame(replace(native, host_timestamp_ns=ns, frame_index=i, sdk_timestamp=i*.01,
                              joints=joints, capture_context=claim), {}, enabled_segments=(), heading_offset_rad=.3)
        r.tick()
    assert s.done.wait(10) and not s.error, s.error
    a, own_frames, dt = read_bvh(s.path/'custom.bvh')
    b, noitom_frames, noitom_dt = read_bvh(s.path/'noitom.bvh')
    assert own_frames.shape == noitom_frames.shape == (21,45)
    assert dt == noitom_dt == .05
    np.testing.assert_allclose(own_frames, noitom_frames, atol=1e-8)
    assert s.metadata['source_rates_hz'] == pytest.approx([50.,100.])


@pytest.mark.parametrize('limit', [0,101,float('nan'),float('inf')])
def test_invalid_export_limit_cannot_start(tmp_path, limit):
    with pytest.raises(ValueError, match='BVH'):
        PairedRecorder(watchdog=False).start(tmp_path, {'subject_id':'PTEST','word_set_id':'SET01','export_fps':limit})


@pytest.mark.parametrize('source', ['custom', 'noitom'])
def test_single_suit_exports_without_other_stream_and_repeats_index(tmp_path, source):
    clock = Clock()
    recorder = PairedRecorder(clock=clock, watchdog=False)
    selected = own if source == 'custom' else noitom
    excluded = noitom if source == 'custom' else own
    excluded_name = 'noitom' if source == 'custom' else 'custom'
    for repeat in range(2):
        session = recorder.start(tmp_path, {'subject_id':'PTEST', 'word_set_id':'SET01',
            'duration_s':1., 'recording_sources':[source]})
        start = clock.now
        for i in range(51):
            clock.now = start + i * 20_000_000
            selected(recorder, clock, i)
            excluded(recorder, clock, i, **({'invalid': True} if excluded_name == 'noitom' else {'complete': False}))
            recorder.event(('noitom' if excluded_name == 'noitom' else 'own') + '_disconnect', incomplete=True)
            if i == 0:
                assert session.ready.is_set() and session.t0_ns == start
            assert session.pending == 0
            assert not (session.noitom if excluded_name == 'noitom' else session.own)
            recorder.tick()
        assert session.done.wait(10) and not session.error, session.error
        assert {path.name for path in session.path.iterdir()} == {source + '.bvh'}
        _, frames, dt = read_bvh(session.path / (source + '.bvh'))
        assert frames.shape == (51,45) and dt == .02
        assert session.metadata['source_rates_hz'] == [50.]
        assert session.metadata['recording_sources'] == [source]
        assert session.export_progress['percent'] == 100
        entries = list(csv.DictReader((tmp_path/'records.csv').open(encoding='utf-8')))
        assert len(entries) == repeat + 1
        assert entries[-1][source + '_bvh_path'] == f'{session.path.name}/{source}.bvh'
        assert entries[-1][excluded_name + '_bvh_path'] == ''
        append_index(tmp_path, entries[-1])
        assert len(list(csv.DictReader((tmp_path/'records.csv').open(encoding='utf-8')))) == repeat + 1
        clock.now += 20_000_000


@pytest.mark.parametrize('source', ['custom', 'noitom'])
def test_single_suit_gap_still_fails(tmp_path, source):
    clock = Clock()
    recorder = PairedRecorder(clock=clock, watchdog=False)
    session = recorder.start(tmp_path, {'subject_id':'PTEST', 'word_set_id':'SET01',
        'recording_sources':[source]})
    (own if source == 'custom' else noitom)(recorder, clock, 0)
    clock.now += 501_000_000
    recorder.tick()
    assert session.done.wait(3) and 'разрыв' in session.error
    assert not (tmp_path/'records.csv').exists()


@pytest.mark.parametrize('sources', [[], ['other'], ['custom', 'custom'], 'custom', None])
def test_invalid_source_selection_cannot_start(tmp_path, sources):
    recorder = PairedRecorder(watchdog=False)
    with pytest.raises(ValueError, match='костюм'):
        recorder.start(tmp_path, {'subject_id':'PTEST','word_set_id':'SET01', 'recording_sources':sources})
    assert recorder.active is None and not recorder.sessions
    assert not (tmp_path/'application.log').exists()
