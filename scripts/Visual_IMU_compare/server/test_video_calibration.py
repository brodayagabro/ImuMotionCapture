"""Temporary calibration video lifecycle, real codecs/detection and HTTP controls."""
import json
import time
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from test_visual_mocap_server import s


@pytest.fixture
def args(tmp_path):
    args = s.arguments(['--demo', '--device', 'cpu', '--calibration-duration', '1'])
    args.calibration_temp = tmp_path/'videos'
    args.calibration = tmp_path/'calibration.json'
    return args


def wait_until(predicate, timeout=5):
    end = time.monotonic()+timeout
    while time.monotonic() < end:
        if predicate():
            return
        time.sleep(.01)
    raise AssertionError('Timed out waiting for calibration worker')


@pytest.fixture
def job(args):
    sequence = 0
    applied = []
    def histories():
        nonlocal sequence
        sequence += 1
        now = time.monotonic()
        return {n:[s.Frame(sequence,now,now,np.full((48,64,3),sequence % 255,dtype=np.uint8))]
                for n in ('a','b')}
    instance = s.VideoCalibration(args, ['a','b'], histories, applied.append)
    yield instance, applied
    instance.close()


def test_video_groups_use_timestamps_offsets_and_never_reuse_frames():
    # Camera B started later; its AVI indices cannot be paired with A's indices.
    rows = {'a':[dict(aligned=i*.6) for i in range(18)],
            'b':[dict(aligned=i*.6+.02) for i in range(3,18) if i != 8]}
    groups = s.calibration_video_groups(rows,['a','b'],.05)
    assert groups[0] == {'a':3,'b':0}
    assert len(groups) == 14
    assert all(g['a'] != 8 for g in groups)
    for name in rows:
        assert len({g[name] for g in groups}) == len(groups)
    assert s.calibration_video_groups(rows,['a','b'],.005) == []
    with pytest.raises(ValueError,match='Nonmonotonic'):
        s.calibration_video_groups({'a':[dict(aligned=1),dict(aligned=0)]},['a'],.05)


def test_recording_real_videos_and_sidecars_then_auto_stop_cleanup(job, monkeypatch):
    instance, applied = job
    inspected = []
    def process(folder, sizes):
        for name in instance.ids:
            rows = [json.loads(line) for line in (folder/f'{name}.jsonl').read_text().splitlines()]
            cap = s.cv2.VideoCapture(str(folder/f'{name}.avi'))
            try:
                assert int(cap.get(s.cv2.CAP_PROP_FRAME_COUNT)) == len(rows) > 3
                assert [r['index'] for r in rows] == list(range(len(rows)))
                ok, image = cap.read()
                assert ok and image.shape[:2] == (48,64)
                assert sizes[name] == (64,48)
            finally:
                cap.release()
        inspected.append(folder)
        return dict(rms_px={'a':.1,'b':.1})
    monkeypatch.setattr(instance,'process',process)
    instance.start()
    with pytest.raises(ValueError,match='already'):
        instance.start()
    wait_until(lambda:not instance.snapshot()['active'])
    state = instance.snapshot()
    assert state['status'] == 'complete' and state['temporary_files_removed']
    assert len(applied) == 1 and len(inspected) == 1
    assert not inspected[0].exists()
    assert not list(instance.args.calibration_temp.iterdir())


@pytest.mark.parametrize('mode',['cancel','shutdown','process_error','save_error'])
def test_worker_cleanup_and_no_partial_result(job, monkeypatch, mode):
    instance, applied = job
    instance.args.calibration_temp.mkdir()
    sentinel = instance.args.calibration_temp/'unrelated.txt'
    sentinel.write_text('keep')
    def process(folder, sizes):
        if mode == 'process_error':
            raise ValueError('Insufficient varied board poses')
        return dict(rms_px={'a':.1})
    monkeypatch.setattr(instance,'process',process)
    if mode == 'save_error':
        monkeypatch.setattr(instance,'apply',lambda _: (_ for _ in ()).throw(OSError('Cannot save calibration')))
    instance.start()
    wait_until(lambda:instance.snapshot()['frames']['a'] >= 3)
    if mode == 'cancel':
        instance.cancel()
    elif mode == 'shutdown':
        instance.close()
    else:
        instance.stop_recording()
    wait_until(lambda:not instance.snapshot()['active'])
    state = instance.snapshot()
    assert state['status'] == ('cancelled' if mode in ('cancel','shutdown') else 'error')
    assert state['temporary_files_removed'] and not applied
    assert list(instance.args.calibration_temp.iterdir()) == [sentinel]
    assert sentinel.read_text() == 'keep'


def test_cancel_during_processing_prevents_late_apply(job, monkeypatch):
    import threading
    instance, applied = job
    entered, release = threading.Event(), threading.Event()
    def process(folder, sizes):
        entered.set()
        assert release.wait(3)
        return dict(rms_px={'a':.1})
    monkeypatch.setattr(instance,'process',process)
    instance.start()
    wait_until(lambda:instance.snapshot()['frames']['a'] >= 3)
    instance.stop_recording()
    assert entered.wait(3)
    instance.cancel()
    release.set()
    wait_until(lambda:not instance.snapshot()['active'])
    assert instance.snapshot()['status'] == 'cancelled'
    assert instance.snapshot()['temporary_files_removed'] and not applied


@pytest.mark.parametrize('failure',['writer','disk','size'])
def test_capture_failures_are_reported_and_cleaned(job, monkeypatch, failure):
    instance, applied = job
    if failure == 'writer':
        monkeypatch.setattr(s.cv2,'VideoWriter',lambda *a:SimpleNamespace(isOpened=lambda:False,release=lambda:None))
    elif failure == 'disk':
        monkeypatch.setattr(s.shutil,'disk_usage',lambda _:SimpleNamespace(free=0))
    else:
        histories = instance.histories
        calls = 0
        def changed():
            nonlocal calls
            calls += 1
            frames = histories()
            if calls > 5:
                f = frames['a'][0]
                frames['a'][0] = s.Frame(f.sequence,f.received,f.aligned,np.zeros((50,64,3),dtype=np.uint8))
            return frames
        instance.histories = changed
    instance.start()
    wait_until(lambda:not instance.snapshot()['active'])
    assert instance.snapshot()['status'] == 'error'
    assert instance.snapshot()['error']
    assert instance.snapshot()['temporary_files_removed'] and not applied


def calibration_images(args, count=28, camera_count=2):
    """Render a physical board in known stereo geometry; no detector mocks."""
    session = s.CharucoObservations(args)
    board = session.board.generateImage((1400,1000), marginSize=0)
    size = (1280,960)
    k = np.array([[1000.,0,640],[0,1000.,480],[0,0,1]])
    r,_ = s.cv2.Rodrigues(np.array([.01,.13,-.01]))
    cameras = {'a':dict(K=k,D=np.zeros(5),R=np.eye(3),t=np.zeros(3)),
               'b':dict(K=k,D=np.zeros(5),R=r,t=np.array([-.30,.01,.02]))}
    if camera_count == 3:
        r3,_ = s.cv2.Rodrigues(np.array([-.02,-.15,.02]))
        cameras['c'] = dict(K=k,D=np.zeros(5),R=r3,t=np.array([.25,.03,.06]))
    outline = np.array([[0,0,0],[.28,0,0],[.28,.20,0],[0,.20,0]])
    pixels = np.float32([[0,0],[1399,0],[1399,999],[0,999]])
    frames = []
    for i in range(count):
        rotation,_ = s.cv2.Rodrigues(np.array([.45*np.sin(i*.7),.5*np.cos(i*.9),.15*np.sin(i*.5)]))
        position = (rotation @ outline.T).T + [-.12+.07*np.sin(i*.8),-.1+.05*np.cos(i*.6),.8+.14*np.sin(i*.4)]
        pair = {}
        for name,camera in cameras.items():
            transform = s.cv2.getPerspectiveTransform(pixels,s.project(position,camera).astype(np.float32))
            image = s.cv2.warpPerspective(board,transform,size,borderValue=255)
            pair[name] = s.cv2.cvtColor(image,s.cv2.COLOR_GRAY2BGR)
        frames.append(pair)
    return frames,cameras,size


@pytest.mark.parametrize('camera_count', [2,3])
def test_real_video_detection_and_calibration_recover_camera_geometry(args,tmp_path,camera_count):
    frames, expected, size = calibration_images(args,camera_count=camera_count)
    folder = tmp_path/'input'
    folder.mkdir()
    for name in expected:
        writer = s.cv2.VideoWriter(str(folder/f'{name}.avi'),s.cv2.VideoWriter_fourcc(*'MJPG'),30,size)
        assert writer.isOpened()
        try:
            with (folder/f'{name}.jsonl').open('w') as file:
                for i,pair in enumerate(frames):
                    writer.write(pair[name])
                    file.write(json.dumps(dict(index=i,sequence=i+1,received=10+i*.6,aligned=10+i*.6))+ '\n')
        finally:
            writer.release()
    instance = s.VideoCalibration(args,list(expected),lambda:{},lambda _:None)
    result = instance.process(folder,dict.fromkeys(expected,size))
    s.checked_calibration(result,list(expected))
    assert result['solver']['name'] == 'freemocap_anipose'
    assert result['solver']['version'] == 'v2.0.0-alpha.25'
    assert result['solver']['commit'] == '700c8614d11d886dacfea8e52333d3c0b942ef52'
    assert result['solver']['median_reprojection_px'] < 1
    assert result['video_calibration']['samples'] >= 12
    assert max(result['rms_px'].values()) < 1
    for name in expected:
        np.testing.assert_allclose(result['cameras'][name]['t'],expected[name]['t'],atol=.008)
        np.testing.assert_allclose(result['cameras'][name]['R'],expected[name]['R'],atol=.015)
        np.testing.assert_allclose(np.diag(result['cameras'][name]['K'])[:2],[1000,1000],rtol=.03)


def test_static_board_does_not_count_as_twelve_poses(args):
    pair, _, _ = calibration_images(args,1)
    session = s.CharucoObservations(args)
    frames = {name:s.Frame(1,1,1,image) for name,image in pair[0].items()}
    assert session.capture(frames,['a','b'],distinct=True) == 1
    frames = {name:s.Frame(2,2,2,image) for name,image in pair[0].items()}
    with pytest.raises(ValueError,match='Repeated board pose'):
        session.capture(frames,['a','b'],distinct=True)
    assert len(session.samples) == 1


def test_apply_calibration_preserves_previous_file_on_invalid_result(args):
    engine = s.Engine(args)
    args.calibration.write_text('old calibration')
    previous = engine.calibration
    bad = s.demo_calibration(engine.ids)
    bad['cameras']['cam1']['R'][0][0] = 5
    with pytest.raises(ValueError,match='rotation'):
        engine.apply_calibration(bad)
    assert args.calibration.read_text() == 'old calibration'
    assert engine.calibration is previous
    stream_id = engine.bvh.stream_id
    engine.apply_calibration(s.demo_calibration(engine.ids))
    assert json.loads(args.calibration.read_text())['schema_version'] == 1
    assert engine.bvh.stream_id != stream_id
    assert engine.armature.snapshot(time.monotonic())['status'] == 'rest'


def test_api_video_controls_keep_state_responsive(args,monkeypatch):
    engine = s.Engine(args)
    args.demo = False  # Test real API actions using manually published camera frames.
    monkeypatch.setattr(engine,'start',lambda:None)
    with TestClient(s.create_app(engine)) as client:
        assert client.post('/api/calibration/start').status_code == 409
        for camera in engine.cameras.values():
            camera.publish(np.zeros((48,64,3),np.uint8))
        assert client.post('/api/calibration/start').status_code == 202
        assert client.post('/api/calibration/start').status_code == 409
        assert client.get('/api/state').json()['video_calibration']['active']
        assert client.post('/api/calibration/cancel').status_code == 202
        wait_until(lambda:not engine.video_calibration.snapshot()['active'])
        assert client.get('/api/state').json()['video_calibration']['temporary_files_removed']
        assert client.post('/api/calibration/stop').status_code == 409
        assert client.post('/api/calibration/capture').status_code == 404


def test_rejects_video_timestamp_count_mismatch(args,tmp_path):
    folder = tmp_path/'truncated'
    folder.mkdir()
    for name in ('a','b'):
        writer = s.cv2.VideoWriter(str(folder/f'{name}.avi'),s.cv2.VideoWriter_fourcc(*'MJPG'),30,(64,48))
        assert writer.isOpened()
        try:
            for _ in range(12 if name == 'a' else 11):
                writer.write(np.zeros((48,64,3),np.uint8))
        finally:
            writer.release()
        (folder/f'{name}.jsonl').write_text(''.join(json.dumps(dict(
            index=i,sequence=i+1,received=i*.6,aligned=i*.6))+'\n' for i in range(12)))
    instance = s.VideoCalibration(args,['a','b'],lambda:{},lambda _:None)
    with pytest.raises(OSError,match='frame count'):
        instance.process(folder,dict.fromkeys(('a','b'),(64,48)))


def test_temporary_video_size_limit(job,tmp_path):
    instance,_ = job
    instance.args.calibration_max_mb = 16
    folder = tmp_path/'limit'
    folder.mkdir()
    with (folder/'a.avi').open('wb') as file:
        file.truncate(17*1024**2)
    with pytest.raises(OSError,match='size limit'):
        instance.check_disk(folder)


def test_atomic_save_failure_keeps_previous_calibration(args,monkeypatch):
    engine = s.Engine(args)
    args.calibration.write_text('previous calibration')
    previous = engine.calibration
    replace = type(args.calibration).replace
    def fail_replace(path,target):
        if target == args.calibration:
            raise OSError('Read-only target')
        return replace(path,target)
    monkeypatch.setattr(type(args.calibration),'replace',fail_replace)
    with pytest.raises(OSError,match='Read-only'):
        engine.apply_calibration(s.demo_calibration(engine.ids))
    assert args.calibration.read_text() == 'previous calibration'
    assert engine.calibration is previous
    assert not list(args.calibration.parent.glob('*.tmp'))
