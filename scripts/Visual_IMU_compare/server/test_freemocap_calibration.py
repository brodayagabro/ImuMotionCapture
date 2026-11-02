"""Native calibration worker lifecycle and coordinate-format adapter tests."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from test_visual_mocap_server import s

spec = importlib.util.spec_from_file_location('freemocap_calibration',Path(__file__).with_name('freemocap_calibration.py'))
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


def test_coordinate_adapter_preserves_projections_with_rotated_reference():
    k = np.array([[900.,0,640],[0,900,360],[0,0,1]])
    cameras = []
    for name,rvec,tvec in [('a',[.2,-.3,.1],[300.,-100.,200.]),('b',[-.1,.2,.3],[-400.,80.,100.])]:
        r,_ = s.cv2.Rodrigues(np.array(rvec))
        cameras.append(SimpleNamespace(id=name,image_size=(1280,720),
            intrinsics=SimpleNamespace(to_camera_matrix=lambda:k,to_dist_coeffs_5=lambda:np.zeros(5)),
            extrinsics=SimpleNamespace(rotation_matrix=r,translation=np.array(tvec))))
    result = adapter.convert_cameras(cameras)
    np.testing.assert_allclose(result['a']['R'],np.eye(3),atol=1e-12)
    np.testing.assert_allclose(result['a']['t'],0,atol=1e-12)
    world_mm = np.array([[40.,30.,2500.],[-100.,-200.,3000.]])
    reference = cameras[0].extrinsics
    world_m = (world_mm @ reference.rotation_matrix.T + reference.translation)/1000
    checked = s.checked_calibration(dict(schema_version=1,units='m',extrinsics_convention='world_to_camera',cameras=result),['a','b'])
    for camera in cameras:
        expected = s.cv2.projectPoints(world_mm,camera.extrinsics.rotation_matrix,
            camera.extrinsics.translation,k,np.zeros(5))[0].reshape(-1,2)
        np.testing.assert_allclose(s.project(world_m,checked[camera.id]),expected,atol=1e-8)


@pytest.fixture
def worker(tmp_path,monkeypatch):
    args = s.arguments(['--device','cpu'])
    args.calibration_python = Path(sys.executable)
    monkeypatch.setattr(s,'HERE',tmp_path)
    script = tmp_path/'freemocap_calibration.py'
    return args,script,tmp_path


@pytest.mark.parametrize('mode',['cancel','timeout'])
def test_solver_process_is_reaped_on_cancel_or_timeout(worker,mode):
    args,script,folder = worker
    script.write_text("import os,pathlib,time\npathlib.Path(__file__).with_suffix('.pid').write_text(str(os.getpid()))\ntime.sleep(30)\n")
    cancelled = threading.Event()
    if mode == 'timeout':
        args.calibration_timeout = .3
    timer = threading.Timer(.3,cancelled.set) if mode == 'cancel' else None
    if timer:
        timer.start()
    started = time.monotonic()
    try:
        with pytest.raises(s.CalibrationCancelled if mode == 'cancel' else TimeoutError):
            s.run_freemocap_calibration(args,{},folder,cancelled)
    finally:
        if timer:
            timer.join()
    assert time.monotonic()-started < 5
    pid = int(script.with_suffix('.pid').read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid,0)
    assert not (folder/'calibration-result.json').exists()


def test_worker_failure_reports_log_without_falling_back(worker):
    args,script,folder = worker
    script.write_text("raise RuntimeError('native solver rejected observations')\n")
    with pytest.raises(ValueError,match='native solver rejected observations'):
        s.run_freemocap_calibration(args,{},folder,threading.Event())


def test_worker_requires_environment_and_native_provenance(worker):
    args,script,folder = worker
    real_python = args.calibration_python
    args.calibration_python = folder/'missing/python'
    with pytest.raises(ValueError,match='environment is missing'):
        s.run_freemocap_calibration(args,{},folder,threading.Event())
    args.calibration_python = real_python
    # A successful process without a native-solver stamp is not accepted.
    result = s.demo_calibration(['a','b'])
    script.write_text('import pathlib,sys\npathlib.Path(sys.argv[sys.argv.index("--output")+1]).write_text('+repr(json.dumps(result))+')\n')
    with pytest.raises(ValueError,match='native FreeMoCap result'):
        s.run_freemocap_calibration(args,{'camera_ids':['a','b']},folder,threading.Event())


def test_missing_environment_is_reported_before_recording(tmp_path):
    args = s.arguments(['--device','cpu','--calibration-python',str(tmp_path/'missing')])
    now = time.monotonic()
    histories = lambda:{n:[s.Frame(1,now,now,np.zeros((48,64,3),np.uint8))] for n in ('a','b')}
    job = s.VideoCalibration(args,['a','b'],histories,lambda _:None)
    with pytest.raises(ValueError,match='environment is missing'):
        job.start()
    assert not job.active and job.thread is None
