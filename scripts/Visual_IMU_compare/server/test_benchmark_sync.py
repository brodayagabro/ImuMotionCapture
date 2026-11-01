"""Known-offset signals, ambiguous patterns, gaps and real recording/API lifecycle."""
import csv
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

spec = importlib.util.spec_from_file_location('benchmark_sync',Path(__file__).with_name('benchmark_sync.py'))
s = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = s
spec.loader.exec_module(s)


def rows(delay=0, duration=90, noise=1):
    rng = np.random.default_rng(4)
    return [dict(frame_id=i+1,segment=1,t_s=float(t),brightness=s.synthetic_value(t-delay)+float(rng.normal(0,noise)))
            for i,t in enumerate(np.arange(0,duration,1/30))]


def edges(delay=0,slope=0):
    return [dict(kind='rise' if i%2==0 else 'fall',time_s=float(t*(1+slope)+delay),
                 lo_s=float(t*(1+slope)+delay-.001),hi_s=float(t*(1+slope)+delay+.001),segment=1,
                 frame_before=i*10,frame_after=i*10+1)
            for i,t in enumerate(s.PATTERN[s.PATTERN<100])]


@pytest.mark.parametrize('delay',[.12,-.14,.42])
def test_recovers_shift_with_frame_quantization(delay):
    a,_=s.events_from_samples(rows())
    b,_=s.events_from_samples(rows(delay))
    r=s.compare_events(a,b)
    assert r['accepted'],r['status']
    assert abs(r['offset_median_ms']-delay*1000)<34
    assert abs(r['suggested_offset_ms']+delay*1000)<34
    assert 30 < r['transition_bracket_halfwidth_median_ms'] < 35


def test_known_drift_and_missing_edges():
    a,b=edges(),edges(.12,.0005)
    b=[e for i,e in enumerate(b) if i%17!=0]
    r=s.compare_events(a,b)
    assert r['accepted'] and r['match_fraction']>.95
    assert r['observed_trend_ms_per_min']==pytest.approx(30,abs=.1)
    assert r['detrended_jitter_p95_ms']<.1
    assert len({p['camera_event'] for p in r['matches']})==r['matched_events']


def test_periodic_stimulus_is_not_an_unambiguous_clock():
    a=edges()[:100]
    for i,e in enumerate(a):
        e.update(time_s=i*.5,lo_s=i*.5-.01,hi_s=i*.5+.01)
    b=[dict(e,time_s=e['time_s']+.1,lo_s=e['lo_s']+.1,hi_s=e['hi_s']+.1) for e in a]
    r=s.compare_events(a,b,max_lag=2)
    assert r['status']=='ambiguous_pattern'
    assert r['suggested_offset_ms'] is None


def test_constant_or_low_contrast_never_produces_sync_result():
    data=rows()
    for r in data:r['brightness']=100+.1*(r['frame_id']%3)
    events,stats=s.events_from_samples(data)
    assert events==[] and stats['status']=='low_contrast'
    assert not s.compare_events(events,edges())['accepted']


def test_no_transition_is_invented_across_capture_gap_or_reconnect():
    data=[dict(frame_id=i,segment=1,t_s=i/30,brightness=20) for i in range(30)]
    data += [dict(frame_id=i+30,segment=2,t_s=2+i/30,brightness=220) for i in range(30)]
    events,stats=s.events_from_samples(data)
    assert events==[] and stats['segments']==2 and stats['gap_count']==1
    for row in data:row['segment']=1
    assert s.events_from_samples(data)[0]==[]


def test_short_noise_pulses_are_rejected():
    data=rows()
    for i in range(100,120):data[i]['brightness']=20
    data[110]['brightness']=220
    events,_=s.events_from_samples(data)
    assert not any(e['frame_after']==111 for e in events)


def test_reconnect_disables_global_drift_fit():
    a,b=edges(),edges(.1)
    for e in b[len(b)//2:]:e['segment']=2
    r=s.compare_events(a,b)
    assert r['accepted']
    assert r['observed_trend_ms_per_min'] is None
    assert not r['continuous_segment']


def test_search_boundary_does_not_suggest_auto_offset():
    r=s.compare_events(edges(),edges(.96),max_lag=1)
    assert r['status']=='lag_at_search_boundary' and not r['accepted']


def test_low_matching_coverage_is_rejected():
    b=edges(.1)
    for i,e in enumerate(b):
        if i%2==0:
            for key in ('time_s','lo_s','hi_s'):e[key]+=.2
    b.sort(key=lambda e:e['time_s'])
    assert not s.compare_events(edges(),b)['accepted']


def test_invalid_timestamps_are_not_silently_sorted():
    data=rows(); data[30]['t_s']=data[29]['t_s']
    with pytest.raises(ValueError,match='timestamps'):s.events_from_samples(data)


def test_saved_demo_and_offline_reanalysis(tmp_path):
    directory,report=s.demo_session(tmp_path,60)
    assert report['synthetic'] and report['comparisons']['cam2']['accepted']
    assert abs(report['comparisons']['cam2']['offset_median_ms']-120)<34
    assert (directory/'report.html').is_file()
    assert len(s.load_samples(directory/'cam1.csv'))==1800
    assert json.loads((directory/'report.json').read_text())==s.analyze(directory)


def test_http_roi_recording_and_stop_retains_csv(tmp_path):
    args=s.arguments(['serve','--demo','--output',str(tmp_path),'--duration','5'])
    engine=s.Engine(args,s.read_cameras(args))
    with TestClient(s.make_app(engine)) as client:
        deadline=time.monotonic()+3
        while time.monotonic()<deadline:
            if client.get('/frame/cam1.jpg').status_code==200:break
            time.sleep(.05)
        response=client.get('/frame/cam1.jpg')
        assert response.status_code==200 and response.content.startswith(b'\xff\xd8')
        assert client.post('/api/roi/cam1',json={'roi':[280,200,80,80],'image_size':[320,240]}).status_code==409
        assert client.post('/api/roi/cam1',json={'roi':[280,200,80,80],'image_size':[640,480]}).status_code==200
        assert (tmp_path/'rois.json').is_file()
        assert client.post('/api/start',json={'duration':0}).status_code==409
        assert client.post('/api/start',json={'duration':5}).status_code==200
        assert client.post('/api/start',json={}).status_code==409
        assert client.post('/api/roi/cam1',json={'roi':[280,200,80,80],'image_size':[640,480]}).status_code==409
        time.sleep(.3)
        assert client.post('/api/stop').status_code==200
        assert client.get('/report').status_code==200
        report=client.get('/api/report').json()
        assert not report['comparisons']['cam2']['accepted']  # Too short, never a successful precision claim.
        assert client.get('/frame/unknown.jpg').status_code==404
    for camera in engine.cameras.values():assert not camera.thread.is_alive()
    directories=list(tmp_path.glob('20*'))
    assert len(directories)==1 and s.load_samples(directories[0]/'cam1.csv')


def test_source_credentials_not_in_manifest(tmp_path):
    assert s.safe_source('rtsp://user:secret@192.168.1.2:8554/cam1?token=secret')=='rtsp://192.168.1.2:8554/cam1'


def test_unfinished_session_is_not_accepted(tmp_path):
    directory,_=s.demo_session(tmp_path,30)
    manifest=json.loads((directory/'session.json').read_text())
    manifest['completed']=False
    s.dump(directory/'session.json',manifest)
    result=s.analyze(directory)['comparisons']['cam2']
    assert result['status']=='capture_invalid' and result['suggested_offset_ms'] is None


@pytest.mark.parametrize('failure',['resolution','write'])
def test_capture_failures_leave_raw_data_and_invalidate_estimates(tmp_path,failure):
    args=s.arguments(['serve','--demo','--output',str(tmp_path)])
    engine=s.Engine(args,s.read_cameras(args))
    image=np.zeros((480,640,3),np.uint8)
    for camera in engine.cameras.values():
        camera.segment=1
        camera.accept(image,time.monotonic_ns())
    engine.begin_recording()
    camera=engine.cameras['cam1']
    if failure=='resolution':
        camera.accept(np.zeros((240,320,3),np.uint8),time.monotonic_ns())
    else:
        class FailedWriter:
            def writerow(self,row):raise OSError('simulated disk full')
        engine.session['writers']['cam1']=FailedWriter()
        camera.accept(image,time.monotonic_ns())
        assert engine.session['fatal']
    report=engine.end_recording()
    assert report['comparisons']['cam2']['status']=='capture_invalid'
    assert (engine.last_directory/'cam1.csv').is_file()
    assert engine.finished.is_set() and not engine.analyzing


@pytest.mark.parametrize('roi,size',[([-1,0,20,20],[640,480]),([0,0,1,10],[640,480]),([600,0,80,80],[640,480])])
def test_bad_roi(roi,size):
    with pytest.raises(ValueError):s.checked_roi(roi,size)
