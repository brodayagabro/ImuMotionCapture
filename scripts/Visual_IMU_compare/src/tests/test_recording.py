from dataclasses import replace
import csv
import json
import numpy as np
import pytest
from scripts.Visual_IMU_compare.src.paired_recording import PairedRecorder
from scripts.Visual_IMU_compare.src.paired_bvh import SEGMENTS
from scripts.Visual_IMU_compare.src.visual_protocol import BvhDecoder


class Clock:
    now=10_000_000_000
    def __call__(self): return self.now


def fixture_frame(pair):
    decoder=BvhDecoder()
    decoder.decode(json.dumps(pair['skeleton']).encode(),0)
    return decoder.decode(json.dumps(pair['frame']).encode(),0)


def visual(recorder, frame, index, **changes):
    ns, claim=recorder.stamp_visual_packet()
    recorder.visual_frame(replace(frame, host_timestamp_ns=ns, frame_index=index,
                                  source_timestamp=index*.05, capture_context=claim, **changes), {})


@pytest.mark.parametrize('sources', [['custom','visual'],['visual'],['custom']])
def test_record_selected_sources_on_same_grid_and_csv(tmp_path, packet_pair, sources):
    clock=Clock(); recorder=PairedRecorder(clock=clock,watchdog=False)
    session=recorder.start(tmp_path,dict(subject_id='P001',word_set_id='SET01',duration_s=1.2,recording_sources=sources))
    frame=fixture_frame(packet_pair)
    for index in range(31):
        clock.now=10_000_000_000+index*50_000_000
        if index%2==0:
            _,claim=recorder.stamp_packet()
            recorder.complete_packet(claim,{},dict(measurement_complete=True,measurement_key=(index,index*100),
                bvh_orientations=dict.fromkeys(SEGMENTS,[1.,0.,0.,0.])))
        if index>=1: visual(recorder,frame,index)
        recorder.tick()
        if recorder.active is None: break
    assert session.done.wait(5)
    assert session.error is None,session.error
    assert session.metadata['status']=='complete'
    assert session.metadata['joints']==14 and session.metadata['channels']==45
    headers=[]
    for source in sources:
        text=(session.path/(source+'.bvh')).read_text()
        hierarchy,motion=text.split('MOTION\n')
        lines=motion.splitlines()
        count=int(lines[0].split(':')[1]); dt=float(lines[1].split(':')[1])
        assert (count-1)*dt==pytest.approx(1.2)
        assert len(lines[2:])==count
        assert all(len(row.split())==45 for row in lines[2:])
        headers.append((hierarchy,count,dt))
    assert all(header==headers[0] for header in headers)
    with (tmp_path/'records.csv').open() as stream: rows=list(csv.DictReader(stream))
    assert len(rows)==1 and 'visual_bvh_path' in rows[0] and 'noitom_bvh_path' not in rows[0]
    for source in ['custom','visual']:
        assert bool(rows[0][source+'_bvh_path'])==(source in sources)


@pytest.mark.parametrize('failure',['occlusion','stream_change','stale','stop'])
def test_incomplete_visual_recording_never_added_to_index(tmp_path,packet_pair,failure):
    clock=Clock(); recorder=PairedRecorder(clock=clock,watchdog=False)
    session=recorder.start(tmp_path,dict(subject_id='P001',word_set_id='SET01',duration_s=1.2,recording_sources=['visual']))
    frame=fixture_frame(packet_pair)
    visual(recorder,frame,1)
    clock.now+=50_000_000
    if failure=='occlusion':
        joints=dict(frame.joints); joints['LeftForeArm']=replace(joints['LeftForeArm'],tracked=False)
        visual(recorder,frame,2,joints=joints)
    elif failure=='stream_change': visual(recorder,frame,2,stream_id='restarted')
    elif failure=='stop': recorder.stop()
    else:
        clock.now+=600_000_000; recorder.tick()
    assert session.done.wait(5)
    assert session.error and session.metadata['status']=='incomplete'
    assert not (tmp_path/'records.csv').exists()
    assert session.pending==0
