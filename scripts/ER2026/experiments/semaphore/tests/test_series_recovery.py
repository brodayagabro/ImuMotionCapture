"""Interrupted control handshakes must never stop another recording."""
from types import SimpleNamespace
import pytest
from scripts.ER2026.experiments.semaphore.config import load_config
from scripts.ER2026.experiments.semaphore.series_coordinator import SeriesCoordinator


def coordinator(tmp_path):
    commands = []
    label = SimpleNamespace(setText=lambda text: None)
    client = SimpleNamespace(controlled=SimpleNamespace(connect=lambda cb: None),
        send_control=lambda command, **data: commands.append((command, data)))
    window = SimpleNamespace(client=client, sessions_root=tmp_path, series_summary=label,
        path_label=label, notice=label, session_log=None, _finish_session_log=lambda: None)
    c = SeriesCoordinator(window)
    c.create(load_config(), 'P001', 7, {})
    c.current, c.trials = c.plan.begin()
    c.phase = 'recovering'
    return c, commands


def test_lost_start_ack_is_recovered_by_series_id_then_stopped_by_recording_id(tmp_path):
    c, commands = coordinator(tmp_path)
    c.response({'command':'status', 'accepted':True, 'ready':True, 'recording_session_id':'ours',
                'series_context':{**c.current, 'own_calibration_id':'imu1', 'noitom_calibration_id':'noitom1'}})
    assert commands == [('stop_recording', {'recording_session_id':'ours'})]
    assert c.plan.data['previous_calibrations']['own_calibration_id'] == 'imu1'
    c.response({'command':'stop_recording', 'accepted':True})
    assert c.phase == 'saving'
    c.response({'command':'status', 'accepted':True, 'saving':False, 'last_recording_id':'ours', 'last_recording_status':'complete'})
    assert c.phase == 'between' and c.plan.data['series'][-1]['status'] == 'interrupted'
    assert c.plan.data['active_series_id'] is None and len(c.plan.remaining) == 120


@pytest.mark.parametrize('reason', ['control_timeout', 'ID записи не совпадает'])
def test_ambiguous_command_failure_reconciles_without_touching_foreign_recording(tmp_path, reason):
    c, commands = coordinator(tmp_path)
    c.phase = 'starting' if reason == 'control_timeout' else 'stopping'
    c.response({'command':'start_recording' if c.phase == 'starting' else 'stop_recording',
                'accepted':False, 'reason':reason})
    assert c.phase == 'recovering'
    c.response({'command':'status', 'accepted':True, 'ready':True, 'recording_session_id':'foreign',
                'series_context':{'series_id':'foreign'}, 'saving':False})
    assert not commands and c.phase == 'between'
    assert c.plan.data['series'][-1]['recording_status'] == 'unknown'
