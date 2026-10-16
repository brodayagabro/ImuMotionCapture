from dataclasses import replace
import csv
import json
import socket
import threading
import time

import pytest
from PyQt6.QtCore import QSettings, QTimer
from PyQt6.QtWidgets import QApplication
from pyqt_mocap import mpu_udp_viewer
from pyqt_mocap.mocap_core import DEFAULT_SENSOR_MAPPING
from scripts.ER2026.noitom_comparison.comparison_window import MocapComparisonWindow
from scripts.ER2026.noitom_comparison.config import ComparisonConfig
from scripts.ER2026.noitom_comparison.event_server import EventServer
from scripts.ER2026.noitom_comparison.noitom_client import FakeNoitomClient
from scripts.ER2026.noitom_comparison.tests.test_gui import pump_until
from scripts.ER2026.experiments.semaphore.config import load_config, Timing
from scripts.ER2026.experiments.semaphore.presenter_window import PresenterWindow
from scripts.ER2026.experiments.semaphore.calibration_protocol import calibrate_series
from scripts.ER2026.experiments.semaphore.tests.test_series_calibration import physical_series_captures

pytestmark = pytest.mark.gui


@pytest.mark.parametrize('control', ['tcp', 'builtin'])
def test_tcp_control_udp_acquisition_loop_export_repeat_early_stop_and_reopen(tmp_path, monkeypatch, control):
    monkeypatch.setattr(mpu_udp_viewer, 'QSettings', lambda *a: QSettings(str(tmp_path/'viewer.ini'), QSettings.Format.IniFormat))
    app = QApplication.instance() or QApplication([])
    window = MocapComparisonWindow(ComparisonConfig(fake=True, recordings_dir=str(tmp_path/'records')),
                                   client=FakeNoitomClient(rate_hz=100))
    window.event_server = EventServer(window.recorder, 0, control=window.control_bridge)
    config = replace(load_config(), words={'lower':('А',),'upper':('У',)}, recording_seconds=3.,
                     timing=Timing(.01,.01,.03,.08,.01,.01))
    presenter = PresenterWindow(config, port=window.event_server.port,
                                backend=window._handle_control if control == 'builtin' else None)
    presenter.fullscreen_check.setChecked(False)
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp.bind(('127.0.0.1',0))
    udp.settimeout(1)
    window.config.device_ip, window.config.device_port = udp.getsockname()
    stopping = threading.Event()
    producer = reopened = None
    try:
        assert window.connect_device()
        _, address = udp.recvfrom(1024)
        captures = physical_series_captures()
        window._apply_guided_calibration(calibrate_series(captures))
        profile_path = tmp_path / 'saved_calibration.json'
        profile_path.write_text(json.dumps(window.calibration_document), encoding='utf-8')
        window.load_calibration_profile(profile_path)
        assert abs(window.model.drift_reference_s - time.perf_counter()) < 1.
        assert window._control_status()['own_calibration_method'] == 'neutral_a_semaphore_a_t_u_neutral_a_sensor_y'
        values = captures['neutral_a_pose'].average
        def stream():
            index = 0
            while not stopping.wait(.02):
                index += 1
                payload = f'FRAME {index} {index*20} 5\n'+'\n'.join(
                    f'Q {sid} '+' '.join(map(str,values[name])) for name,sid in DEFAULT_SENSOR_MAPPING.items())
                udp.sendto(payload.encode(), address)
        window.start_stream()
        producer = threading.Thread(target=stream, daemon=True)
        producer.start()
        window.noitom_worker.command('connect')
        window.noitom_worker.command('start')
        pump_until(app, lambda: window.noitom_stats.frames>3 and window.own_stats.frames>3 and
                   (control == 'builtin' or presenter.client.control_supported))
        presenter._send('confirm_noitom_calibration')
        pump_until(app, presenter.start_button.isEnabled)
        saved = []
        for i in range(2):
            presenter.word_set_combo.setCurrentIndex(i)
            presenter._start()
            pump_until(app, lambda: window.session is not None and window.session not in saved)
            session = window.session
            saved.append(session)
            pump_until(app, lambda: presenter.loop is not None)
            assert session.ready.is_set() and presenter.deadline_ns == session.deadline_ns
            assert not window.imu_calibration_button.isEnabled()
            assert not presenter.word_set_combo.isEnabled()
            poses = set()
            def avatar_moves():
                poses.add(presenter.participant_window.avatar.pose)
                return len(poses) > 1
            pump_until(app, avatar_moves)
            pump_until(app, lambda: presenter.loop is not None and presenter.loop.cycles>=2)
            pump_until(app, lambda: session.done.is_set() and presenter.session_id is None, timeout=8)
            assert session.metadata['status']=='complete', session.error
            assert (session.path/'custom.bvh').exists() and (session.path/'noitom.bvh').exists()
            assert session.stop_ns-session.t0_ns==3_000_000_000
            assert presenter.participant_window.caption.text() == 'Запись сохранена'
            assert presenter.export_progress.value() == 100
            pump_until(app, presenter.start_button.isEnabled)
        assert saved[0].path != saved[1].path
        presenter.duration_spin.setValue(3)
        presenter._start()
        pump_until(app, lambda: presenter.session_id is not None and presenter.loop is not None)
        early = window.session
        presenter._stop()
        pump_until(app, lambda: early.done.is_set() and presenter.session_id is None)
        assert early.metadata['status']=='incomplete'
        assert presenter.participant_window.caption.text() == 'Запись остановлена'
        assert early.error in presenter.participant_window.cue.text()
        with (tmp_path/'records/records.csv').open(encoding='utf-8') as f:
            rows=list(csv.DictReader(f))
        assert len(rows)==2 and [r['word_set_id'] for r in rows]==['SET01','SET02']
        assert not list((tmp_path/'records').rglob('*.jsonl'))
        assert not list((tmp_path/'records').rglob('events.csv'))
        presenter.close()
        reopened = PresenterWindow(config, backend=window._handle_control)
        reopened._poll()
        assert reopened.records_table.rowCount()==2
        assert window.model.segment_calibrations
    finally:
        presenter.close()
        if reopened: reopened.close()
        stopping.set()
        if producer: producer.join(2)
        window.close()
        pump_until(app, lambda: window._allow_close, timeout=8)
        udp.close()


def test_full_five_minute_pair_exports_on_one_worker_with_progress_and_responsive_gui(tmp_path, monkeypatch):
    import numpy as np
    from scripts.ER2026.noitom_comparison.paired_recording import PairedRecorder, PairSession
    from scripts.ER2026.noitom_comparison.tests.test_paired_recording import Clock
    from pyqt_mocap.tests.test_bvh import read_bvh
    app = QApplication.instance() or QApplication([])
    clock = Clock()
    recorder = PairedRecorder(clock=clock, watchdog=False)
    session = recorder.start(tmp_path, {'subject_id':'BENCH','word_set_id':'SET01','duration_s':300})
    # A full-size export workload, after acquisition has already completed.
    for rate, target in ((50,session.own),(100,session.noitom)):
        for i in range(300*rate+1):
            angle = np.sin(i/rate)*.5
            q = np.tile([np.cos(angle/2),0,np.sin(angle/2),0],(5,1))
            target.append((clock.now+round(i/rate*1e9),q,np.zeros(3)))
    session.initialize()
    def status(payload):
        return {'records_root':str(tmp_path),'last_recording_id':session.session_id,
                'last_recording_status':session.metadata['status'], 'last_recording_error':session.error,
                'last_recording_path':str(session.path), 'saving':not session.done.is_set(),
                'last_export':session.metadata, 'export_progress':session.export_progress}
    presenter = PresenterWindow(load_config(), backend=status)
    presenter.session_id = session.session_id
    events, heartbeats, visible_progress = [], [], []
    original = PairSession.update_export_progress
    def observe(self, percent, phase):
        original(self,percent,phase)
        events.append((self.export_progress['percent'],phase,threading.get_ident()))
    monkeypatch.setattr(PairSession,'update_export_progress',observe)
    timer = QTimer()
    timer.setInterval(10)
    timer.timeout.connect(lambda: heartbeats.append(time.perf_counter()))
    timer.start()
    try:
        clock.now = session.deadline_ns
        recorder.tick()
        def finished():
            visible_progress.append(presenter.export_progress.value())
            return session.done.is_set() and presenter.session_id is None
        pump_until(app,finished,timeout=20)
        assert not session.error,session.error
        assert len(heartbeats) >= 5
        assert max(np.diff(heartbeats)) < .5
        assert any(0 < value < 100 for value in visible_progress)
        assert presenter.export_progress.value() == 100
        exported = [e for e in events if e[0] > 0]
        assert len({e[2] for e in exported}) == 1
        assert exported[0][2] != threading.get_ident()
        assert [e[0] for e in events] == sorted(e[0] for e in events)
        assert any('custom.bvh' in e[1] for e in events) and any('noitom.bvh' in e[1] for e in events)
        a,fa,dt = read_bvh(session.path/'custom.bvh')
        b,fb,other_dt = read_bvh(session.path/'noitom.bvh')
        assert fa.shape == fb.shape == (15001,45) and dt == other_dt == .02
        assert (len(fa)-1)*dt == 300
        assert len(list(csv.DictReader((tmp_path/'records.csv').open(encoding='utf-8')))) == 1
    finally:
        timer.stop()
        session.done.wait(20)
        presenter.close()


def test_start_rejection_is_visible_on_participant_screen(tmp_path):
    app = QApplication.instance() or QApplication([])
    def reject(payload):
        raise ValueError('Нет свежих кадров Noitom')
    presenter = PresenterWindow(load_config(), backend=reject)
    presenter.fullscreen_check.setChecked(False)
    try:
        presenter._start()
        assert presenter.participant_window.isVisible()
        assert presenter.participant_window.caption.text() == 'Запись не началась'
        assert presenter.participant_window.cue.text() == 'Нет свежих кадров Noitom'
        assert presenter.loop is None and not presenter.starting
    finally:
        presenter.participant_window.close()
        presenter.close()
