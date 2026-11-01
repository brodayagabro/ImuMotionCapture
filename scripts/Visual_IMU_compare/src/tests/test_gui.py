import json
import socket
import time
import pytest
from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QApplication
from pyqt_mocap import mpu_udp_viewer
from scripts.Visual_IMU_compare.src.config import ComparisonConfig
from scripts.Visual_IMU_compare.src.comparison_window import MocapComparisonWindow

pytestmark = pytest.mark.gui


def test_semaphore_visual_only_over_udp_exports_and_invalidates_calibration(tmp_path, monkeypatch, packet_pair):
    monkeypatch.setattr(mpu_udp_viewer, 'QSettings', lambda *args: QSettings(
        str(tmp_path/'visual.ini'), QSettings.Format.IniFormat))
    app = QApplication.instance() or QApplication([])
    with socket.socket(type=socket.SOCK_DGRAM) as reservation:
        reservation.bind(('127.0.0.1', 0)); port = reservation.getsockname()[1]
    window = MocapComparisonWindow(ComparisonConfig(bind='127.0.0.1',port=port,recordings_dir=str(tmp_path/'records')))
    source = socket.socket(type=socket.SOCK_DGRAM)
    index = 0
    last_send = 0.

    def publish():
        nonlocal index, last_send
        if time.monotonic()-last_send < .04:
            return
        last_send=time.monotonic(); index+=1
        packet_pair['frame'].update(frame_id=index,timestamp_monotonic_s=index*.04)
        source.sendto(json.dumps(packet_pair['skeleton']).encode(),('127.0.0.1',port))
        source.sendto(json.dumps(packet_pair['frame']).encode(),('127.0.0.1',port))

    def pump(predicate, *, send=True, timeout=5):
        deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            if send: publish()
            app.processEvents()
            if predicate(): return
            time.sleep(.005)
        raise AssertionError('GUI did not reach the expected state')

    try:
        assert window.settings_application_name != 'MPU UDP Viewer ER2026'
        window.own_record_check.setChecked(False)
        window.visual_worker.command('connect'); window.visual_worker.command('start')
        pump(lambda: window._visual_fresh())
        window.open_experiment_presenter()
        presenter=window.presenter
        presenter.fullscreen_check.setChecked(False)
        presenter.duration_spin.setValue(1.)
        presenter._poll()
        assert not presenter.start_button.isEnabled()
        presenter.visual_button.click()
        assert presenter.start_button.isEnabled()
        assert window.visual_calibration['stream_id']==packet_pair['frame']['stream_id']
        presenter.start_button.click()
        pump(lambda: window.session is not None and window.session.done.is_set())
        assert window.session.error is None, window.session.error
        assert (window.session.path/'visual.bvh').is_file()
        assert not (window.session.path/'custom.bvh').exists()
        presenter._poll()
        assert presenter.records_table.rowCount()==1
        assert presenter.records_table.horizontalHeaderItem(4).text()=='visual.bvh'
        assert 'Axis Studio' not in presenter.visual_button.text()
        packet_pair['skeleton']['stream_id']=packet_pair['frame']['stream_id']='new-stream'
        pump(lambda: window.latest_visual_frame.stream_id=='new-stream')
        assert window.visual_calibration is None
        presenter._poll()
        assert not presenter.start_button.isEnabled()
        pump(lambda: not window._visual_fresh(),send=False,timeout=2)
        window._update_comparison_status()
        assert not window.human_canvas.visual_joints.visible()
    finally:
        source.close()
        window.close()
        pump(lambda: window._allow_close,send=False)
