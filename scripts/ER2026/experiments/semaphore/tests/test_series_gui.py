"""Five-stage measured calibration and preparation-time regression tests."""
import pytest
from PyQt6.QtWidgets import QApplication
from scripts.ER2026.experiments.semaphore.tests.test_series_calibration import physical_series_captures

pytestmark = pytest.mark.gui


def test_wizard_back_retry_restart_timeout_and_cancel(monkeypatch):
    from types import SimpleNamespace
    from scripts.ER2026.experiments.semaphore import guided_dialog
    app = QApplication.instance() or QApplication([])
    clock = [100.]
    monkeypatch.setattr(guided_dialog, 'time', SimpleNamespace(perf_counter=lambda: clock[0]))
    dialog = guided_dialog.GuidedCalibrationDialog(lambda: {}, sample_series=True)
    results, events = [], []
    dialog.result_ready.connect(results.append)
    dialog.diagnostic_event.connect(lambda name, data: events.append(name))
    try:
        dialog._start_capture()
        values = physical_series_captures()['neutral_a_pose'].average
        for i in range(1, 91):
            clock[0] = 103 + i*.02
            dialog.feed_measurement(values, (i, i*20), clock[0])
        assert dialog.stage_index == 1 and len(dialog.captures) == 1
        dialog._repeat_pose()
        assert dialog.frame_recorder.accepted == 0 and len(dialog.captures) == 1
        dialog._previous_pose()
        assert dialog.stage_index == 0 and not dialog.captures
        clock[0] += 124
        dialog._tick()
        assert dialog.phase == 'idle' and dialog.capture_button.isEnabled()
        dialog._restart_calibration()
        assert dialog.stage_index == 0 and dialog.phase == 'preparing'
        dialog.reject()
        assert not results and not dialog.timer.isActive()
        assert {'pose_repeated', 'pose_back', 'pose_rejected', 'restarted', 'cancelled'} <= set(events)
    finally:
        dialog.close()


def test_every_pose_gets_three_seconds_and_transition_packets_are_excluded(monkeypatch):
    from types import SimpleNamespace
    from scripts.ER2026.experiments.semaphore import guided_dialog
    from scripts.ER2026.experiments.semaphore.calibration_protocol import POSE_SEQUENCE
    app = QApplication.instance() or QApplication([])
    clock = [100.]
    monkeypatch.setattr(guided_dialog, 'time', SimpleNamespace(perf_counter=lambda: clock[0]))
    dialog = guided_dialog.GuidedCalibrationDialog(lambda: {}, sample_series=True)
    captures = physical_series_captures()
    results = []
    dialog.result_ready.connect(results.append)
    try:
        dialog._start_capture()
        for stage, name in enumerate(POSE_SEQUENCE):
            start = clock[0]
            assert dialog.stage_index == stage and dialog.phase == 'preparing'
            assert dialog.progress.maximum() == 1500 and dialog.progress.value() == 0
            assert 'мс' in dialog.progress.format()
            for i in range(1, 150):
                clock[0] = start + i*.02
                # Alternating distant poses simulate movement during preparation.
                movement = captures['semaphore_u' if i % 2 else 'neutral_a_pose'].average
                dialog.feed_measurement(movement, (i,i*20), clock[0])
                dialog._tick()
                assert dialog.phase == 'preparing' and dialog.frame_recorder.accepted == 0
            clock[0] = start + 3.
            dialog._tick()
            assert dialog.phase == 'settling'
            dialog.feed_measurement(captures['semaphore_u'].average, (150,3000), start+2.999)
            assert dialog.frame_recorder.frames_seen == 0  # queued pre-deadline packet
            for i in range(1, 91):
                clock[0] = start + 3. + i*.02
                dialog.feed_measurement(captures[name].average, (150+i,3000+i*20), clock[0])
                if i == 89:
                    assert dialog.stage_index == stage and dialog.progress.value() == 1480
            measured = dialog.captures[name]
            assert set(measured.sample_counts.values()) == {76}
            assert measured.started_s >= start+3.
            assert measured.frame_diagnostics['capture_duration_s'] == pytest.approx(1.5)
            assert measured.frame_diagnostics['required_duration_s'] == 1.5
            assert measured.frame_diagnostics['preparation_duration_s'] == 3.
        assert dialog.phase == 'finished' and len(results) == 1
        assert tuple(results[0].captures) == POSE_SEQUENCE
        assert clock[0] == pytest.approx(124.)  # 3s preparation + 0.3s settling + 1.5s capture, five times
    finally:
        dialog.close()
