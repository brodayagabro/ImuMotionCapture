"""Drive the entire wizard with simulated time and fresh sensor generations."""

from types import SimpleNamespace
import math
import json

import pytest
from PyQt6.QtWidgets import QApplication, QMessageBox

from pyqt_mocap import guided_dialog
from pyqt_mocap.calibration import POSE_NAMES
from pyqt_mocap.tests import test_calibration as fixtures

pytestmark = pytest.mark.gui


def test_one_click_records_five_stages(monkeypatch):
    app = QApplication.instance() or QApplication([])
    clock = [100.]
    monkeypatch.setattr(guided_dialog, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    fixture = fixtures.GuidedCalibrationTests()
    fixture.setUp()
    generation = [0]

    def snapshot():
        generation[0] += 1
        pose = fixture.captures[POSE_NAMES[min(dialog.stage_index, len(POSE_NAMES) - 1)]]
        return {name: (q, clock[0], generation[0]) for name, q in pose.average.items()}

    dialog = guided_dialog.GuidedCalibrationDialog(snapshot)
    results = []
    dialog.result_ready.connect(results.append)
    dialog.show()
    app.processEvents()  # Also paint the vector examples.
    try:
        dialog.capture_button.click()
        assert dialog.phase == "preparing"
        assert dialog.recorder is None
        assert not dialog.capture_button.isEnabled()
        for stage in range(len(POSE_NAMES)):
            assert dialog.stage_index == stage
            clock[0] = dialog.phase_started_s + 2.5
            dialog._tick()
            app.processEvents()
            assert dialog.recorder is None
            assert dialog.preview.fraction == pytest.approx(.5)
            clock[0] = dialog.phase_started_s + 5.
            dialog._tick()
            assert dialog.phase == "recording"
            start = clock[0]
            for index in range(1, 51):
                clock[0] = start + index * .1
                dialog._tick()
            assert list(dialog.captures) == list(POSE_NAMES[:stage + 1])
        assert dialog.phase == "finished"
        assert not dialog.timer.isActive()
        assert len(results) == 1
        assert results[0].reference_s > dialog.captures["return_a_pose"].started_s
        assert max(results[0].scores_deg.values()) < 1e-6
    finally:
        dialog.reject()


def test_missing_data_pauses_and_cancel_stops_timer(monkeypatch):
    app = QApplication.instance() or QApplication([])
    clock = [100.]
    monkeypatch.setattr(guided_dialog, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args))
    dialog = guided_dialog.GuidedCalibrationDialog(lambda: {})
    dialog.capture_button.click()
    clock[0] += 5.
    dialog._tick()
    assert warnings
    assert dialog.stage_index == 0
    assert dialog.capture_button.isEnabled()
    assert not dialog.timer.isActive()
    assert not dialog.captures
    dialog.capture_button.click()
    assert dialog.timer.isActive()
    dialog.reject()
    assert not dialog.timer.isActive()


def test_unstable_pose_repeats_only_current_stage_without_applying_profile(monkeypatch):
    app = QApplication.instance() or QApplication([])
    clock = [100.]
    monkeypatch.setattr(guided_dialog, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args))
    fixture = fixtures.GuidedCalibrationTests()
    fixture.setUp()
    generation = [0]
    moving = [False]

    def snapshot():
        generation[0] += 1
        values = dict(fixture.captures[POSE_NAMES[min(dialog.stage_index, 4)]].average)
        if moving[0] and dialog.phase == 'recording':
            values['forearm.L'] = fixtures.raw_quaternion_for(
                fixtures.rotation_x(math.radians(8) * (clock[0] - dialog.capture_started_s)), 'forearm.L')
        return {name: (q, clock[0], generation[0]) for name, q in values.items()}

    dialog = guided_dialog.GuidedCalibrationDialog(snapshot)
    results = []
    dialog.result_ready.connect(results.append)
    def record_stage():
        clock[0] = dialog.phase_started_s + 5.
        dialog._tick()
        start = clock[0]
        for index in range(1, 51):
            clock[0] = start + index * .1
            dialog._tick()
    try:
        dialog.capture_button.click()
        record_stage()
        initial = dialog.captures['a_pose']
        moving[0] = True
        record_stage()
        assert warnings and 'Поза нестабильна' in str(warnings[-1])
        assert dialog.stage_index == 1
        assert dialog.captures == {'a_pose': initial}
        assert dialog.phase == 'idle' and not dialog.timer.isActive()
        assert not results
        moving[0] = False
        dialog.capture_button.click()
        for _ in range(4):
            record_stage()
        assert len(results) == 1
        assert dialog.captures['a_pose'] is initial
    finally:
        dialog.reject()


@pytest.mark.parametrize('restart', [False, True])
def test_return_mismatch_keeps_wizard_and_evidence_then_applies_only_valid_result(monkeypatch, restart):
    from pyqt_mocap.tests.test_t_pose_calibration import physical_captures
    from pyqt_mocap.mocap_core import SENSOR_NEUTRAL_AXIS_FRAMES, matrix_to_quaternion
    app = QApplication.instance() or QApplication([])
    clock = [100.]
    monkeypatch.setattr(guided_dialog, 'time', SimpleNamespace(monotonic=lambda: clock[0]))
    warnings = []
    monkeypatch.setattr(QMessageBox, 'warning', lambda *args: warnings.append(args))
    captures = physical_captures()
    captures['return_a_pose'].average['forearm.R'] = matrix_to_quaternion(
        fixtures.rotation_x(math.radians(40.3)) @ SENSOR_NEUTRAL_AXIS_FRAMES['forearm.R'])
    generation = [0]

    def snapshot():
        generation[0] += 1
        pose = captures[dialog.pose_names[min(dialog.stage_index, 2)]]
        return {s: (q, clock[0], generation[0]) for s, q in pose.average.items()}

    dialog = guided_dialog.GuidedCalibrationDialog(snapshot, t_pose_only=True)
    results, events, rejected = [], [], []
    dialog.result_ready.connect(results.append)
    dialog.diagnostic_event.connect(lambda name, detail: events.append((name, detail)))
    dialog.rejected.connect(lambda: rejected.append(True))
    dialog.show()
    app.processEvents()

    def record_stage():
        clock[0] = dialog.phase_started_s + 5.
        dialog._tick()
        start = clock[0]
        for index in range(1, 51):
            clock[0] = start + index * .1
            dialog._tick()

    try:
        dialog.capture_button.click()
        for _ in range(3):
            record_stage()
        assert warnings and not results and not rejected
        assert dialog.isVisible() and dialog.phase == 'idle'
        assert not dialog.timer.isActive() and dialog.stage_index == 2
        assert dialog.capture_button.text() == 'Повторить последнюю A-позу'
        evidence = [detail for name, detail in events if name == 'fit_rejected'][-1]
        assert evidence['closure_error_deg']['forearm.R'] == pytest.approx(40.3)
        assert set(evidence['poses']) == set(dialog.pose_names)
        # Writer receives JSON-ready full snapshots even though no profile exists.
        assert json.loads(json.dumps(evidence))['poses']['return_a_pose']['sample_counts']['forearm.R'] >= 15
        initial = dialog.captures['a_pose']
        captures['return_a_pose'].average['forearm.R'] = captures['a_pose'].average['forearm.R'].copy()
        if restart:
            dialog.restart_button.click()
            assert not dialog.captures and dialog.stage_index == 0
            for _ in range(3):
                record_stage()
            assert dialog.captures['a_pose'] is not initial
        else:
            dialog.capture_button.click()
            record_stage()
            assert dialog.captures['a_pose'] is initial
        assert len(results) == 1 and dialog.phase == 'finished'
        assert max(results[0].closure_error_deg.values()) < 1e-6
    finally:
        dialog.reject()
