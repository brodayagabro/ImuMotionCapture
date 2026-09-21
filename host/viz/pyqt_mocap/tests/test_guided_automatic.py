"""Drive the entire wizard with simulated time and fresh sensor generations."""

from types import SimpleNamespace

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
