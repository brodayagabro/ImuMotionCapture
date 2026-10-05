import csv
from dataclasses import replace
import json
import time

import pytest
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from scripts.experiments.semaphore.config import load_config, Timing
from scripts.experiments.semaphore.experiment_controller import State
from scripts.experiments.semaphore.presenter_window import PresenterWindow
from scripts.experiments.semaphore.recorder_client import RecorderClient
from scripts.experiments.semaphore.semaphore_model import SEMAPHORE_POSES
from scripts.noitom_comparison.event_server import EventServer
from scripts.noitom_comparison.recording import ExperimentRecorder

pytestmark = pytest.mark.gui


def pump_until(app, predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return
        QTest.qWait(5)
    raise AssertionError("Timed out")


def test_two_windows_training_live_ack_and_recorded_timestamps(tmp_path):
    app = QApplication.instance() or QApplication([])
    recorder = ExperimentRecorder()
    server = EventServer(recorder, 0)
    config = replace(load_config(), sessions_dir=str(tmp_path / "presenter"),
                     timing=Timing(.01, .01, .02, .02, .01, .01))
    window = PresenterWindow(config, port=server.port, seed=21)
    session = None
    try:
        window.show()
        assert not window.buttons["start"].isEnabled()
        window.handedness.setCurrentIndex(window.handedness.findData("left"))
        window.participant_notes.setText("Тестовая карточка")
        window._save_participant_profile()
        pump_until(app, lambda: not window.dataset_tab.busy)
        assert window.pose_browser.letters.count() == 28
        window.pose_browser.letters.setCurrentText("У")
        assert window.participant_window.avatar.pose == SEMAPHORE_POSES["У"]
        window.connect_recorder()
        pump_until(app, lambda: "CONNECTED · включите" in window.connection_label.text())
        assert not window.client.ready and not window.buttons["start"].isEnabled()
        session = recorder.start(tmp_path / "recorder", {})
        pump_until(app, lambda: window.client.ready)
        window.start_session(training=True)
        c = window.controller
        assert c.state == State.READY and len(c.trials) == 4
        assert window.participant_window.isVisible()
        window.action("start_block")
        pump_until(app, lambda: c.state == State.HOLD)
        assert window.participant_window.word.word == c.trial.word
        assert window.participant_window.word.index == c.letter_index
        window.action("pause")
        fraction, pose = c.fraction(), c.pose()
        QTest.qWait(80)
        assert c.state == State.PAUSED and c.fraction() == fraction and c.pose() == pose
        window.action("resume")
        pump_until(app, lambda: c.state == State.READY)
        assert c.trial.block == "upper"  # the second block waits for the operator
        window.action("start_block")
        pump_until(app, lambda: c.state == State.FINISHED and not window.client.pending)
        local_path = window.session_log.path
        window.close()
        recorder.stop()
        assert session.done.wait(3) and not session.error
        with (local_path / "presenter_events.csv").open(encoding="utf-8", newline="") as handle:
            local_events = [json.loads(r["details_json"]) for r in csv.DictReader(handle)]
        with (session.path / "events.csv").open(encoding="utf-8", newline="") as handle:
            saved_events = [r for r in csv.DictReader(handle) if "event_id" in json.loads(r["details_json"])]
        assert [json.loads(r["details_json"]) for r in saved_events] == local_events
        assert all(e["training_trial"] for e in local_events)
        assert sum(e["type"] == "trial_end" for e in local_events) == 4
        metadata = json.loads((local_path / "metadata.json").read_text(encoding="utf-8"))
        assert metadata["status"] == "completed" and metadata["ack_count"] == len(local_events)
        assert metadata["participant_info"] == {"handedness": "left", "notes": "Тестовая карточка"}
        with (local_path / "recorder_acks.csv").open(encoding="utf-8", newline="") as handle:
            acks = {r["event_id"]: r for r in csv.DictReader(handle)}
        for row in saved_events:
            payload = json.loads(row["details_json"])
            assert acks[payload["event_id"]]["host_timestamp_ns"] == row["host_timestamp_ns"]
            assert int(row["host_timestamp_ns"]) >= payload["local_monotonic_ns"]
        with (local_path / "trial_order.csv").open(encoding="utf-8", newline="") as handle:
            assert len(list(csv.DictReader(handle))) == 4
    finally:
        window.close()
        pump_until(app, lambda: not window.client.pending)
        pump_until(app, lambda: not window.timer.isActive())
        assert (tmp_path / "presenter" / "dataset" / "dataset_index.csv").is_file()
        recorder.stop()
        if session:
            assert session.done.wait(3)
        server.close()


def test_disconnect_pauses_and_repeat_recovers(tmp_path):
    app = QApplication.instance() or QApplication([])
    recorder = ExperimentRecorder()
    session = recorder.start(tmp_path / "recorder", {})
    server = EventServer(recorder, 0)
    window = PresenterWindow(replace(load_config(), sessions_dir=str(tmp_path / "presenter")), port=server.port)
    try:
        window.connect_recorder()
        pump_until(app, lambda: window.client.ready)
        window.start_session()
        window.action("start_block")
        pump_until(app, lambda: not window.client.pending)
        server.close()
        pump_until(app, lambda: window.controller.state == State.PAUSED)
        assert window.controller.recovery_required
        assert not window.buttons["resume"].isEnabled()
        server = EventServer(recorder, 0)
        window.port = server.port
        window.connect_recorder()
        pump_until(app, lambda: window.client.ready)
        window._tick()
        window.action("repeat_trial")
        assert window.controller.state == State.PREPARE and window.controller.attempt == 2
        window.close()  # emits final events, waits for ACK asynchronously
        pump_until(app, lambda: window.session_log.closed)
        assert window.session_log.metadata["status"] == "incomplete"
    finally:
        window.close()
        pump_until(app, lambda: not window.client.pending)
        pump_until(app, lambda: not window.timer.isActive())
        recorder.stop()
        assert session.done.wait(3)
        server.close()


def test_offline_requires_explicit_debug_override(tmp_path):
    app = QApplication.instance() or QApplication([])
    window = PresenterWindow(replace(load_config(), sessions_dir=str(tmp_path)), debug=True)
    try:
        assert not window.buttons["start"].isEnabled()
        window.offline.setChecked(True)
        window._refresh()
        assert window.buttons["start"].isEnabled()
        window.start_session()
        assert window.controller.offline and len(window.controller.trials) == 120
        window.action("start_block")
        window.participant_window.close()
        assert window.controller.state == State.PAUSED and window.controller.recovery_required
        assert window.session_log.metadata["offline"] is True
    finally:
        window.close()
        pump_until(app, lambda: not window.timer.isActive())


def test_tcp_open_without_handshake_never_becomes_ready():
    from PyQt6.QtNetwork import QTcpServer, QHostAddress
    app = QApplication.instance() or QApplication([])
    server = QTcpServer()
    assert server.listen(QHostAddress(QHostAddress.SpecialAddress.LocalHost), 0)
    client = RecorderClient(timeout_ms=150)
    failures = []
    client.failed.connect(failures.append)
    try:
        client.connect_recorder(port=server.serverPort())
        pump_until(app, lambda: server.hasPendingConnections())
        peer = server.nextPendingConnection()
        assert not client.ready
        pump_until(app, lambda: bool(failures))
        assert not client.ready
        peer.close()
    finally:
        client.close()
        server.close()


def test_missing_ack_fails_even_while_peer_keeps_sending_ready():
    from PyQt6.QtCore import QTimer
    from PyQt6.QtNetwork import QTcpServer, QHostAddress
    app = QApplication.instance() or QApplication([])
    server = QTcpServer()
    assert server.listen(QHostAddress(QHostAddress.SpecialAddress.LocalHost), 0)
    client = RecorderClient(timeout_ms=250)
    failures = []
    client.failed.connect(failures.append)
    heartbeat = QTimer()
    try:
        client.connect_recorder(port=server.serverPort())
        pump_until(app, lambda: server.hasPendingConnections())
        peer = server.nextPendingConnection()
        ready = b'{"type":"ready","protocol_version":1,"ready":true,"recording_session_id":"test"}\n'
        peer.write(ready)
        heartbeat.timeout.connect(lambda: peer.write(ready))
        heartbeat.start(30)
        pump_until(app, lambda: client.ready)
        assert client.send_event({"type": "test_event", "event_id": "lost", "recording_session_id": "test"})
        pump_until(app, lambda: bool(failures))
        assert not client.ready and not client.pending
        assert "подтверждения" in failures[0]
        peer.close()
    finally:
        heartbeat.stop()
        client.close()
        server.close()
