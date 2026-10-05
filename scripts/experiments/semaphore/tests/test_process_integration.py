"""Full 120-trial protocol through localhost to a recorder in another process."""
import csv
from dataclasses import replace
import hashlib
import json
import multiprocessing
from pathlib import Path
import time
import uuid

import pytest
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from scripts.experiments.semaphore.config import load_config, generate_trials
from scripts.experiments.semaphore.experiment_controller import ExperimentController, State
from scripts.experiments.semaphore.recorder_client import RecorderClient
from scripts.experiments.semaphore.session_log import SessionLog

pytestmark = pytest.mark.gui


def record_process(root, pipe):
    from scripts.noitom_comparison.event_server import EventServer
    from scripts.noitom_comparison.recording import ExperimentRecorder
    recorder = ExperimentRecorder()
    session = recorder.start(root, {})
    server = EventServer(recorder, 0)
    try:
        pipe.send((server.port, str(session.path)))
        pipe.recv()
    finally:
        recorder.stop()
        session.done.wait(5)
        server.close()
        pipe.close()


def test_all_120_trials_to_another_process(tmp_path):
    app = QApplication.instance() or QApplication([])
    ctx = multiprocessing.get_context("spawn")
    parent, child = ctx.Pipe()
    process = ctx.Process(target=record_process, args=(str(tmp_path / "recorder"), child))
    process.start()
    client = RecorderClient()
    log = None
    failures = []
    client.failed.connect(failures.append)
    try:
        assert parent.poll(10), "Recorder did not start"
        port, recorded_path = parent.recv()
        client.connect_recorder(port=port)
        deadline = time.monotonic() + 3
        while not client.ready and time.monotonic() < deadline:
            QTest.qWait(5)
        assert client.ready
        config = replace(load_config(), sessions_dir=str(tmp_path / "presenter"))
        trials = generate_trials(config, 18372)
        log = SessionLog(config, trials, "PTEST", 18372)
        events = []
        def emit(payload):
            payload = {**payload, "event_id": uuid.uuid4().hex,
                       "recording_session_id": client.recording_session_id}
            log.append(payload)
            events.append(payload)
            assert client.send_event(payload), failures
        client.acknowledged.connect(log.acknowledge)
        now = time.monotonic_ns()
        c = ExperimentController(config, trials, log.context, emit, clock=lambda: now)
        c.set_recorder_ready(True)
        c.start()
        steps = 0
        while c.active:
            if c.state == State.READY:
                c.start_block()
            else:
                now = c.phase_started_ns + c.duration_ns
                c.tick()
            steps += 1
            if steps % 10 == 0:
                QTest.qWait(5)
            # Virtual time advances thousands of times faster than the real
            # protocol. Let the real IPC drain without testing an accidental
            # network throughput limit on a busy Windows host.
            deadline = time.monotonic() + 3
            while len(client.pending) > 64 and time.monotonic() < deadline:
                QTest.qWait(5)
            assert not failures
        deadline = time.monotonic() + 3
        while client.pending and time.monotonic() < deadline:
            QTest.qWait(5)
        assert not client.pending and not failures
        log.close()
        parent.send("stop")
        process.join(5)
        assert process.exitcode == 0
        with (Path(recorded_path) / "events.csv").open(encoding="utf-8", newline="") as handle:
            rows = [r for r in csv.DictReader(handle) if "event_id" in json.loads(r["details_json"])]
        assert len(rows) == len(events) == log.metadata["ack_count"]
        assert [json.loads(r["details_json"]) for r in rows] == events
        assert sum(e["type"] == "trial_end" and e["outcome"] == "completed" for e in events) == 120
        assert sum(e["type"] == "block_start" for e in events) == 2
        assert log.metadata["protocol_sha256"] == hashlib.sha256((log.path / "protocol.json").read_bytes()).hexdigest()
        assert log.metadata["status"] == "completed"
    finally:
        client.close()
        if process.is_alive():
            parent.send("stop")
            process.join(5)
        if process.is_alive():
            process.terminate()
            process.join(3)
        if log:
            log.close()
        parent.close()
        child.close()
