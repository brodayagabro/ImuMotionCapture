import csv
import json
import socket
import time

import pytest

from scripts.noitom_comparison.event_server import EventServer, MAX_MESSAGE_BYTES
from scripts.noitom_comparison.recording import ExperimentRecorder


def wire(payload):
    return json.dumps(payload, ensure_ascii=False).encode("utf-8") + b"\n"


def test_fragmented_events_share_recorder_clock_and_session(tmp_path):
    recorder = ExperimentRecorder()
    server = EventServer(recorder, 0)
    session = None
    try:
        with socket.create_connection(("127.0.0.1", server.port), timeout=2) as peer:
            stream = peer.makefile("rb")
            peer.sendall(wire({"type": "hello", "protocol_version": 1}))
            assert json.loads(stream.readline())["ready"] is False
            session = recorder.start(tmp_path, {})
            peer.sendall(wire({"type": "ping"}))
            ready = json.loads(stream.readline())
            assert ready["ready"] and ready["recording_session_id"] == session.session_id
            payloads = [{"type": "arbitrary_stimulus", "event_id": str(i), "word": "ТЕСТ",
                         "local_monotonic_ns": -123, "recording_session_id": session.session_id} for i in range(3)]
            data = b"".join(map(wire, payloads))
            before = time.monotonic_ns()
            peer.sendall(data[:17])
            peer.sendall(data[17:])
            acks = [json.loads(stream.readline()) for _ in payloads]
            after = time.monotonic_ns()
            assert all(a["accepted"] and before <= a["host_timestamp_ns"] <= after for a in acks)
            assert [a["event_id"] for a in acks] == ["0", "1", "2"]
            # The stop and queue insertion use the same lock: no false ACK after stop.
            recorder.stop()
            peer.sendall(wire({**payloads[0], "event_id": "late"}))
            assert json.loads(stream.readline())["accepted"] is False
        assert session.done.wait(3) and not session.error
        with (session.path / "events.csv").open(encoding="utf-8", newline="") as handle:
            rows = [r for r in csv.DictReader(handle) if r["event"] == "arbitrary_stimulus"]
        assert len(rows) == 3
        for row, ack, payload in zip(rows, acks, payloads):
            assert int(row["host_timestamp_ns"]) == ack["host_timestamp_ns"]
            assert int(row["relative_timestamp_ns"]) == ack["host_timestamp_ns"] - session.t0_ns
            assert json.loads(row["details_json"]) == payload
    finally:
        recorder.stop()
        if session:
            assert session.done.wait(3)
        server.close()


@pytest.mark.parametrize("data", [b"[]\n", b"not json\n", b"\xff\n",
    b'{"type":"ping"}\n', wire({"type": "hello", "protocol_version": 99}), b"x" * (MAX_MESSAGE_BYTES + 1)],
    ids=["array", "invalid_json", "invalid_utf8", "no_hello", "version", "oversized"])
def test_malformed_peer_does_not_stop_event_server(data):
    server = EventServer(ExperimentRecorder(), 0)
    try:
        with socket.create_connection(("127.0.0.1", server.port), timeout=2) as peer:
            peer.sendall(data)
            assert json.loads(peer.makefile("rb").readline())["type"] == "error"
        with socket.create_connection(("127.0.0.1", server.port), timeout=2) as peer:
            peer.sendall(wire({"type": "hello", "protocol_version": 1}))
            assert json.loads(peer.makefile("rb").readline())["type"] == "ready"
    finally:
        server.close()


def test_events_cannot_silently_move_to_a_new_recording(tmp_path):
    recorder = ExperimentRecorder()
    first = recorder.start(tmp_path, {})
    recorder.stop()
    second = recorder.start(tmp_path, {})
    try:
        result = recorder.external_event({"type": "stimulus", "recording_session_id": first.session_id}, time.monotonic_ns())
        assert result == {"accepted": False, "reason": "recording_session_changed"}
    finally:
        recorder.stop()
        assert first.done.wait(3) and second.done.wait(3)
