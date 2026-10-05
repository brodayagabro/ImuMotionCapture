from dataclasses import replace
import csv
import json
import threading
import time
from scripts.noitom_comparison.noitom_client import FakeNoitomClient, NoitomReceiverThread
from scripts.noitom_comparison.recording import ExperimentRecorder


def rows(session, stream):
    return [json.loads(line) for line in (session.path / f"{stream}.jsonl").read_text(encoding="utf-8").splitlines()]


def finish(session):
    assert session.done.wait(3), "writer did not finish"
    assert session.error is None
    return json.loads((session.path / "metadata.json").read_text(encoding="utf-8"))


def test_shared_t0_pending_packet_and_independent_indices(tmp_path, monkeypatch):
    recorder = ExperimentRecorder()
    session = recorder.start(tmp_path, {"test": True}, t0_ns=1000)
    monkeypatch.setattr("scripts.noitom_comparison.recording.time.monotonic_ns", lambda: 1100)
    stamp, claim = recorder.stamp_packet()
    recorder.noitom_frame(FakeNoitomClient.frame(89, 1030), {})
    recorder.noitom_frame(FakeNoitomClient.frame(102, 1099), {})
    recorder.stop(stop_ns=1200)
    assert not session.done.is_set()  # delayed Qt processing must still be saved
    recorder.complete_packet(claim, {"payload_base64": "ZGF0YQ=="}, {"frame_index": 1})
    metadata = finish(session)
    assert stamp == 1100
    assert metadata["status"] == "complete"
    assert metadata["recording_t0_ns"] == 1000
    assert metadata["recording_stop_ns"] == 1200
    assert rows(session, "own_frames")[0]["relative_timestamp_ns"] == 100
    assert [row["relative_timestamp_ns"] for row in rows(session, "noitom_frames")] == [30, 99]
    assert [row["sdk_frame_index"] for row in rows(session, "noitom_frames")] == [89, 102]
    assert recorder.stamp_packet()[1] is None


def test_noitom_snapshot_claim_survives_stop(tmp_path):
    recorder = ExperimentRecorder()
    session = recorder.start(tmp_path, {})
    stamp, claim = recorder.stamp_packet()
    frame = replace(FakeNoitomClient.frame(300, stamp), capture_context=claim)
    recorder.stop()
    assert not session.done.is_set()
    recorder.noitom_frame(frame, {})
    assert finish(session)["counts"]["noitom_frames"] == 1


def test_noitom_recording_owns_values_before_source_mutation(tmp_path):
    recorder = ExperimentRecorder()
    session = recorder.start(tmp_path, {})
    frame = FakeNoitomClient.frame(2, time.monotonic_ns())
    orientations = {"spine": frame.joints["Spine2"].rotation.copy()}
    original = frame.joints["Spine2"].rotation.tolist()
    recorder.noitom_frame(frame, orientations)
    frame.joints["Spine2"].rotation[:] = 0
    orientations["spine"][:] = 0
    recorder.stop()
    finish(session)
    saved = rows(session, "noitom_frames")[0]
    assert saved["joints"]["Spine2"]["rotation"] == original
    assert saved["segment_orientations"]["spine"] == original


def test_noitom_snapshot_failure_releases_claim_and_marks_incomplete(tmp_path):
    recorder = ExperimentRecorder()
    session = recorder.start(tmp_path, {})
    _, claim = recorder.stamp_packet()
    recorder.stop()
    recorder.release_claim(claim, "broken quaternion")
    assert "noitom_snapshot_error" in finish(session)["incomplete_reasons"]


def test_old_frames_excluded_and_sessions_do_not_mix(tmp_path):
    recorder = ExperimentRecorder()
    first = recorder.start(tmp_path, {}, t0_ns=100)
    recorder.noitom_frame(FakeNoitomClient.frame(1, 99), {})
    recorder.stop(stop_ns=200)
    second = recorder.start(tmp_path, {}, t0_ns=300)
    recorder.noitom_frame(FakeNoitomClient.frame(8, 350), {})
    recorder.stop(stop_ns=400)
    assert finish(first)["counts"]["noitom_frames"] == 0
    assert finish(second)["counts"]["noitom_frames"] == 1
    assert first.path != second.path


def test_two_asynchronous_producers_and_immutable_snapshots(tmp_path):
    recorder = ExperimentRecorder()
    session = recorder.start(tmp_path, {"sensor_mapping": {"spine": 2}})
    data = {"segment_orientations": {"spine": [1., 0., 0., 0.]}}
    def own():
        for index in range(101):
            _, claim = recorder.stamp_packet()
            recorder.complete_packet(claim, {"index": index}, {"frame_index": index, **data})
    def noitom():
        for index in range(63):
            recorder.noitom_frame(FakeNoitomClient.frame(index * 2, time.monotonic_ns()), {})
    threads = [threading.Thread(target=own), threading.Thread(target=noitom)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    data["segment_orientations"]["spine"][0] = 42
    recorder.event("noitom_disconnect", incomplete=True)
    recorder.stop()
    metadata = finish(session)
    assert metadata["counts"] == {"own_frames": 101, "own_packets": 101, "noitom_frames": 63}
    assert metadata["status"] == "incomplete"
    assert rows(session, "own_frames")[0]["segment_orientations"]["spine"][0] == 1
    with (session.path / "events.csv").open(encoding="utf-8", newline="") as handle:
        assert "noitom_disconnect" in [row["event"] for row in csv.DictReader(handle)]


def test_disk_failure_is_visible_without_blocking_producers(tmp_path):
    root = tmp_path / "file"
    root.write_text("not a directory")
    recorder = ExperimentRecorder()
    session = recorder.start(root, {})
    assert session.done.wait(3)
    assert session.error
    recorder.stop()
    assert recorder.stamp_packet()[1] is None


def test_fake_worker_records_without_render_or_qt(tmp_path):
    recorder = ExperimentRecorder()
    worker = NoitomReceiverThread(FakeNoitomClient(rate_hz=120), recorder)
    session = recorder.start(tmp_path, {})
    worker.start()
    try:
        worker.command("connect")
        worker.command("start")
        deadline = time.monotonic() + 2
        frames = 0
        while frames < 5 and time.monotonic() < deadline:
            kind, _, _ = worker.events.get(timeout=1)
            frames += kind == "frame"
        assert frames >= 5
        with worker.capture_lock:
            recorder.stop()
        assert finish(session)["counts"]["noitom_frames"] >= 5
        assert "noitom_mapping_error" not in session.metadata["incomplete_reasons"]
        assert len(rows(session, "noitom_frames")[0]["segment_orientations"]) == 5
    finally:
        worker.command("shutdown")
        worker.join(3)
        recorder.stop()
        assert not worker.is_alive()
