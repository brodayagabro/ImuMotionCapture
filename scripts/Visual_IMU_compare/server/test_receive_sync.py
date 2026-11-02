import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from receive_sync import plan_frames, read_timestamps, synchronize_received
from visual_mocap_server import Frame, select_frames


def test_reuses_original_matching_with_optional_all_camera_requirement():
    frames = {"a": [Frame(0, 1, 1, None)], "b": [Frame(0, 1.01, 1.01, None)], "c": []}
    assert len(select_frames(frames, 1.02, .5, .05)) == 2
    assert not select_frames(frames, 1.02, .5, .05, require_all=True)


def test_manual_offset_sign_and_fixed_rate_mapping():
    a = [100 + i / 30 for i in range(150)]
    b = [t + .12 for t in a]
    plan = plan_frames({"a": a, "b": b}, 30, 100.5, {"b": -120})
    assert max(abs(x - y) for x, y in zip(plan["indices"]["a"], plan["indices"]["b"])) <= 1
    assert max(plan["skews_ms"]) <= 50
    assert np.diff(plan["times"]) == pytest.approx(np.full(len(plan["times"]) - 1, 1 / 30))


def test_output_tick_between_two_arrivals_keeps_previous_complete_group():
    frames = {"a": [Frame(0, 1.0001, 1.0001, None), Frame(1, 1.0666, 1.0666, None)],
              "b": [Frame(0, 1, 1, None)]}
    group = select_frames(frames, 1.06665, .5, .05, require_all=True)
    assert {name: f.sequence for name, f in group.items()} == {"a": 0, "b": 0}


def test_internal_stall_is_not_removed_or_compressed():
    a = [i / 30 for i in range(180)]
    b = [t for t in a if not 2 <= t < 3]
    with pytest.raises(ValueError, match="No matching camera frames"):
        plan_frames({"a": a, "b": b}, 30, .1)


def test_repeated_frame_is_countable_not_hidden():
    times = {"a": [i / 30 for i in range(120)], "b": [i / 15 for i in range(60)]}
    plan = plan_frames(times, 30, .1, max_skew_ms=80)
    assert len(set(plan["indices"]["b"])) < len(plan["times"])


def test_missing_or_corrupt_timestamps_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="Record a new take"):
        synchronize_received(tmp_path, {}, lambda *args: None)
    path = tmp_path / "cam.timestamps.csv"
    path.write_text("frame_index,receive_monotonic_s\n0,1\n2,1\n")
    with pytest.raises(ValueError, match="Incomplete"):
        read_timestamps(path)


def test_real_videos_align_without_light_change_and_keep_originals(tmp_path):
    raw, capture = tmp_path / "raw_videos", tmp_path / "rtsp_capture"
    raw.mkdir(); capture.mkdir()
    cameras = {}
    originals = {}
    for name, delay in [("a", 0), ("b", .15)]:
        stem = f"take.id-{name}.idx-0"
        source = raw / f"{stem}.mp4"
        writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 20, (64, 48))
        assert writer.isOpened()
        for i in range(90):
            # A moving patch identifies source frames; no common flash.
            frame = np.full((48, 64, 3), 25, np.uint8)
            frame[10:20, i % 50:i % 50 + 8] = 100
            writer.write(frame)
        writer.release()
        originals[name] = source.read_bytes()
        timing_file = f"rtsp_capture/{name}.timestamps.csv"
        with (tmp_path / timing_file).open("w") as stream:
            csv_writer = csv.writer(stream)
            csv_writer.writerow(["frame_index", "receive_monotonic_s"])
            csv_writer.writerows((i, 100 + delay + i / 20) for i in range(90))
        cameras[name] = dict(fps=21.7, nominal_fps=20, video_name=stem, timing_file=timing_file)
    (tmp_path / "capture_report.json").write_text(json.dumps(dict(cameras=cameras)))
    (tmp_path / "capture_timing.json").write_text(json.dumps(dict(
        recording_start_monotonic_s=100.4, time_basis="server_monotonic_after_decode_log_received")))
    result = synchronize_received(tmp_path, dict(camera_offsets_ms={"b": -150}), lambda *args: None)
    assert result["method"] == "server_receive" and not result["hardware_capture_timestamps"]
    assert result["measured_max_skew_ms"] <= 50
    lengths = []
    for name, data in cameras.items():
        assert (raw / f"{data['video_name']}.mp4").read_bytes() == originals[name]
        cap = cv2.VideoCapture(str(tmp_path / "synchronized_videos" / f"synced_{data['video_name']}.mp4"))
        assert cap.get(cv2.CAP_PROP_FPS) == pytest.approx(20)
        lengths.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        cap.release()
    assert lengths == [result["synchronized_frame_count"]] * 2
    assert (tmp_path / result["frame_map"]).is_file()
    assert not list(tmp_path.glob(".receive-sync-*"))
    from freemocap_recording_worker import processing_timeline
    processing_timeline(tmp_path)
    provenance = json.loads((tmp_path / "synchronized_videos/timestamps/provenance.json").read_text())
    assert provenance["fps"] == pytest.approx(20)
    assert provenance["source"].startswith("Server reception")
