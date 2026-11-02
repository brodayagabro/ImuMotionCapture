"""Offline frame selection using the original live server's reception clock matcher."""
from bisect import bisect_left, bisect_right
from concurrent.futures import ProcessPoolExecutor
import csv
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import tempfile

from visual_mocap_server import Frame, select_frames


def read_timestamps(path):
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    times = [float(row["receive_monotonic_s"]) for row in rows]
    if (len(times) < 12 or any(int(row["frame_index"]) != i for i, row in enumerate(rows))
            or any(not math.isfinite(t) for t in times)
            or any(b <= a for a, b in zip(times, times[1:]))):
        raise ValueError(f"Incomplete or invalid frame timestamps: {path.name}; record a new take")
    return times


def plan_frames(timelines, fps, start, offsets=None, max_skew_ms=50, max_age_ms=500):
    """Replay the old bounded-history matcher on a constant-rate output timeline.

    Unlike live inference, a native multi-camera recording requires every view.
    Repeated selections are explicit in the mapping; interior holes are rejected.
    """
    offsets = offsets or {}
    if (len(timelines) < 2 or not math.isfinite(fps) or not 0 < fps <= 240
            or not math.isfinite(start) or not 0 < max_skew_ms <= max_age_ms <= 5000
            or set(offsets) - set(timelines)
            or any(not math.isfinite(v) or abs(v) > 5000 for v in offsets.values())):
        raise ValueError("Invalid reception synchronization settings")
    frames = {name: [Frame(i, t, t + offsets.get(name, 0) / 1000, None) for i, t in enumerate(times)]
              for name, times in timelines.items()}
    begin = max(start, max(ts[0] for ts in timelines.values()))
    end = min(ts[-1] for ts in timelines.values())
    selections = {name: [] for name in timelines}
    grid, skews, ages = [], [], []
    gap = None
    for index in range(max(0, math.floor((end - begin) * fps) + 1)):
        now = begin + index / fps
        histories = {name: frames[name][bisect_left(times, now - max_age_ms / 1000):bisect_right(times, now)]
                     for name, times in timelines.items()}
        group = select_frames(histories, now, max_age_ms / 1000, max_skew_ms / 1000, require_all=True)
        if len(group) != len(timelines):
            if grid and gap is None:
                gap = now
            continue
        if gap is not None:
            raise ValueError(f"No matching camera frames at {gap - begin:.3f}s. "
                             "Check stream delays / manual offsets; originals are preserved")
        grid.append(now)
        skews.append(1000 * (max(f.aligned for f in group.values()) - min(f.aligned for f in group.values())))
        ages.append(1000 * max(now - f.received for f in group.values()))
        for name, frame in group.items():
            selections[name].append(frame.sequence)
    if len(grid) < 12:
        raise ValueError("Need at least 12 matched frames from every camera; record a longer take")
    return dict(indices=selections, times=grid, skews_ms=skews, ages_ms=ages)


def encode_selected(source, destination, indices, input_count, fps):
    """Decode once, write selected frames at common FPS, and verify sidecar cardinality."""
    import av
    rate = Fraction(str(fps)).limit_denominator(100000)
    position = count = 0
    with av.open(str(source)) as video, av.open(str(destination), "w") as output:
        incoming = video.streams.video[0]
        incoming.codec_context.thread_count = 1
        outgoing = output.add_stream("libx264", rate=rate)
        outgoing.width, outgoing.height = incoming.width, incoming.height
        outgoing.pix_fmt = "yuv420p"
        outgoing.codec_context.thread_count = 1
        outgoing.options = {"preset": "veryfast", "crf": "18", "tune": "zerolatency", "bf": "0"}
        for index, frame in enumerate(video.decode(video=0)):
            count += 1
            while position < len(indices) and indices[position] == index:
                frame.pts, frame.time_base = position, 1 / rate
                for packet in outgoing.encode(frame):
                    output.mux(packet)
                position += 1
        for packet in outgoing.encode():
            output.mux(packet)
    if count != input_count or position != len(indices):
        raise ValueError(f"{source.name}: video has {count} frames, timestamps have {input_count}; "
                         "cannot safely associate frames and timestamps")


def synchronize_received(folder, spec, report):
    capture_path = folder / "capture_report.json"
    timing_path = folder / "capture_timing.json"
    if not capture_path.is_file() or not timing_path.is_file():
        raise ValueError("This take has no server reception timestamps. Record a new take with the updated host")
    capture, timing = json.loads(capture_path.read_text()), json.loads(timing_path.read_text())
    info = capture["cameras"]
    timelines = {name: read_timestamps(folder / data["timing_file"]) for name, data in info.items()}
    # Short fragmented MP4s can report inflated average FPS during startup.
    # Use the H.264 nominal rate for the output grid; selection uses host times.
    fps = min(data.get("nominal_fps", data["fps"]) for data in info.values())
    offsets = spec.get("camera_offsets_ms", {})
    skew, age = spec.get("max_skew_ms", 50), spec.get("max_age_ms", 500)
    report("synchronizing", "Matching frames by server reception time; no flash required")
    plan = plan_frames(timelines, fps, timing["recording_start_monotonic_s"], offsets, skew, age)
    target = folder / "synchronized_videos"
    if target.exists():
        raise ValueError("synchronized_videos already exists without a sync report; preserve/remove it before retrying")
    with tempfile.TemporaryDirectory(prefix=".receive-sync-", dir=folder) as temporary:
        work = Path(temporary)
        with ProcessPoolExecutor(max_workers=min(len(info), os.cpu_count() or 1)) as executor:
            futures = [executor.submit(encode_selected,
                        folder / "raw_videos" / f"{data['video_name']}.mp4",
                        work / f"synced_{data['video_name']}.mp4", plan["indices"][name], len(timelines[name]), fps)
                       for name, data in info.items()]
            for future in futures:
                future.result()
        with (work / "frame_map.csv").open("w") as stream:
            writer = csv.writer(stream)
            writer.writerow(["output_frame", "server_monotonic_s", "skew_ms",
                             *[f"{name}_source_frame" for name in info]])
            writer.writerows([i, t, plan["skews_ms"][i], *[plan["indices"][name][i] for name in info]]
                             for i, t in enumerate(plan["times"]))
        work.rename(target)
    result = dict(method="server_receive", time_basis=timing["time_basis"],
                  hardware_capture_timestamps=False, fps=fps, synchronized_frame_count=len(plan["times"]),
                  max_skew_ms=skew, max_age_ms=age, camera_offsets_ms=offsets,
                  measured_max_skew_ms=max(plan["skews_ms"]), measured_max_age_ms=max(plan["ages_ms"]),
                  start_monotonic_s=plan["times"][0], end_monotonic_s=plan["times"][-1],
                  frame_map="synchronized_videos/frame_map.csv",
                  repeated_frames={name: len(indices) - len(set(indices)) for name, indices in plan["indices"].items()},
                  scope="Server decoded-frame reception times, including transport and decoder delay; not camera exposure synchronization")
    temporary_report = folder / "sync_result.tmp"
    temporary_report.write_text(json.dumps(result, indent=2))
    temporary_report.replace(folder / "sync_result.json")
    return result
