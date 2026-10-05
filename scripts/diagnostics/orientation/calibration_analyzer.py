"""Offline quaternion/calibration diagnostics; source recordings are read-only."""
from __future__ import annotations

import argparse
import base64
import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from pyqt_mocap.calibration import NEUTRAL_DIRECTIONS, TARGET_DIRECTIONS
from pyqt_mocap.mocap_core import SEGMENT_NAMES, DEFAULT_SENSOR_MAPPING, axis_map_matrix, mapped_sensor_quaternion
from pyqt_mocap.orientation_frames import SegmentCalibration
from pyqt_mocap.quaternion_utils import (
    angular_distance, average_quaternions, compose, inverse, normalize, relative,
    quaternion_from_rotation_vector, quaternion_to_matrix, matrix_to_quaternion, rotate_vector, twist_angle,
)
from pyqt_mocap.semaphore_calibration import target_direction, base_pose


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def events_from_session(path):
    events_path = Path(path) / "events.csv"
    if not events_path.exists():
        return []
    with events_path.open(encoding="utf-8-sig", newline="") as f:
        return [{**row, "host_timestamp_ns": int(row["host_timestamp_ns"]),
                 "details": json.loads(row["details_json"])} for row in csv.DictReader(f)]


def rejected_calibrations(events):
    """Keep failed attempts separate from applied profiles and repeatability."""
    result = []
    for event in events:
        if event['event'] != 'own_calibration_fit_rejected':
            continue
        detail = event['details']
        poses = detail.get('poses', {})
        for segment, angle in detail.get('closure_error_deg', {}).items():
            result.append({
                'timestamp_ns': event['host_timestamp_ns'], 'segment': segment,
                'sensor_id': detail.get('sensor_mapping', {}).get(segment),
                'closure_error_deg': angle, 'error': detail['error'], 'applied': False,
                'a_started_s': poses.get('a_pose', {}).get('started_s'),
                'a_ended_s': poses.get('a_pose', {}).get('ended_s'),
                'return_a_started_s': poses.get('return_a_pose', {}).get('started_s'),
                'return_a_ended_s': poses.get('return_a_pose', {}).get('ended_s'),
            })
    return result


def compare_fixed_interval(own, noitom, events, start, duration, hold_duration):
    """A calibration/heading change ends the validity of one fixed alignment."""
    changes = [e for e in events if e['host_timestamp_ns'] > start and e['event'] in
               ('own_calibration_state', 'noitom_heading_aligned')]
    stop = min((e['host_timestamp_ns'] for e in changes), default=None)
    if stop is not None and stop < start + duration * 1e9:
        return {'status': 'reference_changed_during_capture', 'valid_until_ns': stop}, [], []
    o = [r for r in own if stop is None or r['t'] < stop]
    n = [r for r in noitom if stop is None or r['t'] < stop]
    comparison, errors = compare_streams(o, n, start, duration)
    comparison['valid_until_ns'] = stop
    return comparison, errors, static_drift(o, start, hold_duration)


def profiles_from_session(path):
    metadata = read_json(Path(path) / "metadata.json")
    events = events_from_session(path)
    candidates = [(metadata["recording_t0_ns"], metadata.get("own_system", {}).get("profile"))]
    candidates += [(e["host_timestamp_ns"], e["details"].get("profile"))
                   for e in events if e["event"] == "own_calibration_state"]
    result, seen = [], set()
    for timestamp, profile in candidates:
        if not profile:
            continue
        key = json.dumps(profile["calibration"], sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        result.append({"profile": profile, "timestamp_ns": timestamp,
                       "source": str(path), "id": profile.get("created_at", str(timestamp))})
    return result, metadata, events


def target_for_pose(name, segment):
    if name.startswith("sem_"):
        return target_direction(name, segment)
    return TARGET_DIRECTIONS.get(name, NEUTRAL_DIRECTIONS)[segment]


def canonical_pose_quaternion(name, segment):
    """Only a comparison gauge: directions alone do not specify anatomical twist."""
    source = NEUTRAL_DIRECTIONS[segment]
    target = target_for_pose(name, segment)
    if target is None:
        return None
    dot = float(np.clip(source @ target, -1., 1.))
    if dot < -1 + 1e-8:
        axis = np.cross(source, np.eye(3)[np.argmin(abs(source))])
        return quaternion_from_rotation_vector(axis / np.linalg.norm(axis) * math.pi)
    return normalize(np.r_[1 + dot, np.cross(source, target)])


def profile_transforms(profile):
    cal = profile["calibration"]
    neutral = cal.get("neutral_raw_quaternions_wxyz") or cal["poses"]["return_a_pose"]["average_quaternions_wxyz"]
    transforms = {}
    for segment in SEGMENT_NAMES:
        q_sensor = mapped_sensor_quaternion(neutral[segment], cal["axis_maps"][segment])
        transforms[segment] = SegmentCalibration.from_pose(q_sensor,
            q_world_alignment=cal.get("axis_alignment_quaternions_wxyz", {}).get(segment, (1., 0., 0., 0.)))
    return transforms


def analyze_profile(profile, profile_id="profile", optimize=False):
    cal = profile["calibration"]
    transforms = profile_transforms(profile)
    enabled = set(profile.get("application", {}).get("enabled_segments", SEGMENT_NAMES))
    rows, summaries, candidates = [], [], {}
    for segment in SEGMENT_NAMES:
        frame = transforms[segment]
        segment_rows, sensors, targets = [], [], []
        for name, pose in cal["poses"].items():
            raw = pose["average_quaternions_wxyz"][segment]
            sensor = mapped_sensor_quaternion(raw, cal["axis_maps"][segment])
            predicted = frame.apply(sensor)
            target = target_for_pose(name, segment)
            if target is None:
                continue
            axis_error = math.degrees(math.acos(float(np.clip(rotate_vector(predicted, NEUTRAL_DIRECTIONS[segment]) @ target, -1, 1))))
            canonical = canonical_pose_quaternion(name, segment)
            error = relative(canonical, predicted)
            twist = twist_angle(error, NEUTRAL_DIRECTIONS[segment])
            row = {"profile_id": profile_id, "segment": segment, "enabled": segment in enabled,
                   "pose": name, "input_norm": float(np.linalg.norm(raw)),
                   "axis_direction_error_deg": axis_error,
                   "canonical_orientation_difference_deg": math.degrees(angular_distance(canonical, predicted)),
                   "twist_vs_canonical_deg": None if twist is None else math.degrees(twist),
                   "absolute_anatomical_twist_verified": False,
                   "spread_deg": pose.get("spread_deg", {}).get(segment), "inconsistent_pose": False}
            segment_rows.append(row)
            sensors.append(sensor)
            targets.append(canonical)
        errors = [r["axis_direction_error_deg"] for r in segment_rows]
        median = float(np.median(errors)) if errors else 0.
        for row in segment_rows:
            row["inconsistent_pose"] = row["enabled"] and row["axis_direction_error_deg"] > max(15., median + 10.)
        rows.extend(segment_rows)
        summaries.append({"profile_id": profile_id, "segment": segment, "enabled": segment in enabled,
                          "q_world_alignment": frame.q_world_alignment,
                          "q_sensor_segment": frame.q_sensor_segment,
                          "mean_axis_error_deg": float(np.mean(errors)) if errors else None,
                          "max_axis_error_deg": max(errors) if errors else None,
                          "basis_determinant": float(np.linalg.det(axis_map_matrix(cal["axis_maps"][segment]))),
                          "estimated_rate_deg_s": math.degrees(float(np.linalg.norm(cal["drift_rates_rad_s"][segment]))),
                          "host_drift_applied_by_current_viewer": False})
        if optimize and segment in enabled and sensors:
            from .reference_math import fit_mounting
            candidates[segment] = fit_mounting(sensors, targets, frame.q_world_alignment, frame.q_sensor_segment)
            candidates[segment]["target_twist_is_assumed_canonical"] = True
            candidates[segment]["deployed"] = False
    return {"id": profile_id, "method": cal.get("method"), "segments": summaries,
            "residuals": rows, "multipose_candidates": candidates}


def repeatability(entries):
    rows = []
    for segment in SEGMENT_NAMES:
        world, mounting = [], []
        for entry in entries:
            p = entry["profile"]
            if segment not in p.get("application", {}).get("enabled_segments", SEGMENT_NAMES):
                continue
            cal = p["calibration"]
            basis = axis_map_matrix(cal["axis_maps"][segment])
            if np.linalg.det(basis) < 0:
                continue  # A reflection is not a quaternion world transform.
            c = profile_transforms(p)[segment]
            b = matrix_to_quaternion(basis)
            world.append(compose(c.q_world_alignment, b))
            mounting.append(compose(inverse(b), c.q_sensor_segment))
        row = {"segment": segment, "repetitions": len(world), "minimum_five_met": len(world) >= 5}
        for label, values in (("world_alignment", world), ("mounting", mounting)):
            mean = average_quaternions(values) if values else None
            errors = [math.degrees(angular_distance(mean, q)) for q in values]
            row[label + "_mean_deviation_deg"] = float(np.mean(errors)) if errors else None
            row[label + "_max_deviation_deg"] = max(errors) if errors else None
        rows.append(row)
    return rows


def sampled_frames(path, source, sample_hz=20.):
    """Stream the original file, retaining only lightweight sampled orientations."""
    rows, next_ns = [], -1
    file = Path(path) / f"{source}_frames.jsonl"
    if not file.exists():
        return rows
    step = round(1e9 / sample_hz)
    with file.open(encoding="utf-8") as handle:
        for line in handle:
            frame = json.loads(line)
            stamp = int(frame["host_timestamp_ns"])
            if stamp < next_ns or frame.get("mapping_error") or frame.get("neutral_pending"):
                continue
            next_ns = stamp + step
            result = {"t": stamp, "q": frame.get("segment_orientations", {}),
                      "enabled": frame.get("enabled_segments", SEGMENT_NAMES),
                      "raw": frame.get("raw_sensor_quaternions", {}),
                      "mapping": frame.get("sensor_mapping", DEFAULT_SENSOR_MAPPING),
                      "heading_offset_rad": frame.get("heading_offset_rad", 0.)}
            if source == "noitom" and "joints" in frame:
                # Shoulder-line heading is independent of any unknown bone twist.
                from scripts.noitom_comparison.noitom_adapter import torso_heading
                joints = {name: SimpleNamespace(**{**joint,
                          "position": None if joint["position"] is None else np.array(joint["position"]),
                          "rotation": np.array(joint["rotation"])}) for name, joint in frame["joints"].items()}
                try:
                    result["torso_heading_rad"] = torso_heading(SimpleNamespace(joints=joints))
                except (ValueError, KeyError):
                    result["torso_heading_rad"] = None
            rows.append(result)
    return rows


def pair_frames(own, noitom, max_delta_ms=40.):
    if not noitom:
        return []
    times = np.array([row["t"] for row in noitom], dtype=np.int64)
    pairs = []
    for row in own:
        i = int(np.searchsorted(times, row["t"]))
        choices = [j for j in (i-1, i) if 0 <= j < len(times)]
        j = min(choices, key=lambda j: abs(int(times[j]) - row["t"]))
        if abs(int(times[j]) - row["t"]) <= max_delta_ms * 1e6:
            pairs.append((row, noitom[j]))
    return pairs


def summarize(values):
    values = [float(v) for v in values if v is not None and math.isfinite(v)]
    return {"n": len(values), "mean_deg": float(np.mean(values)) if values else None,
            "max_deg": max(values) if values else None,
            "p95_deg": float(np.percentile(values, 95)) if values else None}


def compare_streams(own, noitom, static_start_ns, static_duration_s=5., max_delta_ms=40.):
    pairs = pair_frames(own, noitom, max_delta_ms)
    end = static_start_ns + round(static_duration_s * 1e9)
    reference = [(o, n) for o, n in pairs if static_start_ns <= o["t"] < end]
    if len(reference) < 10:
        return {"status": "insufficient_static_reference", "paired_reference_frames": len(reference)}, []
    headings = [n["torso_heading_rad"] + n["heading_offset_rad"] for _, n in reference if n.get("torso_heading_rad") is not None]
    if not headings:
        return {"status": "missing_world_heading", "reason": "shoulder line required to separate world and segment alignment"}, []
    heading = math.atan2(float(np.mean(np.sin(headings))), float(np.mean(np.cos(headings))))
    world = quaternion_from_rotation_vector((0., 0., -heading))
    offsets = {}
    for segment in SEGMENT_NAMES:
        good = [(o, n) for o, n in reference if segment in o["enabled"] and segment in n["enabled"] and segment in o["q"] and segment in n["q"]]
        if len(good) < 10:
            continue
        o_ref = average_quaternions([o["q"][segment] for o, _ in good])
        n_ref = average_quaternions([compose(world, n["q"][segment]) for _, n in good])
        offsets[segment] = compose(inverse(n_ref), o_ref)
    errors = []
    for o, n in pairs:
        if o["t"] < end:
            continue  # Reference frames are never scored as validation data.
        oq, nq = {}, {}
        for segment, offset in offsets.items():
            if segment not in o["enabled"] or segment not in n["enabled"] or segment not in o["q"] or segment not in n["q"]:
                continue
            oq[segment] = o["q"][segment]
            nq[segment] = compose(world, compose(n["q"][segment], offset))
            direction = float(np.clip(rotate_vector(oq[segment], NEUTRAL_DIRECTIONS[segment]) @ rotate_vector(nq[segment], NEUTRAL_DIRECTIONS[segment]), -1, 1))
            twist = twist_angle(relative(nq[segment], oq[segment]), NEUTRAL_DIRECTIONS[segment])
            errors.append({"timestamp_ns": o["t"], "pair_delta_ms": (n["t"]-o["t"])/1e6,
                           "kind": "segment", "segment": segment,
                           "angular_error_deg": math.degrees(angular_distance(nq[segment], oq[segment])),
                           "axis_error_deg": math.degrees(math.acos(direction)),
                           "twist_error_deg": None if twist is None else math.degrees(twist)})
        for side in ("L", "R"):
            upper, forearm = "shoulder."+side, "forearm."+side
            if all(s in oq and s in nq for s in (upper, forearm)):
                error = angular_distance(relative(oq[upper], oq[forearm]), relative(nq[upper], nq[forearm]))
                errors.append({"timestamp_ns": o["t"], "pair_delta_ms": (n["t"]-o["t"])/1e6,
                               "kind": "joint", "segment": "elbow."+side, "angular_error_deg": math.degrees(error),
                               "axis_error_deg": None, "twist_error_deg": None})
    report = {"status": "aligned", "q_world_alignment": world.tolist(),
              "q_noitom_segment_to_own_segment": {s: q.tolist() for s, q in offsets.items()},
              "reference_start_ns": static_start_ns, "reference_end_ns": end,
              "paired_reference_frames": len(reference), "paired_frames": len(pairs),
              "max_pair_delta_ms": max_delta_ms, "alignment_is_constant": True,
              "warning": "Host reception timestamps; no hardware synchronization. Static A-pose and fixed sensor mounting assumed.",
              "segments": {}}
    for segment in (*SEGMENT_NAMES, "elbow.L", "elbow.R"):
        group = [r for r in errors if r["segment"] == segment]
        report["segments"][segment] = {"orientation": summarize([r["angular_error_deg"] for r in group]),
            "direction": summarize([r["axis_error_deg"] for r in group]),
            "twist_abs": summarize([abs(r["twist_error_deg"]) if r["twist_error_deg"] is not None else None for r in group])}
    return report, errors


def static_drift(own, start_ns, duration_s):
    rows = [r for r in own if start_ns <= r["t"] < start_ns + duration_s * 1e9]
    result = []
    for segment in SEGMENT_NAMES:
        available = [r for r in rows if segment in r["enabled"] and segment in r["q"] and str(r["mapping"][segment]) in r["raw"]]
        if len(available) < 2:
            continue
        initial = available[0]
        raw0 = initial["raw"][str(initial["mapping"][segment])]
        raw = [math.degrees(angular_distance(raw0, r["raw"][str(r["mapping"][segment])])) for r in available]
        output = [math.degrees(angular_distance(initial["q"][segment], r["q"][segment])) for r in available]
        result.append({"segment": segment, "frames": len(available),
                       "duration_s": (available[-1]["t"]-initial["t"])/1e9,
                       "raw_change_max_deg": max(raw), "output_change_max_deg": max(output),
                       "extra_output_motion_suspected": max(output) > max(raw) + 1.})
    return result


def packet_norms(path):
    values, invalid = [], 0
    file = Path(path) / "own_packets.jsonl"
    if not file.exists():
        return {"status": "packets_missing"}
    with file.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            payload = base64.b64decode(row["payload_base64"]).decode("utf-8", errors="replace")
            for record in payload.splitlines():
                parts = record.split()
                if len(parts) == 6 and parts[0] in ("Q", "QUAT"):
                    try:
                        norm = float(np.linalg.norm([float(v) for v in parts[2:]]))
                        if not math.isfinite(norm):
                            raise ValueError()
                        values.append(norm)
                    except ValueError:
                        invalid += 1
    return {"count": len(values), "nonfinite_or_invalid": invalid,
            "min_norm": min(values) if values else None, "max_norm": max(values) if values else None,
            "p95_abs_deviation_from_unit": float(np.percentile(np.abs(np.array(values)-1.),95)) if values else None}


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict,list,tuple)) else v for k,v in row.items()} for row in rows)


def run(session=None, profiles=(), output=Path("output/orientation"), static_start=None,
        static_duration=5., hold_duration=28., sample_hz=20., optimize=False):
    output = Path(output)
    if session and (output.resolve() == Path(session).resolve() or Path(session).resolve() in output.resolve().parents):
        raise ValueError("Diagnostics output must be outside the source recording")
    entries, metadata, events = profiles_from_session(session) if session else ([], {}, [])
    for path in profiles:
        profile = read_json(path)
        entries.append({"profile": profile, "id": str(path), "source": str(path), "timestamp_ns": None})
    # A pre-existing profile present at REC is not a new calibration repetition.
    repetitions = [e for e in entries if not session or e["timestamp_ns"] is None or
                   e["profile"]["calibration"]["poses"]["return_a_pose"]["ended_s"] * 1e9 >= metadata["recording_t0_ns"]]
    methods = sorted({e["profile"]["calibration"].get("method", "unknown") for e in repetitions})
    repeat_rows = [{**row, "method": method} for method in methods
                   for row in repeatability([e for e in repetitions if e["profile"]["calibration"].get("method", "unknown") == method])]
    report = {"source_session": None if session is None else str(session),
              "profile_count": len(entries), "profiles": [], "repeatability": repeat_rows,
              "rejected_calibrations": rejected_calibrations(events),
              "limitations": ["Bone directions do not establish anatomical twist truth.",
                              "Repeatability requires unchanged physical sensor mounting and the same sensor world frame."]}
    for entry in entries:
        report["profiles"].append(analyze_profile(entry["profile"], entry["id"], optimize))
    residuals = [r for p in report["profiles"] for r in p["residuals"]]
    errors = []
    if session:
        own, noitom = sampled_frames(session, "own", sample_hz), sampled_frames(session, "noitom", sample_hz)
        report["sampled_frames"] = {"own": len(own), "noitom": len(noitom)}
        report["input_norms"] = packet_norms(session)
        aligned_events = [e for e in events if e["event"] == "noitom_heading_aligned"]
        start = (metadata["recording_t0_ns"] + round(static_start * 1e9) if static_start is not None else
                 aligned_events[-1]["host_timestamp_ns"] + 2_000_000_000 if aligned_events else None)
        if start is None:
            report["comparison"] = {"status": "static_reference_required", "instruction": "Provide --static-start seconds, or record a heading-aligned event then hold A for 30 seconds."}
        else:
            report["comparison"], errors, report["static_drift"] = compare_fixed_interval(
                own, noitom, events, start, static_duration, hold_duration)
            report["calibration_changes_after_reference"] = [e["id"] for e in entries if e["timestamp_ns"] and e["timestamp_ns"] > start]
            if report["calibration_changes_after_reference"]:
                report["limitations"].append("Calibration changed after reference: comparison stops before that change.")
    output.mkdir(parents=True, exist_ok=True)
    (output / "analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    write_csv(output / "pose_residuals.csv", residuals)
    write_csv(output / "repeatability.csv", report["repeatability"])
    write_csv(output / "comparison_errors.csv", errors)
    write_csv(output / "static_drift.csv", report.get("static_drift", []))
    write_csv(output / "rejected_calibrations.csv", report["rejected_calibrations"])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path)
    parser.add_argument("--profile", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, default=Path("output/orientation"))
    parser.add_argument("--static-start", type=float, help="Known static A-pose start, seconds from recording T0")
    parser.add_argument("--static-duration", type=float, default=5.)
    parser.add_argument("--hold-duration", type=float, default=28.)
    parser.add_argument("--sample-hz", type=float, default=20.)
    parser.add_argument("--compare-multipose", action="store_true")
    args = parser.parse_args()
    if not args.session and not args.profile:
        parser.error("Provide --session or at least one --profile")
    if not all(math.isfinite(x) and x > 0 for x in (args.sample_hz, args.static_duration, args.hold_duration)):
        parser.error("Durations and sample frequency must be positive and finite")
    report = run(args.session, args.profile, args.output, args.static_start,
                 args.static_duration, args.hold_duration, args.sample_hz, args.compare_multipose)
    print(json.dumps({"output": str(args.output), "profiles": report["profile_count"],
                      "comparison": report.get("comparison", {}).get("status"),
                      "five_repetitions": any(r["minimum_five_met"] for r in report["repeatability"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
