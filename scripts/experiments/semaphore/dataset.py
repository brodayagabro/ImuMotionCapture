"""Rebuildable CSV dataset index; joins IDs and recorder clocks, never dates.

Reads only metadata and event logs, not large motion streams. Existing source
recordings are never modified. Paths in exports are relative to the CSV folder.
"""
import argparse
from collections import defaultdict
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import uuid

from .config import ROOT, validate_participant

INDEX_FIELDS = """participant_id handedness participant_notes participant_profile_source
session_id started_at ended_at training_trial presenter_status presenter_outcome offline
recording_session_id link_status recording_status recording_started_at
presenter_dir recording_dir presenter_metadata_file protocol_file
recorder_metadata_file events_file own_frames_file noitom_frames_file
application_version protocol_sha256 random_seed block_order repetitions mirror_mode
prepare_s word_preview_s transition_s hold_s post_trial_s inter_trial_s
planned_trials recorded_attempts completed_attempts recorded_holds confirmed_events
ack_events local_events missing_recorded_events delivery_issue_count
own_frame_count noitom_frame_count own_frames_present noitom_frames_present
events_present synthetic_noitom recorder_incomplete_reasons""".split()
TRIAL_FIELDS = """participant_id session_id recording_session_id trial_id attempt
block repeat word training_trial prepare_ns start_ns end_ns duration_ns outcome
has_pause boundaries_complete start_event_id end_event_id""".split()
HOLD_FIELDS = """participant_id session_id recording_session_id trial_id attempt
block repeat word training_trial letter_index letter start_ns end_ns duration_ns
hold_completed has_pause boundaries_complete trial_outcome start_event_id end_event_id
recording_status events_file own_frames_file noitom_frames_file""".split()
PARTICIPANT_FIELDS = "participant_id handedness notes profile_source session_count first_session_at last_session_at".split()
UNLINKED_FIELDS = "recording_session_id started_at status recording_dir recorder_metadata_file".split()


def _json(path, warnings):
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(data, dict):
            raise ValueError("Expected object")
        return data
    except (OSError, ValueError) as error:
        warnings.append(f"{path}: {error}")
        return {}


def _csv(path, warnings):
    if not path.exists():
        return []
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    except (OSError, csv.Error, UnicodeError) as error:
        warnings.append(f"{path}: {error}")
        return []


def _session_dirs(root):
    if not root.is_dir():
        return []
    if (root / "metadata.json").is_file():
        return [root]
    return sorted(p for p in root.iterdir() if p.is_dir() and (p / "metadata.json").is_file())


def _relative(path, output):
    if path is None:
        return ""
    try:
        return Path(os.path.relpath(path, output)).as_posix()
    except ValueError:  # different Windows volumes
        return str(path)


def _atomic_csv(path, fields, rows):
    temp = path.with_suffix("." + uuid.uuid4().hex + ".tmp")
    try:
        with temp.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def _boundaries(row):
    start, end = row.get("start_ns"), row.get("end_ns")
    okay = isinstance(start, int) and isinstance(end, int) and end >= start and not row.pop("_duplicate", False)
    row["boundaries_complete"] = okay
    row["duration_ns"] = end - start if okay else ""


def _intervals(events, context):
    """Keep interrupted attempts and unpaired holds, explicitly flagged.

    A pause inside a hold flags the full interval; it is not falsely described
    as continuous holding. Consumers may discard it or split using events.csv.
    """
    trials, holds = {}, {}
    for e in sorted(events, key=lambda value: value["_host_ns"]):
        if type(e.get("trial_id")) is not int or type(e.get("attempt")) is not int:
            continue
        key = (e["trial_id"], e["attempt"])
        base = {**context, **{k: e.get(k, "") for k in
            ("trial_id", "attempt", "block", "repeat", "word", "training_trial")}}
        trial = trials.setdefault(key, {**base, "has_pause": False, "outcome": "missing_end"})
        kind, stamp = e["type"], e["_host_ns"]
        if kind == "trial_prepare":
            trial["prepare_ns"] = stamp
        elif kind == "trial_start":
            trial["_duplicate"] = "start_ns" in trial
            trial.update(start_ns=stamp, start_event_id=e["event_id"])
        elif kind == "trial_end":
            trial["_duplicate"] = trial.get("_duplicate", False) or "end_ns" in trial
            trial.update(end_ns=stamp, end_event_id=e["event_id"], outcome=e.get("outcome", "unknown"))
        elif kind == "experiment_pause":
            trial["has_pause"] = True
            for hold_key, hold in holds.items():
                if hold_key[:2] == key and "start_ns" in hold and "end_ns" not in hold:
                    hold["has_pause"] = True
        elif kind in {"letter_hold_start", "letter_hold_end"} and type(e.get("letter_index")) is int:
            hold_key = (*key, e["letter_index"])
            hold = holds.setdefault(hold_key, {**base, "letter_index": e["letter_index"],
                "letter": e.get("letter", ""), "has_pause": False, "hold_completed": False})
            prefix = "start" if kind == "letter_hold_start" else "end"
            hold["_duplicate"] = hold.get("_duplicate", False) or prefix + "_ns" in hold
            hold[prefix + "_ns"] = stamp
            hold[prefix + "_event_id"] = e["event_id"]
            if prefix == "end":
                hold["hold_completed"] = e.get("completed") is True
    for row in trials.values():
        _boundaries(row)
    for key, row in holds.items():
        _boundaries(row)
        row["trial_outcome"] = trials[key[:2]]["outcome"]
    # block_start/end also carry a trial_id; omit phantom trials with no trial events.
    trial_rows = [r for r in trials.values() if any(k in r for k in ("prepare_ns", "start_ns", "end_ns"))]
    return trial_rows, list(holds.values())


def build_dataset(presenter_root, recording_roots=(), output=None):
    presenter_root = Path(presenter_root).resolve()
    output = Path(output).resolve() if output else presenter_root / "dataset"
    warnings, sessions, candidates = [], {}, defaultdict(dict)

    def add_recording(path):
        path = Path(path).resolve()
        if any(path in paths for paths in candidates.values()):
            return
        meta = _json(path / "metadata.json", warnings)
        if meta.get("session_id") and "recording_t0_ns" in meta:
            candidates[meta["session_id"]][path] = meta

    for root in recording_roots:
        for path in _session_dirs(Path(root).resolve()):
            add_recording(path)
    for path in _session_dirs(presenter_root):
        meta = _json(path / "metadata.json", warnings)
        sid = meta.get("session_id")
        if not sid or not meta.get("participant_id"):
            continue
        validate_participant(meta["participant_id"])
        if sid in sessions:
            raise ValueError(f"Duplicate presenter session_id: {sid}")
        local, acks = {}, {}
        for row in _csv(path / "presenter_events.csv", warnings):
            try:
                event = json.loads(row["details_json"])
                if event["session_id"] != sid or event["participant_id"] != meta["participant_id"]:
                    raise ValueError("Event/session identity mismatch")
                local[event["event_id"]] = event
            except (KeyError, TypeError, ValueError) as error:
                warnings.append(f"{path}/presenter_events.csv: {error}")
        for row in _csv(path / "recorder_acks.csv", warnings):
            if row.get("event_id") and row.get("recording_session_id"):
                acks[row["event_id"]] = row
                rid = row["recording_session_id"]
                if not candidates[rid] and row.get("recording_path") and Path(row["recording_path"]).is_dir():
                    add_recording(row["recording_path"])
        for rid, link in meta.get("recording_links", {}).items():
            if not candidates[rid] and link.get("recording_path") and Path(link["recording_path"]).is_dir():
                add_recording(link["recording_path"])
        sessions[sid] = {"path": path, "meta": meta, "local": local, "acks": acks}

    recorded = defaultdict(list)
    for rid, paths in candidates.items():
        if len(paths) != 1:
            if len(paths) > 1:
                warnings.append(f"Ambiguous recording_session_id {rid}: {list(map(str, paths))}")
            continue
        path = next(iter(paths))
        seen = set()
        for row in _csv(path / "events.csv", warnings):
            try:
                event = json.loads(row["details_json"])
                if not isinstance(event, dict) or not event.get("session_id") or not event.get("event_id"):
                    continue
                sid, event_id = event["session_id"], event["event_id"]
                if sid not in sessions:
                    continue
                if (event.get("recording_session_id") != rid or event.get("participant_id") != sessions[sid]["meta"]["participant_id"]):
                    raise ValueError("Recorder event/session identity mismatch")
                if event_id in seen:
                    raise ValueError("Duplicate recorded event_id: " + event_id)
                stamp = int(row["host_timestamp_ns"])
                seen.add(event_id)
                recorded[(sid, rid)].append({**event, "_host_ns": stamp})
            except (KeyError, TypeError, ValueError) as error:
                warnings.append(f"{path}/events.csv: {error}")

    index, all_trials, all_holds = [], [], []
    used_recordings = set()
    participant_sessions = defaultdict(list)
    for sid, session in sorted(sessions.items(), key=lambda item: item[1]["meta"].get("started_at", "")):
        meta, path, local, acks = (session[k] for k in ("meta", "path", "local", "acks"))
        participant_sessions[meta["participant_id"]].append(meta)
        links = {e.get("recording_session_id") for e in local.values()} | {a.get("recording_session_id") for a in acks.values()}
        links |= set(meta.get("recording_links", {})) | {r for s, r in recorded if s == sid}
        links.discard(None)
        links.discard("")
        for rid in sorted(links) or [""]:
            paths = candidates.get(rid, {})
            rec_path = next(iter(paths)) if len(paths) == 1 else None
            rec_meta = paths[rec_path] if rec_path else {}
            events = recorded[(sid, rid)]
            ack_ids = {key for key, value in acks.items() if value.get("recording_session_id") == rid}
            local_ids = {key for key, value in local.items() if (value.get("recording_session_id") or "") == rid}
            recorded_ids = {e["event_id"] for e in events}
            for event in events:
                ack = acks.get(event["event_id"])
                if ack and (ack["recording_session_id"] != rid or ack.get("host_timestamp_ns") != str(event["_host_ns"])):
                    warnings.append(f"ACK mismatch for {sid}/{event['event_id']}")
            if not rid:
                link_status = "offline" if meta.get("offline") else "unmatched"
            elif len(paths) > 1:
                link_status = "ambiguous_recording"
            elif not rec_path:
                link_status = "missing_recording"
            elif events:
                link_status = "events_verified"
            else:
                link_status = "ack_only" if ack_ids else "unconfirmed"
            if rid:
                used_recordings.add(rid)
            profile = meta.get("participant_info", {})
            files = {name: _relative(rec_path / filename, output) if rec_path else "" for name, filename in (
                ("recorder_metadata_file", "metadata.json"), ("events_file", "events.csv"),
                ("own_frames_file", "own_frames.jsonl"), ("noitom_frames_file", "noitom_frames.jsonl"))}
            context = {"participant_id": meta["participant_id"], "session_id": sid, "recording_session_id": rid,
                       "recording_status": rec_meta.get("status", "unknown"), **files}
            trials, holds = _intervals(events, context)
            all_trials.extend(trials)
            all_holds.extend(holds)
            index.append({**context, "handedness": profile.get("handedness", ""), "participant_notes": profile.get("notes", ""),
                "participant_profile_source": "session_snapshot" if "participant_info" in meta else "not_collected",
                "started_at": meta.get("started_at", ""), "ended_at": meta.get("ended_at", ""),
                "training_trial": meta.get("training_trial", ""), "presenter_status": meta.get("status", "unknown"),
                "presenter_outcome": meta.get("outcome", ""),
                "offline": meta.get("offline", False), "link_status": link_status,
                "recording_started_at": rec_meta.get("started_at", ""), "presenter_dir": _relative(path, output),
                "recording_dir": _relative(rec_path, output), "presenter_metadata_file": _relative(path / "metadata.json", output),
                "protocol_file": _relative(path / "protocol.json", output),
                **{key: meta.get(key, "") for key in ("application_version", "protocol_sha256", "random_seed", "mirror_mode")},
                "block_order": ">".join(meta.get("block_order", [])), "repetitions": meta.get("number_of_repeats", ""),
                **{key + "_s": value for key, value in meta.get("timing", {}).items()},
                "planned_trials": len(meta.get("word_order", [])), "recorded_attempts": len(trials),
                "completed_attempts": sum(t["outcome"] == "completed" and t["boundaries_complete"] for t in trials),
                "recorded_holds": len(holds), "confirmed_events": len(events), "ack_events": len(ack_ids),
                "local_events": len(local_ids), "missing_recorded_events": len((local_ids | ack_ids) - recorded_ids) if not meta.get("offline") else 0,
                "delivery_issue_count": len(meta.get("delivery_issues", [])),
                "own_frame_count": rec_meta.get("counts", {}).get("own_frames", ""),
                "noitom_frame_count": rec_meta.get("counts", {}).get("noitom_frames", ""),
                "own_frames_present": bool(rec_path and (rec_path / "own_frames.jsonl").is_file()),
                "noitom_frames_present": bool(rec_path and (rec_path / "noitom_frames.jsonl").is_file()),
                "events_present": bool(rec_path and (rec_path / "events.csv").is_file()),
                "synthetic_noitom": rec_meta.get("synthetic_noitom", rec_meta.get("noitom", {}).get("configuration", {}).get("fake", "")),
                "recorder_incomplete_reasons": ";".join(rec_meta.get("incomplete_reasons", []))})
    participants = []
    card_ids = {p.stem for p in (presenter_root / "participants").glob("*.json")}
    for pid in sorted(set(participant_sessions) | card_ids):
        validate_participant(pid)
        metas = participant_sessions.get(pid, [])
        profile_path = presenter_root / "participants" / (pid + ".json")
        profile = _json(profile_path, warnings) if profile_path.is_file() else (metas[-1].get("participant_info", {}) if metas else {})
        participants.append({"participant_id": pid, **{k: profile.get(k, "") for k in ("handedness", "notes")},
            "profile_source": "participant_card" if profile_path.is_file() else ("latest_session_snapshot" if profile else "not_collected"),
            "session_count": len(metas), "first_session_at": metas[0].get("started_at", "") if metas else "",
            "last_session_at": metas[-1].get("started_at", "") if metas else ""})
    unlinked = [{"recording_session_id": rid, "started_at": meta.get("started_at", ""), "status": meta.get("status", ""),
                 "recording_dir": _relative(path, output), "recorder_metadata_file": _relative(path / "metadata.json", output)}
                for rid, paths in candidates.items() if rid not in used_recordings for path, meta in paths.items()]
    output.mkdir(parents=True, exist_ok=True)
    tables = {"dataset_index.csv": (INDEX_FIELDS, index), "participants.csv": (PARTICIPANT_FIELDS, participants),
              "trial_attempts.csv": (TRIAL_FIELDS, all_trials), "letter_holds.csv": (HOLD_FIELDS, all_holds),
              "unlinked_recordings.csv": (UNLINKED_FIELDS, unlinked)}
    for name, (fields, rows) in tables.items():
        _atomic_csv(output / name, fields, rows)
    report = {"schema_version": 1, "generated_at": datetime.now().astimezone().isoformat(),
              "path_base": "directory containing these CSV files", "sessions": len(sessions),
              "counts": {name: len(rows) for name, (_, rows) in tables.items()}, "warnings": warnings}
    temp = output / ("report." + uuid.uuid4().hex + ".tmp")
    try:
        temp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temp.replace(output / "report.json")
    finally:
        temp.unlink(missing_ok=True)
    return {**report, "output_dir": str(output), "index": index}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build participant/session/motion-recording CSV dataset index")
    parser.add_argument("--presenter-root", type=Path, default=ROOT / "records" / "semaphore")
    parser.add_argument("--recordings-root", type=Path, action="append", help="Repeat for multiple recording locations")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = build_dataset(args.presenter_root, args.recordings_root or [ROOT / "recordings"], args.output)
    print(json.dumps({k: v for k, v in result.items() if k != "index"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
