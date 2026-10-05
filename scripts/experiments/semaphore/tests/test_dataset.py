import csv
from dataclasses import replace
import json
from pathlib import Path
import shutil
import uuid

from scripts.experiments.semaphore.config import load_config, generate_trials
from scripts.experiments.semaphore.dataset import build_dataset
from scripts.experiments.semaphore.participant_profile import load_profile, save_profile
from scripts.experiments.semaphore.session_log import SessionLog
from scripts.noitom_comparison.recording import ExperimentRecorder


T0 = 90_071_992_547_400_000  # deliberately beyond float/Excel exact-integer range


def create_log(root, *, offline=False, notes="проверка", hand="left"):
    config = replace(load_config(), sessions_dir=str(root))
    trials = generate_trials(config, 7, training=True)[:1]
    return SessionLog(config, trials, "P001", 7, offline=offline,
                      participant_info={"handedness": hand, "notes": notes})


def emit(log, recorder, name, step, **extra):
    active = recorder.active if recorder else None
    event = {**log.context, "experiment": "semaphore", "type": name,
             "event_id": uuid.uuid4().hex, "local_monotonic_ns": step,
             "recording_session_id": active.session_id if active else None,
             "trial_id": 1, "attempt": 1, "block": "lower", "repeat": 1,
             "word": "ВОДА", "letter_index": 0, "letter": "В", "training_trial": True, **extra}
    log.append(event)
    if active:
        ack = recorder.external_event(event, T0 + step)
        assert ack["accepted"]
        log.acknowledge({"event_id": event["event_id"], **ack})
    return event


def stop(recorder):
    session = recorder.stop(stop_ns=T0 + 10_000)
    assert session.done.wait(3) and not session.error
    return session


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def full_trial(log, recorder, offset=0):
    for i, (kind, values) in enumerate([
        ("experiment_start", {}), ("trial_prepare", {}), ("trial_start", {}),
        ("letter_hold_start", {}), ("experiment_pause", {}), ("experiment_resume", {}),
        ("letter_hold_end", {"completed": True}), ("trial_end", {"outcome": "completed"}),
        ("experiment_end", {"outcome": "completed"}),
    ]):
        emit(log, recorder, kind, offset + i + 1, **values)
    log.close()


def test_many_presenter_sessions_one_recording_and_participant_snapshot(tmp_path):
    recorder = ExperimentRecorder()
    session = recorder.start(tmp_path / "motion", {"synthetic_noitom": False}, t0_ns=T0)
    first = create_log(tmp_path / "presenter", notes="первая сессия")
    full_trial(first, recorder)
    second = create_log(tmp_path / "presenter", notes="вторая сессия")
    full_trial(second, recorder, 100)
    stop(recorder)
    result = build_dataset(tmp_path / "presenter")  # discovers custom root via saved ACK paths
    output = Path(result["output_dir"])
    index = rows(output / "dataset_index.csv")
    assert len(index) == 2 and {r["recording_session_id"] for r in index} == {session.session_id}
    assert {r["participant_notes"] for r in index} == {"первая сессия", "вторая сессия"}
    assert all(r["link_status"] == "events_verified" and r["missing_recorded_events"] == "0" for r in index)
    assert all(r["synthetic_noitom"] == "False" for r in index)
    for row in index:
        assert (output / row["recording_dir"]).resolve() == session.path.resolve()
        assert (output / row["own_frames_file"]).is_file()
    holds = rows(output / "letter_holds.csv")
    assert len(holds) == 2 and holds[0]["start_ns"] == str(T0 + 4)
    assert holds[0]["end_ns"] == str(T0 + 7) and holds[0]["duration_ns"] == "3"
    assert holds[0]["has_pause"] == "True" and holds[0]["boundaries_complete"] == "True"
    assert rows(output / "participants.csv")[0]["session_count"] == "2"
    assert load_profile(tmp_path / "presenter", "P001")["notes"] == "вторая сессия"
    assert first.metadata["participant_info"]["notes"] == "первая сессия"
    assert not result["warnings"]


def test_one_presenter_session_two_recordings_does_not_merge_boundaries(tmp_path):
    recorder = ExperimentRecorder()
    first = recorder.start(tmp_path / "motion", {}, t0_ns=T0)
    log = create_log(tmp_path / "presenter")
    emit(log, recorder, "trial_start", 1)
    emit(log, recorder, "letter_hold_start", 2)
    stop(recorder)
    second = recorder.start(tmp_path / "motion", {}, t0_ns=T0)
    emit(log, recorder, "letter_hold_end", 3, completed=False)
    emit(log, recorder, "trial_end", 4, outcome="repeated")
    emit(log, recorder, "trial_start", 5, attempt=2)
    emit(log, recorder, "letter_hold_start", 6, attempt=2)
    emit(log, recorder, "letter_hold_end", 7, attempt=2, completed=True)
    emit(log, recorder, "trial_end", 8, attempt=2, outcome="completed")
    emit(log, recorder, "experiment_end", 9, outcome="completed")
    log.close()
    stop(recorder)
    result = build_dataset(tmp_path / "presenter")
    output = Path(result["output_dir"])
    assert {r["recording_session_id"] for r in result["index"]} == {first.session_id, second.session_id}
    holds = rows(output / "letter_holds.csv")
    assert len(holds) == 3
    broken = [r for r in holds if r["attempt"] == "1"]
    assert all(r["boundaries_complete"] == "False" and r["duration_ns"] == "" for r in broken)
    complete = next(r for r in holds if r["attempt"] == "2")
    assert complete["boundaries_complete"] == "True" and complete["trial_outcome"] == "completed"


def test_backfill_legacy_sessions_joins_ids_not_folder_names_or_dates(tmp_path):
    recorder = ExperimentRecorder()
    rec = recorder.start(tmp_path / "motion", {}, t0_ns=T0)
    log = create_log(tmp_path / "presenter")
    full_trial(log, recorder)
    stop(recorder)
    # Legacy sessions lack the new profile/recording-path fields. Even missing
    # ACKs can be recovered from the recorder's authoritative event IDs.
    legacy = dict(log.metadata)
    legacy.pop("participant_info")
    legacy.pop("recording_links")
    (log.path / "metadata.json").write_text(json.dumps(legacy), encoding="utf-8")
    (log.path / "recorder_acks.csv").write_text("event_id,host_timestamp_ns,recording_session_id\n", encoding="utf-8")
    moved = rec.path.with_name("renamed_without_date")
    rec.path.rename(moved)
    unlinked = recorder.start(tmp_path / "motion", {}, t0_ns=T0)
    stop(recorder)
    result = build_dataset(tmp_path / "presenter", [tmp_path / "motion"])
    row = result["index"][0]
    assert row["recording_session_id"] == rec.session_id and row["link_status"] == "events_verified"
    assert row["participant_profile_source"] == "not_collected" and not row["handedness"]
    assert row["ack_events"] == 0 and row["confirmed_events"] == 9
    assert rows(Path(result["output_dir"]) / "unlinked_recordings.csv")[0]["recording_session_id"] == unlinked.session_id


def test_offline_and_missing_recordings_stay_visible(tmp_path):
    offline = create_log(tmp_path / "presenter", offline=True)
    full_trial(offline, None)
    missing = create_log(tmp_path / "presenter")
    event = emit(missing, None, "experiment_start", 1)
    missing.acknowledge({"event_id": event["event_id"], "host_timestamp_ns": T0,
                         "recording_session_id": "missing", "recording_path": str(tmp_path / "missing")})
    missing.close()
    result = build_dataset(tmp_path / "presenter")
    index = {r["session_id"]: r for r in result["index"]}
    assert index[offline.context["session_id"]]["link_status"] == "offline"
    assert index[offline.context["session_id"]]["local_events"] == 9
    assert index[offline.context["session_id"]]["missing_recorded_events"] == 0
    assert index[missing.context["session_id"]]["link_status"] == "missing_recording"
    assert not index[missing.context["session_id"]]["recording_dir"]


def test_duplicate_recording_ids_are_ambiguous_not_silently_selected(tmp_path):
    recorder = ExperimentRecorder()
    rec = recorder.start(tmp_path / "motion", {}, t0_ns=T0)
    log = create_log(tmp_path / "presenter")
    full_trial(log, recorder)
    stop(recorder)
    shutil.copytree(rec.path, tmp_path / "motion" / "copy")
    result = build_dataset(tmp_path / "presenter", [tmp_path / "motion"])
    assert result["index"][0]["link_status"] == "ambiguous_recording"
    assert result["counts"]["letter_holds.csv"] == 0
    assert result["warnings"]


def test_participant_card_is_visible_before_first_session(tmp_path):
    save_profile(tmp_path, "P010", {"handedness": "right", "notes": "ещё нет записи"})
    result = build_dataset(tmp_path)
    people = rows(Path(result["output_dir"]) / "participants.csv")
    assert people[0]["participant_id"] == "P010" and people[0]["session_count"] == "0"
    assert people[0]["handedness"] == "right" and people[0]["first_session_at"] == ""
