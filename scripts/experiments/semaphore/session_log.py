"""Local protocol snapshot and diagnostic event log, independent of recorder."""
import csv
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import uuid

from . import __version__
from .config import ROOT, validate_participant
from .semaphore_model import POSE_SOURCE, SEMAPHORE_POSES
from .participant_profile import validate_profile, save_profile


class SessionLog:
    def __init__(self, config, trials, participant, seed, *, offline=False, participant_info=None):
        validate_participant(participant)
        participant_info = validate_profile(participant_info)
        now = datetime.now().astimezone()
        session_id = f"{participant}_{now:%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
        directory = Path(config.sessions_dir)
        self.path = (directory if directory.is_absolute() else ROOT / directory) / session_id
        self.path.mkdir(parents=True, exist_ok=False)
        save_profile(self.path.parent, participant, participant_info)
        protocol = {"config": asdict(config), "poses": {k: asdict(v) for k, v in SEMAPHORE_POSES.items()},
            "pose_source": POSE_SOURCE, "angle_convention": "0=participant right, 90=up, 180=left, 270=down",
            "mirror_true": "participant left shoulder is on the right side of the image"}
        encoded = json.dumps(protocol, ensure_ascii=False, indent=2) + "\n"
        (self.path / "protocol.json").write_bytes(encoded.encode("utf-8"))
        self.context = {"participant_id": participant, "session_id": session_id, "random_seed": seed,
            "protocol_sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest()}
        self.metadata = {**self.context, "application_version": __version__, "started_at": now.isoformat(),
            "number_of_repeats": 1 if trials[0].training_trial else config.repetitions,
            "training_trial": trials[0].training_trial, "block_order": config.blocks,
            "word_order": [asdict(t) for t in trials], "timing": asdict(config.timing),
            "mirror_mode": config.mirror_mode, "offline": offline, "status": "running",
            "participant_info": participant_info, "recording_links": {}, "delivery_issues": [], "ack_count": 0}
        self._save_metadata()
        with (self.path / "trial_order.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(asdict(trials[0])))
            writer.writeheader()
            writer.writerows(asdict(t) for t in trials)
        self.handle = (self.path / "presenter_events.csv").open("w", encoding="utf-8", newline="")
        self.writer = csv.writer(self.handle)
        self.writer.writerow(("local_monotonic_ns", "event", "trial_id", "word", "letter", "details_json"))
        self.ack_handle = (self.path / "recorder_acks.csv").open("w", encoding="utf-8", newline="")
        self.ack_writer = csv.writer(self.ack_handle)
        self.ack_writer.writerow(("event_id", "host_timestamp_ns", "recording_session_id", "recording_path"))
        self.closed = False

    def _save_metadata(self):
        temp = self.path / "metadata.json.tmp"
        temp.write_text(json.dumps(self.metadata, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        temp.replace(self.path / "metadata.json")

    def append(self, event):
        if self.closed:
            return
        self.writer.writerow((event["local_monotonic_ns"], event["type"], event.get("trial_id", ""),
            event.get("word", ""), event.get("letter") or "", json.dumps(event, ensure_ascii=False)))
        self.handle.flush()
        if event["type"] == "experiment_end":
            self.metadata["ended_at"] = datetime.now().astimezone().isoformat()
            self.metadata["outcome"] = event["outcome"]
            self.metadata["status"] = "incomplete" if self.metadata["delivery_issues"] else event["outcome"]
            self._save_metadata()

    def acknowledge(self, ack):
        self.ack_writer.writerow((ack["event_id"], ack["host_timestamp_ns"], ack["recording_session_id"], ack.get("recording_path", "")))
        self.ack_handle.flush()
        self.metadata["ack_count"] += 1
        rid = ack["recording_session_id"]
        is_new = rid not in self.metadata["recording_links"]
        link = self.metadata["recording_links"].setdefault(rid, {"first_host_timestamp_ns": ack["host_timestamp_ns"], "ack_count": 0})
        link.update(last_host_timestamp_ns=ack["host_timestamp_ns"])
        link["ack_count"] += 1
        if ack.get("recording_path"):
            link["recording_path"] = ack["recording_path"]
        if is_new:
            self._save_metadata()

    def issue(self, message):
        self.metadata["delivery_issues"].append(message)
        self.metadata["status"] = "incomplete"
        self._save_metadata()

    def close(self):
        if not self.closed:
            self.handle.close()
            self.ack_handle.close()
            if self.metadata["status"] == "running":
                self.metadata["status"] = "interrupted"
            self._save_metadata()
            self.closed = True
