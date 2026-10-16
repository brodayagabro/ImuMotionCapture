"""Optional participant fields; session metadata retains a historical snapshot."""
import json
from pathlib import Path
import uuid

from .config import validate_participant


def validate_profile(profile=None):
    profile = dict(profile or {})
    if set(profile) - {"handedness", "notes"}:
        raise ValueError("Неизвестные поля участника")
    hand, notes = profile.get("handedness", ""), profile.get("notes", "")
    if hand not in {"", "right", "left", "ambidextrous"}:
        raise ValueError("Неизвестное значение ведущей руки")
    if not isinstance(notes, str) or len(notes) > 2000:
        raise ValueError("Примечание должно быть текстом до 2000 символов")
    return {"handedness": hand, "notes": notes.strip()}


def load_profile(root, participant):
    validate_participant(participant)
    path = Path(root) / "participants" / (participant + ".json")
    return validate_profile(json.loads(path.read_text(encoding="utf-8"))) if path.exists() else validate_profile()


def save_profile(root, participant, profile):
    validate_participant(participant)
    profile = validate_profile(profile)
    path = Path(root) / "participants" / (participant + ".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix("." + uuid.uuid4().hex + ".tmp")
    try:
        temp.write_text(json.dumps(profile, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)
