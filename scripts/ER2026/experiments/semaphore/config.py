"""Validated configuration and deterministic trial order, without Qt."""
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import random
import re

from .semaphore_model import NEUTRAL, SEMAPHORE_POSES, SemaphorePose, letter_group
from scripts.ER2026.paths import EXPERIMENT_ROOT

ROOT = EXPERIMENT_ROOT
DEFAULT_CONFIG = Path(__file__).with_name("configs") / "experiment.json"
BLOCK_LABELS = {"lower": "Нижняя полуплоскость", "upper": "Верхняя полуплоскость"}


@dataclass(frozen=True)
class Timing:
    prepare: float = 2.
    word_preview: float = 1.
    transition: float = .4
    hold: float = 1.6
    post_trial: float = 1.5
    inter_trial: float = 1.5

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Некорректная длительность {name}")
        if self.transition <= 0 or self.hold <= 0:
            raise ValueError("Длительности transition и hold должны быть положительными")


@dataclass(frozen=True)
class ExperimentConfig:
    words: dict[str, tuple[str, ...]]
    repetitions: int = 3
    timing: Timing = field(default_factory=Timing)
    randomize_words: bool = True
    mirror_mode: bool = True
    neutral_pose: SemaphorePose = NEUTRAL
    blocks: tuple[str, ...] = ("lower", "upper")
    training_words: tuple[str, ...] = ("ВОДА", "БАТОН", "ЖУК", "ХМЕЛЬ")
    sessions_dir: str = "records"
    series_seconds: float = 0.  # zero preserves legacy/manual REC workflows
    series_prepare: float = 3.
    recording_seconds: float = 300.
    export_fps: float = 50.
    word_set_ids: dict[str, str] = field(default_factory=lambda: {'lower': 'SET01', 'upper': 'SET02'})

    def __post_init__(self):
        if (isinstance(self.export_fps, bool) or not isinstance(self.export_fps, (int, float))
                or not math.isfinite(self.export_fps) or not 1 <= self.export_fps <= 100):
            raise ValueError('Частота BVH должна быть от 1 до 100 Гц')
        if (isinstance(self.recording_seconds, bool) or not isinstance(self.recording_seconds, (int, float))
                or not math.isfinite(self.recording_seconds) or not 1 <= self.recording_seconds <= 3600):
            raise ValueError('Продолжительность записи должна быть от 1 до 3600 секунд')
        if set(self.word_set_ids) != set(self.blocks) or len(set(self.word_set_ids.values())) != len(self.blocks):
            raise ValueError('Для каждого набора нужен уникальный word_set_id')
        for word_set_id in self.word_set_ids.values():
            validate_participant(word_set_id)
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v >= 0
                   for v in (self.series_seconds, self.series_prepare)):
            raise ValueError('Длительность серии и подготовки должна быть конечной и неотрицательной')
        if type(self.repetitions) is not int or not 1 <= self.repetitions <= 100:
            raise ValueError("Число повторов должно быть от 1 до 100")
        if len(self.blocks) != 2 or set(self.blocks) != set(BLOCK_LABELS):
            raise ValueError("Укажите оба блока по одному разу: lower и upper")
        if type(self.mirror_mode) is not bool or type(self.randomize_words) is not bool:
            raise ValueError("mirror_mode и randomize_words должны быть true/false")
        for block in self.blocks:
            words = self.words.get(block, ())
            if not words or len(set(words)) != len(words):
                raise ValueError(f"Пустой список или повторяющиеся слова в блоке {block}")
            for word in words:
                validate_word(word, block)
        if not self.training_words:
            raise ValueError("Не задан список тренировочных слов")
        for word in self.training_words:
            validate_word(word)
        if not isinstance(self.sessions_dir, str) or not self.sessions_dir.strip():
            raise ValueError("Не задан каталог сессий")


def validate_word(word, block=None):
    if not isinstance(word, str) or not word or word != word.upper() or len(word) > 32:
        raise ValueError(f"Некорректное слово: {word!r}")
    unknown = set(word) - SEMAPHORE_POSES.keys()
    if unknown:
        raise ValueError(f"Неизвестные буквы в {word}: {', '.join(sorted(unknown))}")
    groups = {letter_group(letter) for letter in word}
    if len(groups) != 1 or (block is not None and groups != {block}):
        raise ValueError(f"Буквы слова {word} не принадлежат блоку {block or 'целиком одному набору'}")


def validate_participant(value):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,48}", value):
        raise ValueError("Введите код участника, например P001 (латиница, цифры, _ или -)")
    return value


def load_config(path=DEFAULT_CONFIG):
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    words = {}
    for block, filename in data.pop("word_files").items():
        item = json.loads((path.parent / filename).read_text(encoding="utf-8-sig"))
        if item["block"] != block:
            raise ValueError(f"Неверный block в {filename}")
        words[block] = tuple(item["words"])
        if "allowed_letters" in item and any(set(w) - set(item["allowed_letters"]) for w in words[block]):
            raise ValueError(f"Слова {filename} содержат запрещённые буквы")
    return ExperimentConfig(words=words, **{**data, "timing": Timing(**data.get("timing", {})),
        "neutral_pose": SemaphorePose(**data.get("neutral_pose", asdict(NEUTRAL))),
        "blocks": tuple(data.get("blocks", ("lower", "upper"))),
        "training_words": tuple(data.get("training_words", ("ВОДА", "БАТОН", "ЖУК", "ХМЕЛЬ")))})


@dataclass(frozen=True)
class Trial:
    trial_id: int
    block: str
    repeat: int
    word: str
    word_index: int
    training_trial: bool = False


def generate_trials(config, seed, *, training=False):
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Seed должен быть целым числом от 0 до 4294967295")
    rng = random.Random(seed)
    trials = []
    previous = None
    for block in config.blocks:
        for repeat in range(1, (1 if training else config.repetitions) + 1):
            words = ([w for w in config.training_words if letter_group(w[0]) == block]
                     if training else list(config.words[block]))
            if config.randomize_words:
                rng.shuffle(words)
            if len(words) > 1 and words[0] == previous:
                words[0], words[1] = words[1], words[0]
            for index, word in enumerate(words, 1):
                trials.append(Trial(len(trials) + 1, block, repeat, word, index, training))
            if words:
                previous = words[-1]
    return tuple(trials)


def trial_duration(trial, timing):
    return timing.prepare + timing.word_preview + len(trial.word) * (timing.transition + timing.hold) + timing.transition + timing.post_trial


def plan_series(config, trials):
    """Greedy deterministic partition, with no split or renumbering of trials.

    Keep blocks separate; a final short series and a single overlong word are
    allowed and reported explicitly. Operator pauses are not predicted.
    """
    if config.series_seconds <= 0:
        return (tuple(trials),) if trials else ()
    groups, current, duration = [], [], config.series_prepare
    for trial in trials:
        extra = trial_duration(trial, config.timing) + (config.timing.inter_trial if current else 0.)
        if current and (duration + extra > config.series_seconds or current[-1].block != trial.block):
            groups.append(tuple(current))
            current, duration = [], config.series_prepare
            extra = trial_duration(trial, config.timing)
        current.append(trial)
        duration += extra
    if current:
        groups.append(tuple(current))
    return tuple(groups)


def series_duration(config, trials):
    return config.series_prepare + sum(trial_duration(t, config.timing) for t in trials) + max(0, len(trials)-1) * config.timing.inter_trial


def config_from_snapshot(data):
    return ExperimentConfig(**{**data, 'words': {k: tuple(v) for k, v in data['words'].items()},
        'timing': Timing(**data['timing']), 'neutral_pose': SemaphorePose(**data['neutral_pose']),
        'blocks': tuple(data['blocks']), 'training_words': tuple(data['training_words'])})
