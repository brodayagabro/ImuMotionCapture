from collections import Counter
from dataclasses import replace
import json
import math

import pytest

from scripts.experiments.semaphore.config import load_config, generate_trials, Timing, validate_word
from scripts.experiments.semaphore.experiment_controller import ExperimentController, State
from scripts.experiments.semaphore.semaphore_model import (
    LOWER_LETTERS, UPPER_LETTERS, SEMAPHORE_POSES, SemaphorePose, arm_geometry, interpolate_pose,
)


class Clock:
    now = 1_000_000_000

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += round(seconds * 1e9)


def controller(offline=True):
    config = load_config()
    clock, events = Clock(), []
    c = ExperimentController(config, generate_trials(config, 42),
        {"participant_id": "P001", "session_id": "test"}, events.append, clock=clock, offline=offline)
    return c, clock, events


def test_default_protocol_and_randomization():
    config = load_config()
    assert [len(config.words[b]) for b in config.blocks] == [20, 20]
    assert set(SEMAPHORE_POSES) == LOWER_LETTERS | UPPER_LETTERS
    for seed in range(50):
        trials = generate_trials(config, seed)
        assert trials == generate_trials(config, seed)
        assert len(trials) == 120
        assert Counter(t.block for t in trials) == {"lower": 60, "upper": 60}
        assert all(a.word != b.word for a, b in zip(trials, trials[1:]))
        for block in config.blocks:
            for repeat in (1, 2, 3):
                assert sorted(t.word for t in trials if t.block == block and t.repeat == repeat) == sorted(config.words[block])
    assert generate_trials(config, 1) != generate_trials(config, 2)
    assert generate_trials(replace(config, blocks=("upper", "lower")), 1)[0].block == "upper"


@pytest.mark.parametrize("word", ["TEST", "СОН", "", "вода", "ВЖ"])
def test_unknown_and_mixed_alphabet_rejected(word):
    with pytest.raises(ValueError):
        validate_word(word)


def test_timing_validation():
    for value in (-1, float("nan"), float("inf"), True, "1"):
        with pytest.raises(ValueError):
            Timing(hold=value)
    with pytest.raises(ValueError):
        Timing(transition=0)


def test_configuration_files_are_checked_before_start(tmp_path):
    from scripts.experiments.semaphore.config import DEFAULT_CONFIG
    data = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    for block, filename in data["word_files"].items():
        content = json.loads((DEFAULT_CONFIG.parent / filename).read_text(encoding="utf-8"))
        if block == "lower":
            content["words"][0] = "СОН"
            content.pop("allowed_letters")
        (tmp_path / filename).write_text(json.dumps(content), encoding="utf-8")
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="Неизвестные буквы"):
        load_config(path)


def test_training_has_separate_counter():
    config = load_config()
    trials = generate_trials(config, 7, training=True)
    assert {t.word for t in trials} == {"ВОДА", "БАТОН", "ЖУК", "ХМЕЛЬ"}
    assert [t.trial_id for t in trials] == [1, 2, 3, 4]
    events = []
    c = ExperimentController(config, trials, {}, events.append, offline=True)
    c.start()
    c.start_block()
    c.stop()
    assert all(e["training_trial"] for e in events)


def test_angles_interpolation_and_mirror():
    assert interpolate_pose(SemaphorePose(350, 10), SemaphorePose(10, 350), .5) == SemaphorePose(0, 0)
    assert interpolate_pose(SemaphorePose(0, 0), SemaphorePose(90, 90), .25).left_angle_deg == 14.0625
    for letter, pose in SEMAPHORE_POSES.items():
        direct, mirrored = arm_geometry(pose, False), arm_geometry(pose, True)
        assert mirrored["left"][0][0] > 0
        assert mirrored["right"][0][0] < 0
        for arm in direct:
            for a, b in zip(direct[arm], mirrored[arm]):
                assert b == (-a[0], a[1])
                assert abs(b[0]) <= 1.03 and -.98 <= b[1] <= 1.05
        if letter in LOWER_LETTERS:
            assert all(math.sin(math.radians(a)) <= 1e-9 for a in (pose.left_angle_deg, pose.right_angle_deg))


@pytest.mark.parametrize("first,second", [
    ("Б", "Д"), ("В", "Г"), ("Ж", "З"), ("К", "Х"), ("Л", "М"),
    ("Н", "О"), ("П", "Р"), ("Ф", "Ы"), ("Ц", "Ч"), ("Ш", "Щ"), ("Ю", "Я"),
])
def test_chart_pairs_are_anatomically_symmetric(first, second):
    # Independent constraint from the semaphore chart, including cross-body
    # poses. Reflect the complete body: swap arms AND reflect their directions.
    a, b = SEMAPHORE_POSES[first], SEMAPHORE_POSES[second]
    assert b.left_angle_deg == (180 - a.right_angle_deg) % 360
    assert b.right_angle_deg == (180 - a.left_angle_deg) % 360


@pytest.mark.parametrize("letter,left_vertical", [("Х", -1), ("Ю", 0)])
@pytest.mark.parametrize("mirror_mode", [False, True])
def test_cross_body_right_arm_is_above_left_in_both_views(letter, left_vertical, mirror_mode):
    arms = arm_geometry(SEMAPHORE_POSES[letter], mirror_mode)
    left_start, left_end = arms["left"]
    right_start, right_end = arms["right"]
    assert right_end[1] > right_start[1]  # anatomical right arm is raised
    if left_vertical < 0:
        assert left_end[1] < left_start[1]
    else:
        assert left_end[1] == pytest.approx(left_start[1])
    # Both arms point to the participant's right, even after display reflection.
    direction = -1 if mirror_mode else 1
    assert direction * (left_end[0] - left_start[0]) > 0
    assert direction * (right_end[0] - right_start[0]) > 0


def test_symmetric_poses_and_e_alias_from_chart():
    for letter in ("А", "Т", "У", "Ь"):
        pose = SEMAPHORE_POSES[letter]
        assert pose.left_angle_deg == (180 - pose.right_angle_deg) % 360
    assert SEMAPHORE_POSES["Е"] == SEMAPHORE_POSES["Э"]
    arms = arm_geometry(SEMAPHORE_POSES["Е"], False)
    assert arms["right"][1][1] > arms["right"][0][1]
    assert arms["left"][1][1] < arms["left"][0][1]


def test_start_requires_ready_and_every_block_requires_operator():
    c, clock, events = controller(offline=False)
    with pytest.raises(ValueError):
        c.start()
    c.set_recorder_ready(True)
    c.start()
    assert c.state == State.READY
    clock.advance(1000)
    c.tick()
    assert c.state == State.READY
    blocks = 0
    while c.active:
        if c.state == State.READY:
            blocks += 1
            c.start_block()
        else:
            clock.now = c.phase_started_ns + c.duration_ns
            c.tick()
    assert blocks == 2
    assert sum(e["type"] == "trial_end" for e in events) == 120
    assert sum(e["type"] == "letter_hold_start" for e in events) == sum(len(t.word) for t in c.trials)
    assert sum(e["type"] == "letter_hold_start" for e in events) == sum(e["type"] == "letter_hold_end" for e in events)
    hold = next(e for e in events if e["type"] == "letter_hold_start")
    assert hold["letter"] == hold["word"][hold["letter_index"]]
    assert hold["participant_id"] == "P001" and hold["experiment"] == "semaphore"
    assert events[-1]["type"] == "experiment_end" and events[-1]["outcome"] == "completed"


def test_scheduler_pause_and_late_wake_do_not_skip_stimuli():
    c, clock, events = controller()
    c.start()
    c.start_block()
    clock.advance(1)
    c.pause()
    pose, fraction = c.pose(), c.fraction()
    clock.advance(100)
    c.tick()
    assert c.state == State.PAUSED and c.pose() == pose and c.fraction() == fraction
    c.resume()
    clock.advance(.999)
    c.tick()
    assert c.state == State.PREPARE
    clock.advance(.001)
    c.tick()
    assert c.state == State.WORD_PREVIEW
    clock.advance(100)
    c.tick()
    assert c.state == State.TRANSITION and c.fraction() == 0
    clock.advance(.2)
    c.pause()
    pose = c.pose()
    clock.advance(100)
    assert c.pose() == pose
    c.resume()
    clock.advance(.2)
    c.tick()
    assert c.state == State.HOLD
    assert any(e["type"] == "scheduler_delay" for e in events)


def test_connection_loss_requires_repeat_with_new_attempt():
    c, clock, events = controller(offline=False)
    c.set_recorder_ready(True)
    c.start()
    c.start_block()
    c.set_recorder_ready(False)
    assert c.state == State.PAUSED and c.recovery_required
    c.set_recorder_ready(True)
    with pytest.raises(ValueError):
        c.resume()
    c.repeat_trial()
    assert c.state == State.PREPARE and c.attempt == 2 and c.trial_index == 0
    assert next(e for e in events if e["type"] == "trial_end")["outcome"] == "repeated"
    c.skip_trial()
    assert c.trial_index == 1 and c.attempt == 1
    c.stop()
    assert c.state == State.FINISHED
