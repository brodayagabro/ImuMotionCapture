from dataclasses import replace
from collections import Counter
import pytest
from scripts.ER2026.experiments.semaphore.config import load_config, generate_trials, plan_series, series_duration, Timing
from scripts.ER2026.experiments.semaphore.series_plan import SeriesPlan


@pytest.mark.parametrize('seconds', [300., 600., 175.])
def test_full_plan_partition_has_no_loss_duplicates_or_renumbering(seconds):
    config = replace(load_config(), series_seconds=seconds)
    trials = generate_trials(config, 193)
    groups = plan_series(config, trials)
    assert tuple(t for g in groups for t in g) == trials
    assert all(series_duration(config, g) <= seconds for g in groups)
    assert groups == plan_series(config, generate_trials(config, 193))
    assert Counter(t.trial_id for g in groups for t in g) == Counter(range(1, 121))
    assert all(len({t.block for t in group}) == 1 for group in groups)


def test_hold_doubled_and_transition_independent():
    assert Timing().hold == load_config().timing.hold == 1.6
    assert load_config().timing.transition == .4
    config = load_config()
    trial, = generate_trials(replace(config, words={'lower': ('ВОДА',), 'upper': ('ЖУК',)}, repetitions=1), 1)[:1]
    old = replace(config, timing=replace(config.timing, hold=.8))
    assert series_duration(config, [trial]) - series_duration(old, [trial]) == pytest.approx(len(trial.word)*.8)


def test_stop_and_restart_keep_unfinished_trials_and_unique_attempts(tmp_path):
    plan = SeriesPlan.create(tmp_path, load_config(), 'P001', 8, {'handedness': 'left', 'notes': ''})
    series, trials = plan.begin()
    first, second = trials[:2]
    plan.mark_event({'type':'trial_prepare', 'trial_id':first.trial_id, 'attempt':1})
    plan.mark_event({'type':'trial_end', 'trial_id':first.trial_id, 'attempt':1, 'outcome':'completed'})
    plan.mark_event({'type':'trial_prepare', 'trial_id':second.trial_id, 'attempt':1})
    restored = SeriesPlan.load(plan.path)
    assert restored.data['active_series_id'] == series['series_id']
    assert restored.remaining[0] == second
    assert restored.data['next_attempts'][str(second.trial_id)] == 2
    restored.finish('interrupted')
    next_series, next_trials = restored.begin()
    assert next_series['series_id'] != series['series_id']
    assert next_series['series_index'] == 2 and next_trials[0] == second
    assert len(restored.remaining) == 119


def test_one_overlong_word_remains_whole():
    config = replace(load_config(), series_seconds=.1)
    trials = generate_trials(config, 1)
    assert all(len(g) == 1 for g in plan_series(config, trials))
