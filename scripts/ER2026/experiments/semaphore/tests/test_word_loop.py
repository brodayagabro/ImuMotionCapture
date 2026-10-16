from dataclasses import replace
import pytest
from scripts.ER2026.experiments.semaphore.config import load_config, Timing
from scripts.ER2026.experiments.semaphore.word_loop import WordLoop
from scripts.ER2026.experiments.semaphore.experiment_controller import State
from scripts.ER2026.experiments.semaphore.frame_calibration import POSE_CAPTURE_S, POSE_PREPARATION_S
from scripts.ER2026.experiments.semaphore.calibration_protocol import POSE_SEQUENCE


def test_config_defaults_and_unique_existing_set_ids():
    config=load_config()
    assert config.recording_seconds==300 and config.timing.hold==1.6
    assert config.word_set_ids=={'lower':'SET01','upper':'SET02'}
    assert POSE_CAPTURE_S==1.5 and POSE_PREPARATION_S==3
    assert len(POSE_SEQUENCE)==5
    with pytest.raises(ValueError, match='уникальный'):
        replace(config, word_set_ids={'lower':'SET01','upper':'SET01'})
    with pytest.raises(ValueError, match='Неизвестный'):
        WordLoop(config, 'bad')


def test_word_set_loops_only_after_last_letter_hold_and_word_complete():
    clock=[0]
    config=replace(load_config(),words={'lower':('А','Т'),'upper':('У',)},randomize_words=False,
                   timing=Timing(.01,.01,.01,.05,.01,.01))
    loop=WordLoop(config,'SET01',clock=lambda:clock[0])
    words=[]
    holds=[]
    for i in range(100):
        c=loop.controller
        if c.state==State.WORD_PREVIEW: words.append(c.trial.word)
        if c.state==State.HOLD:
            holds.append((c.trial.word,loop.cycles))
            clock[0]+=49_000_000
            loop.tick()
            assert c.state==State.HOLD
            clock[0]+=1_000_000
        else: clock[0]+=10_000_000
        loop.tick()
        if loop.cycles==3: break
    assert words==['А','Т','А','Т']
    assert holds==[('А',1),('Т',1),('А',2),('Т',2)]
