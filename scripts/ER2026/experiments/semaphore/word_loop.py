"""Loop one existing word set without plans, trial files, or stimulus logging."""
import random
import time
from .config import Trial
from .experiment_controller import ExperimentController, State


class WordLoop:
    def __init__(self, config, word_set_id, *, seed=None, clock=time.perf_counter_ns):
        self.config, self.clock = config, clock
        block = next((k for k, v in config.word_set_ids.items() if v == word_set_id), None)
        if block is None:
            raise ValueError('Неизвестный набор слов: ' + word_set_id)
        words = list(config.words[block])
        if config.randomize_words:
            random.Random(seed).shuffle(words)
        self.trials = tuple(Trial(i + 1, block, 1, w, i + 1) for i, w in enumerate(words))
        self.cycles = 0
        self._begin()

    def _begin(self):
        self.controller = ExperimentController(self.config, self.trials, {}, lambda event: None,
                                               clock=self.clock, offline=True)
        self.controller.start()
        self.controller.start_block()
        self.cycles += 1

    def tick(self):
        self.controller.tick()
        if self.controller.state == State.FINISHED:
            self._begin()

    def stop(self):
        self.controller.stop()
