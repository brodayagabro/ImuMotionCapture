"""Monotonic state machine. No Qt, socket, hardware or disk dependencies."""
from enum import StrEnum
import time

from .semaphore_model import SEMAPHORE_POSES, interpolate_pose


class State(StrEnum):
    IDLE = "IDLE"
    CONNECTING = "CONNECTING"
    READY = "READY"
    PREPARE = "PREPARE"
    WORD_PREVIEW = "WORD_PREVIEW"
    TRANSITION = "TRANSITION"
    HOLD = "HOLD"
    RETURN_NEUTRAL = "RETURN_NEUTRAL"
    POST_TRIAL = "POST_TRIAL"
    INTER_TRIAL = "INTER_TRIAL"
    PAUSED = "PAUSED"
    FINISHED = "FINISHED"


TIMED_STATES = {State.PREPARE, State.WORD_PREVIEW, State.TRANSITION, State.HOLD,
                State.RETURN_NEUTRAL, State.POST_TRIAL, State.INTER_TRIAL}


class ExperimentController:
    def __init__(self, config, trials, context, emit, *, clock=time.monotonic_ns, offline=False):
        if not trials:
            raise ValueError("Нет испытаний")
        self.config, self.trials, self.context = config, trials, dict(context)
        self.emit_callback, self.clock, self.offline = emit, clock, offline
        self.state = State.IDLE
        self.recorder_ready = False
        self.recovery_required = False
        self.trial_index = 0
        self.letter_index = None
        self.attempt = 1
        self.phase_started_ns = self.clock()
        self.duration_ns = 0
        self.paused_at_ns = None
        self.paused_state = None
        self.source_pose = self.target_pose = config.neutral_pose
        self.trial_open = self.block_open = self.hold_open = False
        self.started = False

    @property
    def trial(self):
        return self.trials[min(self.trial_index, len(self.trials) - 1)]

    @property
    def active(self):
        return self.started and self.state != State.FINISHED

    @property
    def phase(self):
        return self.paused_state if self.state == State.PAUSED else self.state

    def _emit(self, event, **extra):
        trial = self.trial
        payload = {**self.context, "type": event, "experiment": "semaphore",
            "local_monotonic_ns": self.clock(), "phase": self.state.value,
            "training_trial": trial.training_trial}
        if self.started and event not in {"experiment_start", "experiment_end"}:
            payload.update(trial_id=trial.trial_id, block=trial.block, word=trial.word,
                repeat=trial.repeat, attempt=self.attempt, letter_index=self.letter_index,
                letter=trial.word[self.letter_index] if self.letter_index is not None else None)
        self.emit_callback({**payload, **extra})

    def set_recorder_ready(self, ready):
        self.recorder_ready = ready
        if not ready and self.active and not self.offline:
            self.recovery_required = self.trial_open
            self.pause(reason="recorder_connection_lost")

    def start(self):
        if self.started:
            raise ValueError("Сессия уже начата")
        if not self.recorder_ready and not self.offline:
            raise ValueError("Рекордер не готов: подключитесь и включите REC")
        self.started = True
        self.state = State.READY
        self._emit("experiment_start", total_trials=len(self.trials), offline=self.offline)

    def start_block(self):
        if self.state != State.READY or not self.started:
            raise ValueError("Сейчас нельзя начать блок")
        if not self.recorder_ready and not self.offline:
            raise ValueError("Рекордер не готов")
        self.block_open = True
        self._emit("block_start")
        self._begin_trial()

    def _enter(self, state, seconds):
        self.state, self.phase_started_ns = state, self.clock()
        self.duration_ns = round(seconds * 1e9)

    def _begin_trial(self):
        self.trial_open = True
        self.letter_index = None
        self.source_pose = self.target_pose = self.config.neutral_pose
        self._enter(State.PREPARE, self.config.timing.prepare)
        self._emit("trial_prepare")

    def _letter_transition(self, index):
        self.source_pose = self.target_pose
        self.letter_index = index
        self.target_pose = SEMAPHORE_POSES[self.trial.word[index]]
        self._enter(State.TRANSITION, self.config.timing.transition)
        self._emit("letter_transition_start")

    def tick(self):
        if self.state not in TIMED_STATES:
            return
        elapsed = self.clock() - self.phase_started_ns
        if elapsed < self.duration_ns:
            return
        # A late GUI wake-up must not fabricate presentations of unseen letters.
        # Advance at most one phase and start its full duration at actual now.
        if elapsed - self.duration_ns > 100_000_000:
            self._emit("scheduler_delay", lateness_ns=elapsed - self.duration_ns)
        timing = self.config.timing
        if self.state == State.PREPARE:
            self._enter(State.WORD_PREVIEW, timing.word_preview)
            self._emit("word_show")
        elif self.state == State.WORD_PREVIEW:
            self._emit("trial_start")
            self._letter_transition(0)
        elif self.state == State.TRANSITION:
            self._enter(State.HOLD, timing.hold)
            self.hold_open = True
            self._emit("letter_hold_start")
        elif self.state == State.HOLD:
            self._close_hold(completed=True)
            if self.letter_index + 1 < len(self.trial.word):
                self._letter_transition(self.letter_index + 1)
            else:
                self.source_pose = self.target_pose
                self.target_pose = self.config.neutral_pose
                self.letter_index = None
                self._enter(State.RETURN_NEUTRAL, timing.transition)
                self._emit("neutral_transition_start")
        elif self.state == State.RETURN_NEUTRAL:
            self._enter(State.POST_TRIAL, timing.post_trial)
            self._emit("neutral_hold_start")
        elif self.state == State.POST_TRIAL:
            self._end_trial("completed")
            self._advance_trial()
        elif self.state == State.INTER_TRIAL:
            self._begin_trial()

    def _close_hold(self, *, completed):
        if self.hold_open:
            self._emit("letter_hold_end", completed=completed)
            self.hold_open = False

    def _end_trial(self, outcome):
        if self.trial_open:
            self._close_hold(completed=False)
            self._emit("trial_end", outcome=outcome)
            self.trial_open = False

    def _advance_trial(self):
        block = self.trial.block
        next_index = self.trial_index + 1
        block_done = next_index == len(self.trials) or self.trials[next_index].block != block
        if block_done:
            self._emit("block_end", outcome="completed")
            self.block_open = False
        self.source_pose = self.target_pose = self.config.neutral_pose
        self.letter_index = None
        self.attempt = 1
        self.recovery_required = False
        if next_index == len(self.trials):
            self.state = State.FINISHED
            self._emit("experiment_end", outcome="completed")
        else:
            self.trial_index = next_index
            if block_done:
                self.state = State.READY  # operator confirms every block
            else:
                self._enter(State.INTER_TRIAL, self.config.timing.inter_trial)

    def pause(self, *, reason="operator"):
        if self.state not in TIMED_STATES:
            return
        self.paused_state, self.paused_at_ns = self.state, self.clock()
        self.state = State.PAUSED
        self._emit("experiment_pause", reason=reason, paused_phase=self.paused_state.value)

    def resume(self):
        if self.state != State.PAUSED:
            return
        if self.recovery_required:
            raise ValueError("После потери связи повторите текущую пробу")
        if not self.recorder_ready and not self.offline:
            raise ValueError("Рекордер не готов")
        self.phase_started_ns += self.clock() - self.paused_at_ns
        self.state = self.paused_state
        self._emit("experiment_resume")

    def repeat_trial(self):
        if not self.trial_open:
            raise ValueError("Нет активной пробы для повтора")
        if not self.recorder_ready and not self.offline:
            raise ValueError("Сначала восстановите связь с рекордером")
        self._end_trial("repeated")
        self.recovery_required = False
        self.attempt += 1
        self._emit("experiment_resume", reason="repeat_trial")
        self._begin_trial()

    def skip_trial(self):
        if not self.trial_open:
            return
        if not self.recorder_ready and not self.offline:
            raise ValueError("Сначала восстановите связь с рекордером")
        self._end_trial("skipped")
        self._advance_trial()

    def stop(self):
        if not self.active:
            return
        self._end_trial("stopped")
        if self.block_open:
            self._emit("block_end", outcome="stopped")
            self.block_open = False
        self.source_pose = self.target_pose = self.config.neutral_pose
        self.state = State.FINISHED
        self._emit("experiment_end", outcome="stopped")

    def fraction(self):
        now = self.paused_at_ns if self.state == State.PAUSED else self.clock()
        return min(1., max(0., (now - self.phase_started_ns) / max(1, self.duration_ns)))

    def pose(self):
        if self.phase in {State.TRANSITION, State.RETURN_NEUTRAL}:
            return interpolate_pose(self.source_pose, self.target_pose, self.fraction())
        return self.target_pose

    def letter_progress(self):
        timing = self.config.timing
        if self.phase == State.TRANSITION:
            return self.fraction() * timing.transition / (timing.transition + timing.hold)
        if self.phase == State.HOLD:
            return (timing.transition + self.fraction() * timing.hold) / (timing.transition + timing.hold)
        return 0.
