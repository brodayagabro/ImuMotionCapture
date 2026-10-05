"""python -m scripts.experiments.semaphore.main --participant P001"""
import argparse
from dataclasses import asdict, replace
import math
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.experiments.semaphore.config import DEFAULT_CONFIG, Timing, load_config, validate_participant


def main(argv=None):
    parser = argparse.ArgumentParser(description="Semaphore Experiment Presenter")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--participant", default="P001")
    parser.add_argument("--recorder-host", default="127.0.0.1")
    parser.add_argument("--recorder-port", type=int, default=5055)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-speed", type=float, default=1., help="Speed multiplier, requires --debug")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        validate_participant(args.participant)
        if not math.isfinite(args.debug_speed) or not 0 < args.debug_speed <= 100:
            raise ValueError("--debug-speed должен быть >0 и <=100")
        if args.debug_speed != 1 and not args.debug:
            raise ValueError("--debug-speed требует --debug")
        if not 1 <= args.recorder_port <= 65535:
            raise ValueError("Неверный порт рекордера")
        if args.seed is not None and not 0 <= args.seed < 2**32:
            raise ValueError("Seed должен быть от 0 до 4294967295")
        if args.debug_speed != 1:
            config = replace(config, timing=Timing(**{k: v / args.debug_speed for k, v in asdict(config.timing).items()}))
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    from PyQt6.QtWidgets import QApplication
    from scripts.experiments.semaphore.presenter_window import PresenterWindow
    app = QApplication([sys.argv[0]])
    app.setApplicationName("Semaphore Presenter")
    window = PresenterWindow(config, participant=args.participant, host=args.recorder_host,
                             port=args.recorder_port, seed=args.seed, debug=args.debug)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
