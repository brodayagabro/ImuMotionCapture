"""Run from any directory: python /path/to/Visual_IMU_compare/src/main.py."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[3]
for directory in (ROOT, ROOT / 'host', ROOT / 'host/viz'):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))


def main(argv=None):
    parser = argparse.ArgumentParser(description='IMU + visual BVH: semaphore experiment')
    parser.add_argument('--bind', default='0.0.0.0', help='Local BVH listener address')
    parser.add_argument('--port', type=int, default=7001, help='Local BVH listener port')
    parser.add_argument('--transport', choices=['udp', 'tcp'], default='udp')
    parser.add_argument('--source-ip', help='Accept only this video server IPv4 address')
    parser.add_argument('--stale-timeout-ms', type=int, default=500)
    parser.add_argument('--records-dir', dest='recordings_dir')
    parser.add_argument('--event-port', type=int, default=0, help='Optional localhost experiment control port')
    parser.add_argument('--calibration-profile', type=Path, help='Restore own IMU calibration')
    parser.add_argument('--manual-connect', action='store_true', help='Use Connect/Start buttons instead of automatic listen')
    args = parser.parse_args(argv)
    from scripts.Visual_IMU_compare.src.config import ComparisonConfig
    options = {key: value for key, value in vars(args).items()
               if key not in {'manual_connect', 'calibration_profile'} and value is not None}
    try:
        config = ComparisonConfig(**options)
    except ValueError as error:
        parser.error(str(error))
    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QApplication
    from scripts.Visual_IMU_compare.src.comparison_window import MocapComparisonWindow
    app = QApplication([sys.argv[0]])
    window = MocapComparisonWindow(config)
    if args.calibration_profile:
        window.load_calibration_profile(args.calibration_profile)
    window.show()
    if not args.manual_connect:
        QTimer.singleShot(0, lambda: (window.visual_worker.command('connect'),
                                     window.visual_worker.command('start')))
    return app.exec()


if __name__ == '__main__':
    raise SystemExit(main())
