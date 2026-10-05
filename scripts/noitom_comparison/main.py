"""Launch with python scripts/noitom_comparison/main.py [--fake]."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "host" / "viz"))


def main(argv=None):
    parser = argparse.ArgumentParser(description="IMU + Noitom simultaneous capture")
    parser.add_argument("--sdk-library", help="Path to MocapApi.dll / libMocapApi.so")
    parser.add_argument("--transport", choices=("udp", "tcp"), default="udp")
    parser.add_argument("--port", type=int, default=7012)
    parser.add_argument("--server", default="127.0.0.1")
    parser.add_argument("--bvh-format", choices=("binary", "string", "legacy"), default="binary")
    parser.add_argument("--bvh-rotation", choices=("XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"), default="XYZ")
    parser.add_argument("--avatar-index", type=int)
    parser.add_argument("--stale-timeout-ms", type=int, default=500)
    parser.add_argument("--recordings-dir", default=str(ROOT / "recordings"))
    parser.add_argument("--fake", action="store_true", help="Synthetic Noitom, no SDK or suit")
    parser.add_argument("--event-port", type=int, default=5055,
                        help="Localhost event receiver port; 0 disables it (default: 5055)")
    parser.add_argument("--replay", type=Path, help="Replay a recorded session without hardware")
    parser.add_argument("--probe-seconds", type=float, help="Test live Noitom reception without opening a window")
    parser.add_argument("--probe-output", type=Path, help="Save the live reception report as JSON")
    args = parser.parse_args(argv)
    from scripts.noitom_comparison.config import ComparisonConfig
    options = {key: value for key, value in vars(args).items()
               if key not in {"replay", "probe_seconds", "probe_output"}}
    try:
        config = ComparisonConfig(**options)
    except ValueError as error:
        parser.error(str(error))
    if args.probe_seconds is not None:
        if args.probe_seconds <= 0 or args.fake or args.replay:
            parser.error("--probe-seconds must be positive and uses a live stream (no --fake/--replay)")
        from scripts.noitom_comparison.probe import run_probe
        report = run_probe(config, args.probe_seconds, args.probe_output)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["ok"] else 1
    from PyQt6.QtWidgets import QApplication
    app = QApplication([sys.argv[0]])
    if args.replay:
        from scripts.noitom_comparison.replay import ReplayWindow
        window = ReplayWindow(args.replay)
    else:
        from scripts.noitom_comparison.comparison_window import MocapComparisonWindow
        window = MocapComparisonWindow(config)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
