"""Bounded live SDK reception check, independent of Qt and synthetic fixtures."""
import json
from pathlib import Path
import time
from .noitom_adapter import NoitomAdapter
from .noitom_client import NoitomClient


def run_probe(config, duration_s, output_path=None):
    if duration_s <= 0:
        raise ValueError("Probe duration must be positive")
    client, adapter = NoitomClient(config), NoitomAdapter()
    report = {"configuration": config.metadata(), "frames": 0, "mapped_frames": 0,
              "index_discontinuities": 0, "errors": []}
    previous = None
    started = time.monotonic()
    try:
        client.connect()
        report.update(sdk_version=client.version, delivery_mode=client.delivery_mode,
                      sdk_render_settings=client.sdk_render_settings)
        client.start()
        started = time.monotonic()
        while time.monotonic() - started < duration_s:
            for frame in client.poll_events():
                report["frames"] += 1
                report["joint_count"] = len(frame.joints)
                report["avatar_name"] = frame.avatar_name
                report["avatar_index"] = frame.avatar_index
                report.setdefault("first_sdk_frame_index", frame.frame_index)
                report["last_sdk_frame_index"] = frame.frame_index
                if previous is not None and frame.frame_index is not None:
                    report["index_discontinuities"] += (frame.frame_index - previous) % 2**32 != 1
                previous = frame.frame_index
                try:
                    report["mapped_segments"] = list(adapter.orientations(frame))
                    report["mapped_frames"] += 1
                except ValueError as error:
                    if str(error) not in report["errors"]:
                        report["errors"].append(str(error))
            time.sleep(.001)
    except Exception as error:
        report["errors"].append(str(error))
    finally:
        report["elapsed_s"] = time.monotonic() - started
        try:
            client.disconnect()
        except Exception as error:
            report["errors"].append(str(error))
    report["rate_hz"] = report["frames"] / max(report["elapsed_s"], 1e-9)
    report["ok"] = report["frames"] > 0 and report["mapped_frames"] == report["frames"] and not report["errors"]
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
