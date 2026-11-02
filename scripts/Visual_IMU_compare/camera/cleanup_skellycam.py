#!/usr/bin/env python3
"""Remove obsolete camera SkellyCam dependencies; list only unless --apply is given."""
import argparse
from contextlib import ExitStack
import fcntl
import os
from pathlib import Path
import shutil
import subprocess


TARGETS = (".venv-skellycam", ".tools/recording-bootstrap", "repositories/skellycam")


def check_inactive(root):
    for unit in Path("/etc/systemd/system").glob("visual-mocap-camera-*.service"):
        text = unit.read_text()
        if str(root) in text and ("skellycam_node.py" in text or ".venv-skellycam" in text):
            raise ValueError(f"Replace the old service with install_service.py --install first: {unit}")
    for process in Path("/proc").glob("[0-9]*"):
        if process.name == str(os.getpid()):
            continue
        try:
            args = (process / "cmdline").read_bytes().decode(errors="replace").split("\0")
            old_env = any(str(root / ".venv-skellycam") in arg for arg in args)
            old_node = any(Path(arg).name == "skellycam_node.py" and
                           (arg.startswith(str(root) + "/") or (process / "cwd").resolve() == root)
                           for arg in args if arg)
            if old_env or old_node:
                raise ValueError(f"Stop the old SkellyCam process first (PID {process.name})")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue


def cleanup(root, apply=False):
    root = root.resolve()
    targets = []
    for relative in TARGETS:
        target = root / relative
        for part in (target, *target.parents):
            if part == root:
                break
            if part.is_symlink():
                raise ValueError(f"Refusing symlink: {part}")
        if target.exists():
            if not target.is_dir():
                raise ValueError(f"Expected a directory: {target}")
            targets.append(target)
    if not apply or not targets:
        return targets
    # The RTSP environment may use the shared .tools/python. Never remove it.
    python = root / ".venv/bin/python"
    if not python.is_file():
        raise ValueError("Run camera/setup.sh to prepare .venv before cleanup")
    if any(python.resolve().is_relative_to(target) for target in targets):
        raise ValueError("RTSP Python depends on an old directory; recreate .venv first")
    subprocess.run([str(python), "-c", "import sys; assert sys.version_info >= (3,10)"], check=True)
    with ExitStack() as stack:
        for name in ("setup.lock", "recording-setup.lock"):
            lock = stack.enter_context((root / ".tools" / name).open("a"))
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError("Setup is running; wait for it to finish") from exc
        check_inactive(root)
        repo = root / "repositories/skellycam"
        if repo in targets:
            if not (repo / ".git").is_dir():
                raise ValueError(f"Unrecognized checkout; preserve it manually: {repo}")
            status = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True)
            if status.strip():
                raise ValueError(f"Local modifications in {repo}; preserve them before cleanup")
        # Validate everything before deleting the first directory.
        for target in targets:
            shutil.rmtree(target)
    return targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Delete the listed obsolete dependencies")
    args = parser.parse_args()
    try:
        targets = cleanup(Path(__file__).parent, args.apply)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")
    for target in targets:
        print(f"{'Removed' if args.apply else 'Would remove'}: {target}")
    print("Recordings, .venv, .tools/python, caches and server dependencies are preserved.")
    if targets and not args.apply:
        print("Replace/stop the old service, then repeat with --apply to remove these directories.")


if __name__ == "__main__":
    main()
