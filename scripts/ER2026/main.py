"""Run from any working directory: python /path/to/scripts/ER2026/main.py."""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for directory in (PROJECT_ROOT, PROJECT_ROOT / 'host', PROJECT_ROOT / 'host' / 'viz'):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from scripts.ER2026.noitom_comparison.main import main

if __name__ == '__main__':
    raise SystemExit(main())
