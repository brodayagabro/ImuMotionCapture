from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = EXPERIMENT_ROOT.parents[1]
RECORDS_ROOT = EXPERIMENT_ROOT / 'records'
