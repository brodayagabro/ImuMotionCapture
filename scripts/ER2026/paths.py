"""Location-independent paths for the conference experiment and shared SDK."""
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_ROOT.parents[1]
RECORDS_ROOT = EXPERIMENT_ROOT / 'records'
OUTPUT_ROOT = EXPERIMENT_ROOT / 'output'
SDK_ROOT = PROJECT_ROOT / 'scripts' / 'MocapApi'
