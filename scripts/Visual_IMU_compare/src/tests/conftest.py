import json
from pathlib import Path
import pytest


@pytest.fixture
def packet_pair():
    return json.loads((Path(__file__).parent / 'fixtures/visual_cm.json').read_text())
