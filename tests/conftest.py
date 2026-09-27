import sys
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")


@pytest.fixture(scope="session")
def store():
    from dashboard.inference import ModelStore
    return ModelStore()
