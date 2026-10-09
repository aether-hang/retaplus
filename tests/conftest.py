import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

torch.set_num_threads(1)


def pytest_runtest_setup(item):
    torch.manual_seed(7)
