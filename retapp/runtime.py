"""Reproducible seeds, real-time logs, and atomic checkpoint output."""

import hashlib
import json
import random
import sys
from pathlib import Path
import numpy as np
import torch


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_tensor(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


class Log:
    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open("a", encoding="utf-8", buffering=1)

    def __call__(self, **values):
        text = json.dumps(values)
        print(text, flush=True)
        self.file.write(text + "\n")

    def close(self):
        self.file.close()
