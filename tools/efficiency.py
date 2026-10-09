"""Summarize measured recovery time and peak allocation from a completed run."""

import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retapp.data import save_json


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--run", type=Path, required=True)
    a = p.parse_args()
    rows = [json.loads(path.read_text()) for path in (a.run/"synthetic").glob("*/complete.json")]
    if not rows:
        raise FileNotFoundError("No completed classes")
    seconds, images = sum(x["seconds"] for x in rows), sum(x["images"] for x in rows)
    memory = a.run/"recovery_memory.json"
    result = {"classes": len(rows), "images": images, "seconds": seconds,
              "seconds_per_image": seconds/images,
              "peak_allocated_gib": json.loads(memory.read_text())["peak_bytes"]/2**30 if memory.exists() else None}
    save_json(a.run/"results/efficiency.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
