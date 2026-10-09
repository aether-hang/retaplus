"""Summarize completed evaluation files, grouped by setting and architecture."""

import argparse
import json
from pathlib import Path
import statistics
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retapp.data import save_json


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("summary.json"))
    args = parser.parse_args()
    groups = {}
    for path in sorted(args.root.rglob("results/*.json")):
        value = json.loads(path.read_text())
        if "test_accuracy" not in value:
            continue
        run = json.loads((path.parent.parent/"run.json").read_text())
        config = {k: v for k, v in run["config"].items() if k != "seed"}
        key = json.dumps([config, value["architecture"]], sort_keys=True)
        groups.setdefault(key, []).append({**value, "file": str(path)})
    summaries = []
    for key, runs in groups.items():
        config, architecture = json.loads(key)
        seeds = [run["seed"] for run in runs]
        if len(seeds) != len(set(seeds)):
            raise ValueError("Duplicate seed for the same setting; use a narrower --root")
        values = [run["test_accuracy"] for run in runs]
        summaries.append({"config": config, "architecture": architecture, "runs": runs,
                          "mean": statistics.mean(values),
                          "sample_std": statistics.stdev(values) if len(values) > 1 else None})
    save_json(args.output, summaries)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
