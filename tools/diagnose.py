"""Aggregate class diagnostics emitted by recovery."""

import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from retapp.data import save_json
from retapp.diagnostics import alignment, summarize


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--run", type=Path, required=True)
    a = p.parse_args()
    rows, angles = [], []
    for path in sorted((a.run/"diagnostics").glob("class_*.pt")):
        diagnostic = torch.load(path, map_location="cpu", weights_only=True)
        rows.append(summarize(diagnostic))
        angles.extend(alignment(x["before"], x["anchors"]) for x in diagnostic["injections"])
    if not rows:
        raise FileNotFoundError("Run recovery before computing diagnostics")
    result = {"classes": rows}
    for key in ("shared", "betti_gap", "coverage", "weak_images"):
        values = [row[key] for row in rows if row[key] is not None]
        result[key] = sum(values)/len(values) if values else None
    angles = torch.cat(angles) if angles else torch.empty(0)
    result["cosine_median"] = float(angles.median()) if len(angles) else None
    result["spread_curve"] = torch.tensor([row["spread_curve"] for row in rows]).mean(0).tolist()
    save_json(a.run/"results/diagnostics.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "classes"}, indent=2))


if __name__ == "__main__":
    main()
