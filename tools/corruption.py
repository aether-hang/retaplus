"""Evaluate the 15 x 5 common-corruption conditions, then aggregate runs."""

import argparse
import json
from pathlib import Path
import statistics
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from retapp.cli import read_run
from retapp.data import save_json
from retapp.models import build_model, load_weights
from retapp.training import accuracy, select_preset

CORRUPTIONS = ("gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur",
               "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog",
               "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression")


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--runs", type=Path, nargs="+", required=True)
    p.add_argument("--corruptions", type=Path, required=True,
                   help="root/type/severity/class/*.jpg")
    p.add_argument("--architecture", default="resnet18")
    p.add_argument("--output", type=Path, default=Path("corruption_summary.json"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    results, settings = [], None
    for run in a.runs:
        config, metadata = read_run(run)
        current = {k: v for k, v in config.to_dict().items() if k != "seed"}
        if settings is not None and current != settings:
            raise ValueError("Aggregate only identical configurations with different seeds")
        settings = current
        checkpoint, chosen = select_preset(run/"students", a.architecture)
        model = build_model(a.architecture, config.classes, config.size).to(a.device)
        load_weights(model, checkpoint)
        transform = transforms.Compose([transforms.Resize((config.size, config.size)), transforms.ToTensor()])
        conditions = []
        for corruption in CORRUPTIONS:
            for severity in range(1, 6):
                dataset = ImageFolder(str(a.corruptions/corruption/str(severity)), transform=transform)
                if dataset.classes != metadata["classes"]:
                    raise ValueError("Corruption class order differs from the recovery class order")
                loader = DataLoader(dataset, batch_size=128, num_workers=a.workers)
                value = accuracy(model, loader, config.dataset, a.device)
                conditions.append({"corruption": corruption, "severity": severity, "accuracy": value})
                print(json.dumps({"seed": config.seed, **conditions[-1]}), flush=True)
        result = {"seed": config.seed, "preset": chosen["preset"], "conditions": conditions,
                  "mean_accuracy": statistics.mean(x["accuracy"] for x in conditions)}
        save_json(run/"results"/f"corruption_{a.architecture}.json", result)
        results.append(result)
    if len({r["seed"] for r in results}) != len(results):
        raise ValueError("Independent runs must have distinct seeds")
    values = [r["mean_accuracy"] for r in results]
    save_json(a.output, {"runs": results, "mean": statistics.mean(values),
                        "sample_std": statistics.stdev(values) if len(values) > 1 else None})


if __name__ == "__main__":
    main()
