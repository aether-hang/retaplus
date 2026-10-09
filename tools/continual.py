"""Ten-task class-incremental ImageNet learning with a fixed total image buffer."""

import argparse
import json
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from retapp.cli import make_teacher, read_run
from retapp.config import PRESETS
from retapp.data import ImageSet, RealView, Source, save_json
from retapp.models import build_model, load_weights
from retapp.replay import apply_crop, draw_crop
from retapp.runtime import Log, file_hash, save_tensor, seed_all
from retapp.training import accuracy, normalize


def allocate_memory(order, capacity, images):
    if not order:
        return []
    quotient, remainder = divmod(capacity, len(order))
    chosen = []
    for position, cls in enumerate(order):
        count = quotient + (position < remainder)
        candidates = [i for i, label in enumerate(images.labels) if label == cls]
        if len(candidates) < count:
            raise ValueError(f"Class {cls} needs {count} stored images; recover a larger IPC first")
        chosen.extend(candidates[:count])
    assert len(chosen) == capacity
    return chosen


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--teacher-checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--capacity", type=int, choices=(200, 500, 1000), required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    config, metadata = read_run(a.run)
    if config.dataset != "imagenet1k":
        raise ValueError("This protocol requires ImageNet-1K")
    if file_hash(a.teacher_checkpoint) != metadata["teacher_sha256"]:
        raise ValueError("Use the recovery teacher checkpoint")
    if a.output.exists() and any(a.output.iterdir()):
        raise ValueError("Use a new output directory for each capacity and seed")
    a.output.mkdir(parents=True, exist_ok=True)
    order = np.random.RandomState(0).permutation(1000).tolist()
    split = json.loads((a.run/"split.json").read_text())
    source, test_source = Source(a.data, config.dataset), Source(a.data, config.dataset, train=False)
    if source.classes != metadata["classes"] or test_source.classes != metadata["classes"]:
        raise ValueError("Class order differs from the recovery dataset")
    images = ImageSet(a.run/"synthetic")
    teacher = make_teacher(config, a.teacher_checkpoint, a.device)
    save_json(a.output/"protocol.json", {"class_order": order, "capacity": a.capacity,
              "epochs": a.epochs, "batch_size": a.batch_size, "seed": a.seed,
              "synthetic_sha256": images.digest(), "split_sha256": metadata["split_sha256"]})
    results = []
    log = Log(a.output/"output.txt")
    try:
        for preset, (lr, period) in PRESETS.items():
            seed_all(a.seed)
            generator = torch.Generator().manual_seed(a.seed)
            model = build_model("resnet18", 1000, config.size).to(a.device)
            development = []
            for task in range(10):
                current, previous, seen = order[100*task:100*(task+1)], order[:100*task], order[:100*(task+1)]
                memory = allocate_memory(previous, a.capacity, images)
                train_indices = [i for cls in current for i in split["train"][str(cls)]]
                dataset = RealView(source, train_indices, config.size, augment=True)
                loader = DataLoader(dataset, batch_size=a.batch_size, shuffle=True, num_workers=a.workers)
                optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=config.weight_decay)
                for epoch in range(a.epochs):
                    model.train()
                    rate = lr*(1+math.cos(math.pi*epoch/(a.epochs*period)))/2
                    for group in optimizer.param_groups:
                        group["lr"] = rate
                    total, count = 0., 0
                    for x, y in loader:
                        logits = model(normalize(x.to(a.device), config.dataset))
                        # Mask unseen classes, without exposing task identity at evaluation.
                        loss = F.cross_entropy(logits[:, seen], torch.tensor([seen.index(int(v)) for v in y], device=a.device))
                        if memory:
                            ids = torch.randint(len(memory), (len(x),), generator=generator).tolist()
                            crops = []
                            for index in ids:
                                image_id = memory[index]
                                image = images.image(image_id)
                                grid = config.grid_side if config.ipc == 1 and config.cell_replay and config.cell_slots else 0
                                state = draw_crop(image_id, *image.shape[-2:], generator, grid, config.crop_min)
                                crops.append(apply_crop(image, state, config.size))
                            replay = torch.stack(crops).to(a.device)
                            with torch.no_grad():
                                target, _ = teacher(replay)
                            prediction = model(normalize(replay, config.dataset))
                            temp = config.temperature
                            loss = loss + F.kl_div(F.log_softmax(prediction[:, seen]/temp, 1),
                                                 F.softmax(target[:, seen]/temp, 1), reduction="batchmean")*temp**2
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        optimizer.step()
                        total += float(loss.detach())*len(x)
                        count += len(x)
                    log(preset=preset, task=task, epoch=epoch, loss=total/count, memory_images=len(memory))
                ids = [i for cls in seen for i in split["development"][str(cls)]]
                dev = DataLoader(RealView(source, ids, config.size), batch_size=a.batch_size, num_workers=a.workers)
                development.append(accuracy(model, dev, config.dataset, a.device, seen))
                save_tensor(a.output/preset/f"task_{task:02d}.pt", {"model": {k: v.cpu() for k, v in model.state_dict().items()}})
            results.append({"preset": preset, "development_curve": development,
                            "development_average": sum(development)/10})
        best = max(results, key=lambda x: x["development_average"])
        curve = []
        for task in range(10):
            seen = order[:100*(task+1)]
            load_weights(model, a.output/best["preset"]/f"task_{task:02d}.pt")
            indices = [i for i, label in enumerate(test_source.labels) if label in set(seen)]
            loader = DataLoader(RealView(test_source, indices, config.size), batch_size=a.batch_size, num_workers=a.workers)
            curve.append(accuracy(model, loader, config.dataset, a.device, seen))
        save_json(a.output/"results.json", {"selection": results, "preset": best["preset"],
                  "stage_accuracy": curve, "average_incremental_accuracy": sum(curve)/10})
    finally:
        teacher.close()
        log.close()


if __name__ == "__main__":
    main()
