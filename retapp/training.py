"""Cached teacher supervision, held-out selection, and fresh-student training."""

import json
import math
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from .config import PRESETS
from .data import ImageSet, RealView, save_json
from .models import build_model, normalization
from .replay import apply_crop, draw_crop, encode_states, replay_batch
from .runtime import save_tensor, seed_all


def normalize(images, dataset):
    mean, std = normalization(dataset)
    return (images-images.new_tensor(mean)[None, :, None, None])/images.new_tensor(std)[None, :, None, None]


@torch.no_grad()
def relabel(images, teacher, config, directory, log):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    metadata = {"images_sha256": images.digest(), "epochs": config.student_epochs,
                "batch_size": config.batch_size, "seed": config.seed, "size": config.size,
                "cell_grid": config.grid_side if config.ipc == 1 and config.cell_slots and config.cell_replay else 0,
                "crop_min": config.crop_min, "classes": teacher.classes}
    metadata_path = directory/"metadata.json"
    if metadata_path.exists() and json.loads(metadata_path.read_text()) != metadata:
        raise ValueError("Label-cache geometry, image checksum, or schedule changed")
    save_json(metadata_path, metadata)
    for epoch in range(config.student_epochs):
        folder = directory/f"epoch_{epoch:04d}"
        marker = folder/"complete.json"
        if marker.exists():
            continue
        folder.mkdir(exist_ok=True)
        generator = torch.Generator().manual_seed(config.seed+epoch*7919)
        order = torch.randperm(len(images), generator=generator).tolist()
        count = 0
        for start in range(0, len(order), config.batch_size):
            states, crops = [], []
            for index in order[start:start+config.batch_size]:
                image = images.image(index)
                state = draw_crop(index, *image.shape[-2:], generator,
                                  metadata["cell_grid"], config.crop_min)
                states.append(state)
                crops.append(apply_crop(image, state, config.size))
            logits, _ = teacher(torch.stack(crops).to(teacher.mean.device))
            payload = {"states": encode_states(states), "logits": logits.cpu(),
                       "labels": torch.tensor([images.labels[s.image] for s in states])}
            save_tensor(folder/f"batch_{count:06d}.pt", payload)
            count += 1
        save_json(marker, {"batches": count, "images": len(images)})
        log(stage="relabel", epoch=epoch, batches=count, images=len(images))
    return metadata


@torch.no_grad()
def accuracy(model, loader, dataset, device, seen_classes=None):
    model.eval()
    correct, count = 0, 0
    for images, labels in loader:
        logits = model(normalize(images.to(device), dataset))
        if seen_classes is not None:
            mask = torch.ones(logits.shape[1], dtype=torch.bool, device=device)
            mask[seen_classes] = False
            logits[:, mask] = -torch.inf
        correct += int((logits.argmax(1).cpu() == labels).sum())
        count += len(labels)
    if not count:
        raise ValueError("Evaluation loader is empty")
    return 100*correct/count


def train_student(images, config, labels_dir, development, arch, preset, output, device, log):
    seed_all(config.seed)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((Path(labels_dir)/"metadata.json").read_text())
    if metadata["images_sha256"] != images.digest():
        raise ValueError("Synthetic images differ from those used for relabeling")
    if metadata["batch_size"] != config.batch_size or metadata["epochs"] != config.student_epochs:
        raise ValueError("Student batch size/epochs must match the saved label cache")
    model = build_model(arch, metadata["classes"], config.size).to(device)
    lr, period = PRESETS[preset]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=config.weight_decay)
    for epoch in range(config.student_epochs):
        model.train()
        rate = (1+math.cos(math.pi*epoch/(config.student_epochs*period)))/2
        for group in optimizer.param_groups:
            group["lr"] = lr*rate
        folder = Path(labels_dir)/f"epoch_{epoch:04d}"
        completed = json.loads((folder/"complete.json").read_text())
        files = sorted(folder.glob("batch_*.pt"))
        if len(files) != completed["batches"]:
            raise ValueError(f"Incomplete label cache in {folder}")
        total, count = 0., 0
        for path in files:
            batch = torch.load(path, map_location="cpu", weights_only=True)
            x = replay_batch(images, batch["states"], config.size).to(device)
            target = batch["logits"].to(device)
            # A final singleton batch can use frozen BN statistics without dropping it.
            if len(x) == 1:
                for layer in model.modules():
                    if isinstance(layer, torch.nn.BatchNorm2d):
                        layer.eval()
            else:
                model.train()
            logits = model(normalize(x, config.dataset))
            temperature = config.temperature
            loss = F.kl_div(F.log_softmax(logits/temperature, 1),
                            F.softmax(target/temperature, 1), reduction="batchmean")*temperature**2
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach())*len(x)
            count += len(x)
        log(stage="student", architecture=arch, preset=preset, epoch=epoch,
            loss=total/count, learning_rate=lr*rate)
    development_accuracy = accuracy(model, development, config.dataset, device)
    save_tensor(output/"student.pt", {"model": model.cpu().state_dict(), "architecture": arch,
                                      "classes": metadata["classes"], "config": config.to_dict()})
    result = {"architecture": arch, "preset": preset, "seed": config.seed,
              "development_accuracy": development_accuracy, "epochs": config.student_epochs}
    save_json(output/"metrics.json", result)
    return result


def select_preset(directory, architecture):
    directory = Path(directory)/architecture
    results = []
    for preset in PRESETS:
        path = directory/preset/"metrics.json"
        if not path.exists():
            raise FileNotFoundError(f"Train all four held-out presets before selection: {path}")
        results.append(json.loads(path.read_text()))
    best = max(results, key=lambda result: result["development_accuracy"])
    return directory/best["preset"]/"student.pt", best
