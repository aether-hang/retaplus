"""Train the supervised teacher used by all three decoupled stages."""

import argparse
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from retapp.config import Config, DATASETS, STUDENTS
from retapp.data import RealView, Source, split_indices
from retapp.models import build_model
from retapp.runtime import Log, save_tensor, seed_all
from retapp.training import accuracy, normalize


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--dataset", choices=DATASETS, required=True)
    p.add_argument("--architecture", choices=STUDENTS, default="resnet18")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--download", action="store_true")
    a = p.parse_args()
    seed_all(a.seed)
    config = Config(dataset=a.dataset)
    source = Source(a.data, a.dataset, download=a.download)
    split = split_indices(source.labels, a.dataset, config.split_seed)
    train_indices = [i for rows in split["train"].values() for i in rows]
    dev_indices = [i for rows in split["development"].values() for i in rows]
    train = DataLoader(RealView(source, train_indices, config.size, True), batch_size=a.batch_size,
                       shuffle=True, num_workers=a.workers)
    dev = DataLoader(RealView(source, dev_indices, config.size), batch_size=a.batch_size,
                     num_workers=a.workers)
    model = build_model(a.architecture, config.classes, config.size).to(a.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, a.epochs)
    log = Log(a.output.with_suffix(".output.txt"))
    best = -1
    for epoch in range(a.epochs):
        model.train()
        for images, labels in train:
            loss = F.cross_entropy(model(normalize(images.to(a.device), a.dataset)), labels.to(a.device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        scheduler.step()
        score = accuracy(model, dev, a.dataset, a.device)
        log(epoch=epoch, development_accuracy=score, loss=float(loss.detach()))
        if score > best:
            best = score
            save_tensor(a.output, {"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                                   "architecture": a.architecture, "dataset": a.dataset, "seed": a.seed})
    log.close()


if __name__ == "__main__":
    main()
