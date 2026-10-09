"""Dataset views and fixed class-stratified development splits."""

import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
from torchvision.transforms import functional as TF


class Source:
    def __init__(self, root, dataset, train=True, download=False):
        root = Path(root)
        if dataset in ("cifar10", "cifar100"):
            factory = CIFAR10 if dataset == "cifar10" else CIFAR100
            self.data = factory(str(root), train=train, download=download)
            self.labels = list(self.data.targets)
        else:
            folder = root / ("train" if train else "val")
            self.data = ImageFolder(str(folder))
            self.labels = list(self.data.targets)
        self.classes = list(self.data.classes)

    def __len__(self):
        return len(self.labels)

    def image(self, index, size):
        image, _ = self.data[index]
        return TF.pil_to_tensor(image.resize((size, size), Image.Resampling.BILINEAR)).float()/255

    def batch(self, indices, size):
        return torch.stack([self.image(int(i), size) for i in indices])


def split_indices(labels, dataset, seed=42):
    generator = np.random.default_rng(seed)
    labels = np.asarray(labels)
    train, development = {}, {}
    for cls in np.unique(labels):
        indices = np.flatnonzero(labels == cls)
        indices = generator.permutation(indices)
        count = 50 if dataset == "imagenet1k" else max(1, int(np.ceil(0.1*len(indices))))
        if count >= len(indices):
            raise ValueError("A class has no examples remaining after its development split")
        development[str(cls)] = sorted(indices[:count].tolist())
        train[str(cls)] = sorted(indices[count:].tolist())
    return {"seed": seed, "dataset": dataset, "train": train, "development": development}


def split_digest(split):
    return hashlib.sha256(json.dumps(split, sort_keys=True).encode()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


class RealView(torch.utils.data.Dataset):
    def __init__(self, source, indices, size, augment=False):
        self.source, self.indices, self.size, self.augment = source, list(indices), size, augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        index = self.indices[index]
        image = self.source.image(index, self.size)
        if self.augment:
            from torchvision.transforms import RandomCrop
            image = TF.pad(image, [4, 4, 4, 4], padding_mode="reflect")
            top, left, height, width = RandomCrop.get_params(image, (self.size, self.size))
            image = TF.crop(image, top, left, height, width)
            if torch.rand(()) < .5:
                image = TF.hflip(image)
        return image, self.source.labels[index]


class ImageSet:
    def __init__(self, root):
        self.root = Path(root)
        self.files = sorted(self.root.glob("[0-9]*/image_*.png"))
        if not self.files:
            raise FileNotFoundError(f"No synthetic images under {root}")
        self.labels = [int(path.parent.name) for path in self.files]

    def __len__(self):
        return len(self.files)

    def image(self, index):
        with Image.open(self.files[index]) as image:
            return TF.pil_to_tensor(image.convert("RGB")).float()/255

    def digest(self):
        digest = hashlib.sha256()
        for path in self.files:
            digest.update(path.relative_to(self.root).as_posix().encode())
            digest.update(path.read_bytes())
        return digest.hexdigest()
