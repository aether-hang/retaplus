"""Configuration shared by recovery, cached relabeling, and evaluation."""

from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path


DATASETS = {
    "cifar10": (10, 32), "cifar100": (100, 32),
    "tiny_imagenet": (200, 64), "imagenet1k": (1000, 224),
    **{name: (10, 224) for name in (
        "imagenette", "imagewoof", "imagefruit", "imageyellow",
        "imagemeow", "imagesquawk")},
}
STUDENTS = ("resnet18", "resnet50", "resnet101", "efficientnet_b0", "mobilenet_v2",
            "shufflenet_v2_x0_5", "swin_t", "wide_resnet50_2",
            "densenet121", "densenet169", "densenet201")
PRESETS = {"S1": (0.001, 1), "S2": (0.001, 2),
           "S3": (0.0005, 1), "S4": (0.0005, 2)}


@dataclass
class Config:
    dataset: str = "imagenette"
    ipc: int = 10
    seed: int = 42
    split_seed: int = 42
    teacher: str = "resnet18"
    recovery_steps: int = 0
    blocks: int = 4
    recovery_lr: float = 0.25
    alpha: float = 0.5
    bn_weight: float = 0.01
    first_bn_multiplier: float = 10.0
    group_size: int = 4
    candidates: int = 32
    complexity_weight: float = 0.1
    structure_weight: float = 0.2
    entropy: float = 0.05
    transport_updates: int = 3
    sinkhorn_iterations: int = 1000
    sinkhorn_tolerance: float = 1e-7
    assignment: str = "stc"
    topology: str = "hfta"
    hfta_weight: float = 0.5
    hfta_scales: int = 8
    tau_fraction: float = 0.5
    rho_fraction: float = 0.05
    cloud_cap: int = 32
    reference_samples: int = 20
    topology_every: int = 10
    grid_side: int = 2
    cell_slots: bool = True
    cell_replay: bool = True
    epochs: int = 0
    batch_size: int = 16
    temperature: float = 20.0
    crop_min: float = 0.08
    weight_decay: float = 0.01
    recovery_jitter: int = 4
    log_every: int = 10

    @property
    def size(self):
        return DATASETS[self.dataset][1]

    @property
    def classes(self):
        return DATASETS[self.dataset][0]

    @property
    def slots(self):
        return self.grid_side ** 2 if self.ipc == 1 and self.cell_slots else self.ipc

    @property
    def effective_group(self):
        return min(self.slots, self.group_size) if self.ipc > 1 else self.slots

    @property
    def steps(self):
        return self.recovery_steps or (2000 if self.ipc == 1 and self.dataset in
                                      ("cifar100", "tiny_imagenet") else 300)

    @property
    def student_epochs(self):
        return self.epochs or (1000 if self.dataset in ("cifar10", "cifar100") or
                              (self.dataset == "tiny_imagenet" and self.ipc == 1) else 300)

    def validate(self):
        if self.dataset not in DATASETS or self.ipc < 1:
            raise ValueError("Unknown dataset or nonpositive IPC")
        if not 0 <= self.alpha <= 1 or self.entropy <= 0:
            raise ValueError("Invalid residual or entropy setting")
        if self.assignment not in ("stc", "independent", "static", "balanced", "none"):
            raise ValueError("Unknown assignment")
        if self.topology not in ("hfta", "none", "pta"):
            raise ValueError("Unknown topology objective")
        for key in ("blocks", "group_size", "candidates", "hfta_scales", "cloud_cap",
                    "reference_samples", "topology_every", "grid_side", "batch_size"):
            if getattr(self, key) < 1:
                raise ValueError(f"{key} must be positive")
        if self.cloud_cap < self.effective_group or self.candidates < self.effective_group:
            raise ValueError("Cloud and candidate caps must include the active group")
        if self.steps < self.blocks or self.hfta_scales < 2:
            raise ValueError("Insufficient steps or filtration scales")
        if self.tau_fraction <= 0 or self.rho_fraction <= 0 or self.temperature <= 0:
            raise ValueError("Smoothing and temperature must be positive")
        if self.transport_updates < 0 or self.sinkhorn_iterations < 1:
            raise ValueError("Invalid solver iteration counts")
        if self.cell_slots and self.ipc == 1 and self.size % self.grid_side:
            raise ValueError("Image size must be divisible by grid_side")
        return self

    def to_dict(self):
        return asdict(self)

    @classmethod
    def read(cls, path, **overrides):
        values = json.loads(Path(path).read_text(encoding="utf-8")) if path else {}
        values.update({k: v for k, v in overrides.items() if v is not None})
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
        return cls(**values).validate()
