import json
from dataclasses import replace
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from retapp.config import Config, PRESETS
from retapp.data import ImageSet, save_json, split_indices
from retapp.models import Teacher
from retapp.recovery import recover_class
from retapp.replay import apply_crop, draw_crop, pack_cells, replay_batch
from retapp.training import relabel, select_preset, train_student


def tiny_model(classes=2):
    return nn.Sequential(nn.Conv2d(3, 6, 3, padding=1), nn.BatchNorm2d(6), nn.ReLU(),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(6, classes))


class MiniSource:
    def __init__(self):
        self.images = torch.rand(12, 3, 32, 32)

    def batch(self, indices, size):
        return torch.nn.functional.interpolate(self.images[list(indices)], (size, size), mode="bilinear", align_corners=False)


def test_packed_crop_replay():
    cells = torch.arange(4).float()[:, None, None, None].expand(4, 3, 8, 8)/3
    packed = pack_cells(cells, 2)
    generator = torch.Generator().manual_seed(0)
    for _ in range(20):
        state = draw_crop(0, 16, 16, generator, 2)
        crop = apply_crop(packed, state, 32)
        torch.testing.assert_close(crop, torch.full_like(crop, state.cell/3))


@pytest.mark.parametrize("ipc", [1, 5])
def test_real_recovery_gradients_and_diagnostics(tmp_path, ipc):
    config = Config(dataset="cifar100", ipc=ipc, recovery_steps=4, hfta_scales=2,
                    reference_samples=2, cloud_cap=5, topology_every=1, candidates=8)
    teacher = Teacher(tiny_model(), config.dataset, config.size)
    images, diagnostic = recover_class(MiniSource(), list(range(12)), teacher, config, 0, tmp_path, lambda **kwargs: None)
    teacher.close()
    assert images.shape == (ipc, 3, 32, 32)
    assert len(diagnostic["injections"]) == 3
    expected_groups = 1 if ipc == 1 else 2
    assert len(diagnostic["gradient_statistics"]) == 4*expected_groups
    assert all(torch.isfinite(z).all() for z in diagnostic["checkpoints"])
    assert len(ImageSet(tmp_path/"synthetic")) == ipc
    for item in diagnostic["injections"]:
        assert len(item["indices"][:4].unique()) == 4


def test_cached_labels_training_and_heldout_selection(tmp_path, monkeypatch):
    config = Config(dataset="cifar100", ipc=1, recovery_steps=4, topology="none",
                    epochs=1, batch_size=2, candidates=8)
    teacher = Teacher(tiny_model(), config.dataset, config.size)
    source = MiniSource()
    for cls in range(2):
        recover_class(source, list(range(12)), teacher, config, cls, tmp_path, lambda **kw: None)
    images = ImageSet(tmp_path/"synthetic")
    directory = tmp_path/"labels"
    meta = relabel(images, teacher, config, directory, lambda **kw: None)
    batch = torch.load(directory/"epoch_0000/batch_000000.pt", weights_only=True)
    teacher.eval()
    with torch.no_grad():
        expected, _ = teacher(replay_batch(images, batch["states"], config.size))
    torch.testing.assert_close(batch["logits"], expected)
    assert meta["cell_grid"] == 2
    dev = DataLoader(TensorDataset(torch.rand(4, 3, 32, 32), torch.tensor([0, 1, 0, 1])), batch_size=2)
    monkeypatch.setattr("retapp.training.build_model", lambda name, classes, size: tiny_model(classes))
    for preset in PRESETS:
        result = train_student(images, config, directory, dev, "resnet18", preset,
                               tmp_path/"students/resnet18"/preset, "cpu", lambda **kw: None)
        assert 0 <= result["development_accuracy"] <= 100
    path, selected = select_preset(tmp_path/"students", "resnet18")
    assert path.exists() and "test_accuracy" not in selected
    with pytest.raises(ValueError, match="batch size"):
        train_student(images, replace(config, batch_size=4), directory, dev, "resnet18", "S1", tmp_path/"bad", "cpu", lambda **kw: None)
    teacher.close()


def test_split_and_paper_defaults():
    split = split_indices([0]*20+[1]*20, "imagenette")
    for cls in ("0", "1"):
        assert len(split["development"][cls]) == 2
        assert not set(split["train"][cls]) & set(split["development"][cls])
    config = Config(ipc=1)
    assert config.slots == 4 and config.topology == "hfta" and config.cell_replay
    assert Config(dataset="cifar100", ipc=1).steps == 2000
    assert Config(dataset="imagenet1k", ipc=10).steps == 300
