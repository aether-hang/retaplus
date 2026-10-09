import json
from pathlib import Path
import pytest
import torch
from torch import nn
from PIL import Image
from retapp.cli import main
from retapp.config import Config, STUDENTS
from retapp.models import Teacher, build_model
from retapp.plugin import RecoveryPlugin
from retapp.pta import persistence_image
from retapp.transport import squared_distances
from tools.continual import allocate_memory
from tools.experiments import cases


def small_model(classes):
    return nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4), nn.ReLU(),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, classes))


def test_pta_critical_distance_gradients():
    z = torch.randn(5, 3, requires_grad=True)
    output = persistence_image(squared_distances(z), cutoff=20.)
    assert output.shape == (512,)
    output.square().sum().backward()
    assert torch.isfinite(z.grad).all() and z.grad.abs().sum() > 0


def test_plugin_gradient_and_connection():
    config = Config(ipc=4, candidates=8, reference_samples=2, hfta_scales=2)
    teacher = Teacher(small_model(2), "imagenette", 16)
    real = torch.rand(8, 3, 16, 16)
    with torch.no_grad():
        features = teacher(real)[1]
    plugin = RecoveryPlugin(teacher, features, torch.rand(8), lambda ids, size: real[ids], 4, config)
    x = torch.rand(4, 3, 16, 16, requires_grad=True)
    plugin.refresh(x)
    plugin.loss(x, torch.arange(4)).backward()
    assert x.grad.abs().sum() > 0 and torch.isfinite(x.grad).all()
    updated = plugin.connect(x)
    assert updated.shape == x.shape and not updated.requires_grad
    teacher.close()


def test_replay_capacity_is_total_not_per_class():
    class Images:
        labels = [cls for cls in range(1000) for _ in range(10)]
    for old, budget in [(100, 1000), (900, 200), (900, 500)]:
        indices = allocate_memory(list(range(old)), budget, Images())
        assert len(indices) == budget and len(set(indices)) == budget
        counts = [sum(Images.labels[i] == c for i in indices) for c in range(old)]
        assert max(counts)-min(counts) <= 1
    assert allocate_memory([], 1000, Images()) == []


def test_experiment_matrix():
    assert len(cases("stc-sweep")) == 35
    assert len(cases("main")) == 15
    assert len(cases("high-ipc")) == 7
    assert len(cases("cross")[0][3]) == 8
    for study in ("main", "high-ipc", "cross", "components", "assignment", "packed", "stc-sweep", "hfta-sweep", "group-pool"):
        for dataset, ipc, options, students, variant in cases(study):
            Config(dataset=dataset, ipc=ipc, **options).validate()


@pytest.mark.parametrize("architecture", STUDENTS)
def test_student_architectures_forward(architecture):
    model = build_model(architecture, classes=10, size=224).eval()
    with torch.no_grad():
        logits = model(torch.rand(1, 3, 224, 224))
    assert logits.shape == (1, 10) and torch.isfinite(logits).all()


@pytest.mark.parametrize("size", [32, 64])
def test_low_resolution_teacher_input_gradients(size):
    teacher = Teacher(build_model("resnet18", classes=10, size=size), "cifar100", size)
    image = torch.rand(2, 3, size, size, requires_grad=True)
    teacher.recovery_loss(image, torch.tensor([0, 1]), .01).backward()
    assert image.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in teacher.model.parameters())
    teacher.close()


def test_cli_all_with_imagefolder_and_four_presets(tmp_path, monkeypatch):
    monkeypatch.setattr("retapp.cli.build_model", lambda arch, classes, size: small_model(classes))
    monkeypatch.setattr("retapp.training.build_model", lambda arch, classes, size: small_model(classes))
    root = tmp_path/"data"
    for split, count in (("train", 6), ("val", 2)):
        for cls in range(10):
            folder = root/split/f"c{cls:02d}"
            folder.mkdir(parents=True)
            for index in range(count):
                rgb = ((cls*23+index*7) % 255, (cls*13+index*37) % 255, (index*31+15) % 255)
                Image.new("RGB", (24, 24), rgb).save(folder/f"{index}.png")
    checkpoint = tmp_path/"teacher.pt"
    torch.save(small_model(10).state_dict(), checkpoint)
    output = tmp_path/"run"
    main(["all", "--dataset", "imagenette", "--ipc", "1", "--data", str(root),
          "--teacher-checkpoint", str(checkpoint), "--output", str(output), "--workers", "0",
          "--device", "cpu", "--set", "recovery_steps=4", "--set", "epochs=1",
          "--set", "candidates=4", "--set", "hfta_scales=2", "--set", "reference_samples=1"])
    result = json.loads((output/"results/resnet18.json").read_text())
    assert 0 <= result["test_accuracy"] <= 100
    assert len(list((output/"synthetic").glob("*/image_*.png"))) == 10
    assert (output/"logs/recovery.output.txt").stat().st_size > 0
    assert len(list((output/"students/resnet18").glob("*/metrics.json"))) == 4
