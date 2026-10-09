"""Teacher/student construction and frozen-teacher recovery statistics."""

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from vendor import cifar_resnet


def build_model(name, classes, size):
    if name in ("resnet18", "resnet50", "resnet101") and size <= 64:
        return getattr(cifar_resnet, name.replace("resnet", "ResNet"))(classes)
    width = 0.5 if name == "mobilenet_v2_0.5" else 1.0
    name = "mobilenet_v2" if name == "mobilenet_v2_0.5" else name
    kwargs = {"width_mult": width} if name == "mobilenet_v2" else {}
    model = getattr(models, name)(weights=None, num_classes=classes, **kwargs)
    return model


def load_weights(model, path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("state_dict", "model"):
        if key in checkpoint and isinstance(checkpoint[key], dict):
            checkpoint = checkpoint[key]
            break
    checkpoint = {k.removeprefix("module."): v for k, v in checkpoint.items()}
    model.load_state_dict(checkpoint, strict=True)


def normalization(dataset):
    if dataset in ("cifar10", "cifar100"):
        return (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class Teacher(nn.Module):
    def __init__(self, model, dataset, size, first_bn_multiplier=10):
        super().__init__()
        self.model, self.size = model.eval(), size
        self.model.requires_grad_(False)
        mean, std = normalization(dataset)
        device = next(model.parameters()).device
        self.register_buffer("mean", torch.tensor(mean, device=device).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std, device=device).view(1, 3, 1, 1))
        self.feature = None
        self.bn_terms, self.collect_bn = [], False
        self.first_bn_multiplier = first_bn_multiplier
        self.handles = []
        linear = [module for module in model.modules() if isinstance(module, nn.Linear)][-1]
        self.classes = linear.out_features
        self.handles.append(linear.register_forward_pre_hook(self._feature_hook))
        for number, module in enumerate(m for m in model.modules() if isinstance(m, nn.BatchNorm2d)):
            self.handles.append(module.register_forward_pre_hook(self._bn_hook(number)))

    def _feature_hook(self, module, values):
        self.feature = values[0].flatten(1)

    def _bn_hook(self, index):
        def hook(module, values):
            if not self.collect_bn:
                return
            x = values[0]
            mean, var = x.mean((0, 2, 3)), x.var((0, 2, 3), unbiased=False)
            weight = self.first_bn_multiplier if index == 0 else 1
            self.bn_terms.append(weight * ((mean-module.running_mean).norm() +
                                           (var-module.running_var).norm()))
        return hook

    def normalize(self, images):
        return (images-self.mean)/self.std

    def denormalize(self, images):
        return images*self.std+self.mean

    def forward(self, images, collect_bn=False):
        if images.shape[-2:] != (self.size, self.size):
            images = F.interpolate(images, (self.size, self.size), mode="bilinear", align_corners=False)
        self.bn_terms, self.collect_bn = [], collect_bn
        logits = self.model(self.normalize(images))
        self.collect_bn = False
        return logits, F.normalize(self.feature, dim=1)

    def recovery_loss(self, images, labels, weight):
        logits, _ = self(images, collect_bn=True)
        bn = torch.stack(self.bn_terms).sum() if self.bn_terms else images.sum()*0
        return F.cross_entropy(logits, labels) + weight*bn

    def close(self):
        for handle in self.handles:
            handle.remove()
