"""Grouped pixel recovery with stage-wise distinct residual anchors."""

import math
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from torchvision.transforms import functional as TF
from .data import save_json
from .replay import pack_cells
from .runtime import save_tensor
from .topology import ClassTopology
from .transport import assign, patch_complexity


@torch.no_grad()
def encode(teacher, images, batch_size=32):
    features, probabilities = [], []
    for batch in images.split(batch_size):
        logits, z = teacher(batch)
        features.append(z.detach())
        probabilities.append(logits.softmax(1))
    return torch.cat(features), torch.cat(probabilities)


class Pool:
    def __init__(self, source, indices, teacher, config, cls):
        self.source, self.indices, self.teacher = source, list(indices), teacher
        self.config, self.cls = config, cls
        device = teacher.mean.device
        features, confidence = [], []
        with torch.no_grad():
            for start in range(0, len(indices), 32):
                images = source.batch(self.indices[start:start+32], config.size).to(device)
                z, probs = encode(teacher, images)
                features.append(z)
                confidence.append(probs[:, cls])
        self.features = torch.cat(features)
        self.confidence = torch.cat(confidence)
        self.complexities = {}

    def images(self, indices, size):
        if isinstance(indices, torch.Tensor):
            indices = indices.cpu().tolist()
        return self.source.batch([self.indices[i] for i in indices], size).to(self.teacher.mean.device)

    @torch.no_grad()
    def complexity(self, size):
        if size not in self.complexities:
            parts = [patch_complexity(self.images(range(start, min(start+32, len(self.indices))), size))
                     for start in range(0, len(self.indices), 32)]
            self.complexities[size] = torch.cat(parts)
        return self.complexities[size]


def stage_sizes(config):
    values = ([200, 224, 200, 224] if config.size == 224 and config.blocks == 4
              else [config.size]*config.blocks)
    if config.ipc == 1 and config.cell_slots:
        return [size//config.grid_side for size in values]
    return values


def recover_class(source, real_indices, teacher, config, cls, output, log):
    output = Path(output)
    device = teacher.mean.device
    generator = torch.Generator().manual_seed(config.seed + cls*1009)
    elapsed_start, iteration = time.perf_counter(), 0
    pool = Pool(source, real_indices, teacher, config, cls)
    if len(real_indices) < max(config.slots, 2):
        raise ValueError(f"Class {cls} has fewer training patches than requested slots")
    n, sizes = config.slots, stage_sizes(config)
    if config.ipc == 1 and config.cell_slots:
        initial_ids = pool.confidence.argsort(descending=True, stable=True)[:n]
    else:
        initial_ids = torch.randperm(len(real_indices), generator=generator)[:n].to(device)
    # Parameters use the baseline's normalized pixel coordinates.
    pixels = teacher.normalize(pool.images(initial_ids, sizes[0]))
    groups = [torch.arange(start, min(start+config.effective_group, n), device=device)
              for start in range(0, n, config.effective_group)]
    topology = None
    if config.topology == "hfta" and config.hfta_weight > 0 and n > 1:
        topology = ClassTopology(pool.features, n, config, config.seed+cls)
    elif config.topology == "pta" and config.hfta_weight > 0 and n > 1:
        from .pta import ClassPTA
        topology = ClassPTA(pool.features, n, config, config.seed+cls)
    steps = [config.steps//config.blocks + (i < config.steps % config.blocks)
             for i in range(config.blocks)]
    reference_ids = torch.randperm(len(real_indices), generator=generator)[:min(n, config.cloud_cap)].to(device)
    diagnostics = {"class": cls, "real_features": pool.features[reference_ids].cpu(),
                   "checkpoints": [], "injections": [], "gradient_statistics": []}

    def checkpoint():
        z, _ = encode(teacher, teacher.denormalize(pixels).clamp(0, 1))
        ids = torch.randperm(n, generator=generator)[:min(n, config.cloud_cap)].to(device)
        diagnostics["checkpoints"].append(z[ids].cpu())
        return z

    checkpoint()
    for block, (size, budget) in enumerate(zip(sizes, steps)):
        pixels = F.interpolate(pixels, (size, size), mode="bilinear", align_corners=False)
        parameters = [torch.nn.Parameter(pixels[ids].detach().clone()) for ids in groups]
        optimizers = [torch.optim.Adam([x], lr=config.recovery_lr, betas=(.5, .9)) for x in parameters]

        def all_pixels():
            return torch.cat([p.detach() for p in parameters])

        for step in range(budget):
            topology_step = (sum(steps[:block]) + step) % config.topology_every == 0
            if topology is not None and topology_step:
                z, _ = encode(teacher, teacher.denormalize(all_pixels()).clamp(0, 1))
                topology.refresh(z)
            for group, (ids, parameter, optimizer) in enumerate(zip(groups, parameters, optimizers)):
                progress = (sum(steps[:block]) + step) / max(1, config.steps-1)
                optimizer.param_groups[0]["lr"] = config.recovery_lr*(1+math.cos(math.pi*progress))/2
                optimizer.zero_grad(set_to_none=True)
                raw = teacher.denormalize(parameter)
                augmented = raw
                if config.recovery_jitter:
                    offsets = torch.randint(-config.recovery_jitter, config.recovery_jitter+1,
                                            (2,), generator=generator).tolist()
                    augmented = torch.roll(augmented, offsets, (-2, -1))
                if torch.rand((), generator=generator) < .5:
                    augmented = augmented.flip(-1)
                labels = torch.full((len(ids),), cls, dtype=torch.long, device=device)
                recovery = teacher.recovery_loss(augmented, labels, config.bn_weight)
                geometric = raw.sum()*0
                if topology is not None and topology_step:
                    _, z = teacher(raw)
                    geometric = topology.loss(z, ids)
                    from .diagnostics import gradient_statistics
                    diagnostics["gradient_statistics"].append(
                        gradient_statistics(topology, z, ids, parameter))
                loss = recovery + config.hfta_weight * geometric
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite recovery loss in class {cls}")
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    parameter.copy_(teacher.normalize(teacher.denormalize(parameter).clamp(0, 1)))
                if iteration % config.log_every == 0:
                    log(stage="recover", cls=cls, block=block, group=group, step=step,
                        recovery=float(recovery.detach()), topology=float(geometric.detach()),
                        seconds=round(time.perf_counter()-elapsed_start, 3))
                iteration += 1
        pixels = all_pixels()
        if block < config.blocks-1 and config.assignment != "none":
            before, _ = encode(teacher, teacher.denormalize(pixels).clamp(0, 1))
            anchor_features, anchor_indices = [], []
            for ids in groups:
                if config.assignment == "static":
                    selected = initial_ids[ids]
                    anchors = pool.images(selected, size)
                    z_anchors = pool.features[selected]
                else:
                    result = assign(before[ids], pool.features, pool.complexity(size), config)
                    selected = result.indices
                    if config.assignment == "balanced":
                        candidates = pool.images(result.candidates, size)
                        anchors = (result.weights.to(candidates) @ candidates.flatten(1)).reshape(len(ids), 3, size, size)
                        z_anchors, _ = encode(teacher, anchors)
                    else:
                        anchors = pool.images(selected, size)
                        z_anchors = pool.features[selected]
                pixels[ids] = config.alpha*pixels[ids] + (1-config.alpha)*teacher.normalize(anchors)
                anchor_features.append(z_anchors.cpu())
                anchor_indices.append(selected.cpu())
            diagnostics["injections"].append({"before": before.cpu(),
                "anchors": torch.cat(anchor_features), "indices": torch.cat(anchor_indices)})
        checkpoint()
    final_size = config.size//config.grid_side if config.ipc == 1 and config.cell_slots else config.size
    pixels = F.interpolate(pixels, (final_size, final_size), mode="bilinear", align_corners=False)
    images = teacher.denormalize(pixels).clamp(0, 1).cpu()
    if config.ipc == 1 and config.cell_slots:
        images = pack_cells(images, config.grid_side).unsqueeze(0)
    folder = output / "synthetic" / f"{cls:05d}"
    folder.mkdir(parents=True, exist_ok=True)
    for index, image in enumerate(images):
        TF.to_pil_image(image).save(folder / f"image_{index:05d}.png")
    save_tensor(output/"diagnostics"/f"class_{cls:05d}.pt", diagnostics)
    save_json(folder/"complete.json", {"images": len(images), "slots": n,
                                       "steps_per_slot": config.steps,
                                       "seconds": time.perf_counter()-elapsed_start})
    return images, diagnostics
