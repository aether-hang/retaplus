"""Measurements used by the anchor-sharing and topology-gradient diagnostics."""

import torch
from .topology import operators


def gradient_statistics(topology, active_features, active_indices, pixels):
    distances = topology.last_distances
    pair_grad, image_grad = torch.autograd.grad(topology.last_loss, (distances, pixels),
                                               retain_graph=True, allow_unused=True)
    pair_grad = torch.zeros_like(distances) if pair_grad is None else pair_grad
    pair_grad = (pair_grad + pair_grad.T).abs()
    i, j = torch.triu_indices(len(distances), len(distances), 1, device=distances.device)
    eligible = (i < len(active_indices)) | (j < len(active_indices))
    values = pair_grad[i[eligible], j[eligible]]
    maximum = values.max() if values.numel() else values.new_tensor(0.)
    coverage = 100*float((values > 1e-6*maximum).float().mean()) if maximum > 0 else 0.
    norms = torch.zeros(len(pixels), device=pixels.device) if image_grad is None else image_grad.flatten(1).norm(dim=1)
    weak = 100*float((norms < .1*norms.max()).float().mean()) if norms.max() > 0 else 100.
    return {"coverage": coverage, "weak_images": weak}


def mean_distance(z):
    return torch.pdist(z).mean() if len(z) > 1 else z.new_tensor(0.)


def shared_fraction(anchors, indices):
    if len(anchors) < 2:
        return 0.
    i, j = torch.triu_indices(len(anchors), len(anchors), 1)
    shared = ((anchors[i]*anchors[j]).sum(1) > .95) | (indices[i] == indices[j])
    return 100*float(shared.float().mean())


def alignment(before, anchors):
    i, j = torch.triu_indices(len(before), len(before), 1)
    x, y = before[i]-before[j], anchors[i]-anchors[j]
    norms = x.norm(dim=1)*y.norm(dim=1)
    mask = norms > 1e-12
    return ((x[mask]*y[mask]).sum(1)/norms[mask]).clamp(-1, 1)


def hard_betti(z, scales):
    result = []
    for scale in scales:
        l0, l1, weights = operators(z.double(), scale, z.new_tensor(1.), hard=True)
        zero = (torch.linalg.eigvalsh(l0).abs() < 1e-7).sum()
        one = (torch.linalg.eigvalsh(l1).abs() < 1e-7).sum() - (1-weights).sum()
        result.append(torch.stack((zero, one)))
    return torch.stack(result).double()


def betti_gap(real, synthetic):
    # Betti curves are step functions; integrate exactly over all change scales.
    distances = torch.cat((torch.pdist(real).square(), torch.pdist(synthetic).square()))
    boundaries = torch.cat((distances.new_zeros(1), distances)).unique(sorted=True)
    if len(boundaries) < 2:
        return 0.
    midpoints = (boundaries[:-1]+boundaries[1:])/2
    gap = (hard_betti(real, midpoints)-hard_betti(synthetic, midpoints)).clamp_min(0)
    return float((gap.sum(1)*boundaries.diff()).sum())


def summarize(diagnostic):
    real = diagnostic["real_features"]
    denominator = mean_distance(real).clamp_min(1e-12)
    spreads = [float(mean_distance(z)/denominator) for z in diagnostic["checkpoints"]]
    injections = diagnostic["injections"]
    angles = [alignment(x["before"], x["anchors"]) for x in injections]
    angles = torch.cat(angles) if angles else torch.empty(0)
    statistics = diagnostic["gradient_statistics"]
    return {"class": diagnostic["class"], "spread_curve": spreads,
            "shared": sum(shared_fraction(x["anchors"], x["indices"]) for x in injections)/max(1, len(injections)),
            "cosine_median": float(angles.median()) if len(angles) else None,
            "betti_gap": betti_gap(real, diagnostic["checkpoints"][-1]),
            "coverage": sum(s["coverage"] for s in statistics)/len(statistics) if statistics else None,
            "weak_images": sum(s["weak_images"] for s in statistics)/len(statistics) if statistics else None}
