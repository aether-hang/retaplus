"""Weighted graph/Hodge spectra and the corrected HFTA Betti responses."""

from functools import lru_cache
from itertools import combinations
import torch
from .transport import squared_distances


@lru_cache(maxsize=16)
def incidence(n):
    edges = list(combinations(range(n), 2))
    triangles = list(combinations(range(n), 3))
    index = {edge: k for k, edge in enumerate(edges)}
    b1 = torch.zeros(n, len(edges), dtype=torch.float64)
    b2 = torch.zeros(len(edges), len(triangles), dtype=torch.float64)
    tri_edges = []
    for k, (i, j) in enumerate(edges):
        b1[i, k], b1[j, k] = -1, 1
    for k, (i, j, h) in enumerate(triangles):
        ids = [index[i, j], index[i, h], index[j, h]]
        b2[ids, k] = torch.tensor([1., -1., 1.], dtype=b2.dtype)
        tri_edges.append(ids)
    return (b1, b2, torch.tensor(edges, dtype=torch.long).reshape(-1, 2),
            torch.tensor(tri_edges, dtype=torch.long).reshape(-1, 3))


def operators(z, scale, tau, hard=False, distances=None):
    b1, b2, edges, tri_edges = (t.to(z.device) for t in incidence(len(z)))
    b1, b2 = b1.to(z.dtype), b2.to(z.dtype)
    delta = ((z[edges[:, 0]] - z[edges[:, 1]]).square().sum(1) if distances is None
             else distances[edges[:, 0], edges[:, 1]])
    if hard:
        weights = (delta < scale).to(z)
        root = weights
        triangle = weights[tri_edges].prod(1)
    else:
        log_weights = torch.nn.functional.logsigmoid((scale-delta) / tau)
        weights = log_weights.exp()
        root = (0.5 * log_weights).exp()
        triangle = (log_weights[tri_edges].mean(1)).exp()
    weighted_b1 = b1 * root[None, :]
    # B2 diag(w_triangle) B2^T avoids a sqrt derivative at zero triangle mass.
    l0 = weighted_b1 @ weighted_b1.T
    l1 = weighted_b1.T @ weighted_b1 + (b2 * triangle[None, :]) @ b2.T
    return l0, l1, weights


class _HeatTrace(torch.autograd.Function):
    @staticmethod
    def forward(ctx, matrix, rho):
        eigenvalues, vectors = torch.linalg.eigh(matrix)
        response = torch.exp(-eigenvalues / rho)
        ctx.save_for_backward(vectors, response, rho)
        return response.sum()

    @staticmethod
    def backward(ctx, gradient):
        vectors, response, rho = ctx.saved_tensors
        # The trace derivative is well-defined at repeated eigenvalues.
        derivative = (vectors * (-response/rho)[None, :]) @ vectors.T
        return gradient * derivative, None


def response(z, scales, tau, rhos, hard=False, distances=None):
    z = z.double()
    b0, b1 = [], []
    for s, scale in enumerate(scales):
        l0, l1, weights = operators(z, scale, tau, hard, distances)
        b0.append(_HeatTrace.apply(l0, rhos[s, 0]))
        b1.append(_HeatTrace.apply(l1, rhos[s, 1]) - (1-weights).sum())
    return torch.cat((torch.stack(b0), torch.stack(b1)))


@torch.no_grad()
def calibrate_reference(real, n, config, generator):
    """One fixed class reference: equal-cardinality real clouds, common scales."""
    real = real.double()
    if n < 2 or n > len(real):
        raise ValueError("Reference cloud must have between 2 and pool-size points")
    distances = torch.pdist(real).square()
    quantiles = torch.linspace(0.1, 0.8, config.hfta_scales, device=real.device, dtype=real.dtype)
    scales = torch.quantile(distances, quantiles)
    tau = (config.tau_fraction * scales.diff().mean()).clamp_min(1e-8)
    eig0, eig1, missing = [], [], []
    for _ in range(config.reference_samples):
        ids = torch.randperm(len(real), generator=generator)[:n].to(real.device)
        zero, one, absent = [], [], []
        for scale in scales:
            l0, l1, weights = operators(real[ids], scale, tau)
            zero.append(torch.linalg.eigvalsh(l0))
            one.append(torch.linalg.eigvalsh(l1))
            absent.append((1-weights).sum())
        eig0.append(torch.stack(zero))
        eig1.append(torch.stack(one))
        missing.append(torch.stack(absent))
    eig0, eig1, missing = torch.stack(eig0), torch.stack(eig1), torch.stack(missing)
    rhos = real.new_empty(config.hfta_scales, 2)
    for s in range(config.hfta_scales):
        for q, spectrum in enumerate((eig0[:, s], eig1[:, s])):
            threshold = max(1e-10, spectrum.abs().max().item() * 1e-10)
            positive = spectrum[spectrum > threshold]
            scale = positive.mean() if positive.numel() else spectrum.new_tensor(1.)
            rhos[s, q] = (config.rho_fraction * scale).clamp_min(1e-8)
    zero = torch.exp(-eig0/rhos[None, :, 0, None]).sum(-1).mean(0)
    one = (torch.exp(-eig1/rhos[None, :, 1, None]).sum(-1) - missing).mean(0)
    return {"scales": scales, "tau": tau, "rhos": rhos,
            "target": torch.cat((zero, one)), "cardinality": n}


class ClassTopology:
    """Inactive slots are detached; the current active group always has gradients."""
    def __init__(self, real_features, slot_count, config, seed):
        self.config = config
        self.generator = torch.Generator().manual_seed(seed)
        self.n = min(slot_count, config.cloud_cap)
        self.reference = calibrate_reference(real_features, self.n, config, self.generator)
        self.cache = None

    def refresh(self, features):
        self.cache = features.detach().clone()

    def cloud(self, active_features, active_indices):
        if self.cache is None:
            raise RuntimeError("Refresh the class cache before evaluating topology")
        if len(active_indices) > self.n:
            raise ValueError("The class cap cannot exclude active slots")
        mask = torch.ones(len(self.cache), dtype=torch.bool, device=self.cache.device)
        mask[active_indices] = False
        inactive = mask.nonzero().flatten()
        order = torch.randperm(len(inactive), generator=self.generator)[:self.n-len(active_indices)]
        chosen = inactive[order.to(inactive.device)]
        return torch.cat((active_features, self.cache[chosen]), 0)

    def loss(self, active_features, active_indices):
        z = self.cloud(active_features, active_indices)
        r = self.reference
        self.last_distances = squared_distances(z.double())
        self.last_loss = (response(z, r["scales"], r["tau"], r["rhos"],
                                  distances=self.last_distances) - r["target"]).square().sum()
        return self.last_loss
