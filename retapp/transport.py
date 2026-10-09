"""Structure Transport Connection: slot capacity, Gromov cost, exact anchors."""

from dataclasses import dataclass
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment, minimize
from scipy.special import logsumexp
import numpy as np


def squared_distances(x):
    return (x.square().sum(1)[:, None] + x.square().sum(1)[None, :]
            - 2 * x @ x.T).clamp_min(0)


def minmax(x):
    span = x.max() - x.min()
    return (x - x.min()) / span.clamp_min(torch.finfo(x.dtype).eps)


def structure_cost(ds, dr, coupling):
    # Expansion of sum_bj (ds_ab - dr_ij)^2 * coupling_bj.
    return (ds.square() @ coupling.sum(1)[:, None]
            + (dr.square() @ coupling.sum(0))[None, :]
            - 2 * ds @ coupling @ dr.T).clamp_min(0)


def sinkhorn(cost, rows, columns, entropy=0.05, iterations=1000, tolerance=1e-7):
    if entropy <= 0 or (rows <= 0).any() or (columns <= 0).any():
        raise ValueError("Sinkhorn requires positive entropy and marginals")
    if not torch.isclose(rows.sum(), columns.sum()):
        raise ValueError("Marginal masses differ")
    kernel = -cost.double() / entropy
    r, c = rows.double(), columns.double()
    u, v = torch.zeros_like(r), torch.zeros_like(c)
    for step in range(iterations):
        u = r.log() - torch.logsumexp(kernel + v[None, :], 1)
        v = c.log() - torch.logsumexp(kernel + u[:, None], 0)
        if step % 20 == 19 or step == iterations - 1:
            plan = (kernel + u[:, None] + v[None, :]).exp()
            error = max((plan.sum(1) - r).abs().max().item(),
                        (plan.sum(0) - c).abs().max().item())
            if error <= tolerance:
                return plan.to(cost.dtype)
    # Near-permutation plans can converge slowly under alternating scaling.
    # Refine the same entropic dual, fixing its additive gauge.
    values, row_mass, col_mass = kernel.cpu().numpy(), r.cpu().numpy(), c.cpu().numpy()

    def dual(free):
        potential = np.r_[free, 0.]
        logits = values + potential[None, :]
        partition = logsumexp(logits, axis=1)
        p = row_mass[:, None]*np.exp(logits-partition[:, None])
        return float(row_mass @ partition-col_mass @ potential), (p.sum(0)-col_mass)[:-1]

    start = (v-v[-1]).cpu().numpy()[:-1]
    solution = minimize(dual, start, jac=True, method="L-BFGS-B",
                        options={"maxiter": iterations, "ftol": 1e-15, "gtol": tolerance*.01,
                                 "maxls": 50, "maxcor": 20})
    v = torch.as_tensor(np.r_[solution.x, 0.], device=cost.device, dtype=torch.double)
    plan = r[:, None]*torch.softmax(kernel+v[None, :], dim=1)
    error = (plan.sum(0)-c).abs().max().item()
    if error > max(tolerance * 10, 1e-6):
        raise RuntimeError(f"Transport marginal error {error:.3g}; increase iterations")
    return plan.to(cost.dtype)


@dataclass
class Assignment:
    indices: torch.Tensor
    candidates: torch.Tensor
    cost: torch.Tensor
    coupling: torch.Tensor
    full_coupling: torch.Tensor
    weights: torch.Tensor


@torch.no_grad()
def assign(z_slots, z_pool, complexity, config):
    z_slots, z_pool = F.normalize(z_slots.double(), dim=1), F.normalize(z_pool.double(), dim=1)
    k, available = len(z_slots), len(z_pool)
    if available < k:
        raise ValueError("The real class pool must contain at least one patch per slot")
    fit_all = 1 - z_slots @ z_pool.T
    selected = fit_all.mean(0).argsort(stable=True)[:min(config.candidates, available)]
    fit = minmax(fit_all[:, selected])
    comp = minmax(complexity[selected].double())[None, :].expand_as(fit)
    base = fit + config.complexity_weight * comp
    ds, dr = squared_distances(z_slots), squared_distances(z_pool[selected])
    m = len(selected)
    balanced = config.assignment == "balanced"
    dummy = not balanced and m > k
    rows = base.new_full((k,), 1 / (k if balanced else m))
    if dummy:
        rows = torch.cat((rows.new_tensor([(m-k)/m]), rows))
    columns = base.new_full((m,), 1/m)

    def solve(cost):
        padded = torch.cat((cost.new_zeros(1, m), cost)) if dummy else cost
        return sinkhorn(padded, rows, columns, config.entropy,
                        config.sinkhorn_iterations, config.sinkhorn_tolerance)

    if config.assignment == "independent":
        local = base.argmin(1)
        weights = F.one_hot(local, m).to(base)
        return Assignment(selected[local], selected, base, weights/k, weights/k, weights)
    full = solve(base)
    for _ in range(config.transport_updates):
        plan = full[1:] if dummy else full
        cost = base + config.structure_weight * minmax(structure_cost(ds, dr, plan))
        full = solve(cost)
    plan = full[1:] if dummy else full
    # Round C(Pi^R), not a stale cost or -log(Pi^R).
    cost = base + config.structure_weight * minmax(structure_cost(ds, dr, plan))
    row_index, col_index = linear_sum_assignment(cost.cpu().numpy())
    local = torch.empty(k, dtype=torch.long, device=cost.device)
    local[torch.as_tensor(row_index, device=cost.device)] = torch.as_tensor(col_index, device=cost.device)
    weights = plan / plan.sum(1, keepdim=True)
    return Assignment(selected[local], selected, cost, plan, full, weights)


def patch_complexity(images):
    """Variance over pixel locations of squared, smoothed gradient norm."""
    channels = images.shape[1]
    kernel = images.new_tensor([1., 2., 1.]) / 4
    blur = (kernel[:, None] * kernel[None, :])[None, None].expand(channels, 1, 3, 3)
    smooth = F.conv2d(F.pad(images, (1, 1, 1, 1), mode="replicate"), blur, groups=channels)
    dx = (smooth[:, :, 1:-1, 2:] - smooth[:, :, 1:-1, :-2]) / 2
    dy = (smooth[:, :, 2:, 1:-1] - smooth[:, :, :-2, 1:-1]) / 2
    energy = (dx.square() + dy.square()).sum(1)
    return energy.flatten(1).var(1, unbiased=False)
