"""Persistent Topology Alignment for the conference RETA component ablations."""

import torch
from .topology import ClassTopology
from .transport import squared_distances


def persistence_image(distance, cutoff, bins=16, neighbors=5):
    import gudhi
    n = len(distance)
    graph = distance.detach().cpu()
    nearest = torch.zeros_like(graph, dtype=torch.bool)
    graph = graph.clone()
    graph.fill_diagonal_(float("inf"))
    ids = graph.argsort(1)[:, :min(neighbors, n-1)]
    nearest.scatter_(1, ids, True)
    mutual = nearest & nearest.T
    tree = gudhi.SimplexTree()
    for i in range(n):
        tree.insert([i], filtration=0)
    for i in range(n):
        for j in range(i+1, n):
            if mutual[i, j] and float(graph[i, j]) <= cutoff:
                tree.insert([i, j], filtration=float(graph[i, j]))
    tree.expansion(2)
    tree.persistence(homology_coeff_field=2, persistence_dim_max=True)
    axes = torch.linspace(0, 1, bins, device=distance.device, dtype=distance.dtype)
    grid = torch.stack(torch.meshgrid(axes, axes, indexing="ij"), -1)
    result = [distance.sum()*0 + distance.new_zeros(bins, bins) for _ in range(2)]

    def value(simplex):
        if len(simplex) < 2:
            return distance.new_tensor(0.)
        index = torch.tensor(simplex, device=distance.device)
        return distance[index[:, None], index[None, :]].max()

    for birth, death in tree.persistence_pairs():
        q = len(birth)-1
        if q not in (0, 1):
            continue
        b = value(birth)/cutoff
        d = value(death)/cutoff if death else distance.new_tensor(1.)
        lifetime = (d-b).clamp_min(0)
        center = torch.stack((b, lifetime))
        result[q] = result[q] + lifetime * torch.exp(-(grid-center).square().sum(-1)/(2*(1/bins)**2))
    return torch.cat([image.flatten() for image in result])


class ClassPTA(ClassTopology):
    def __init__(self, real_features, slot_count, config, seed):
        self.config = config
        self.generator = torch.Generator().manual_seed(seed)
        self.n = min(slot_count, config.cloud_cap)
        self.cutoff = max(float(torch.pdist(real_features).square().max()), 1e-8)
        images = []
        with torch.no_grad():
            for _ in range(config.reference_samples):
                index = torch.randperm(len(real_features), generator=self.generator)[:self.n].to(real_features.device)
                images.append(persistence_image(squared_distances(real_features[index]), self.cutoff))
        self.target = torch.stack(images).mean(0)
        self.cache = None

    def loss(self, active_features, active_indices):
        cloud = self.cloud(active_features, active_indices)
        self.last_distances = squared_distances(cloud)
        self.last_loss = (persistence_image(self.last_distances, self.cutoff)-self.target).square().sum()
        return self.last_loss
