from dataclasses import replace
import pytest
import torch
from scipy.optimize import linear_sum_assignment
from retapp.config import Config
from retapp.transport import assign, minmax, patch_complexity, squared_distances, structure_cost


@pytest.mark.parametrize("k,m", [(1, 8), (4, 8), (4, 4)])
def test_capacity_marginals_and_distinct_assignment(k, m):
    config = Config(candidates=m)
    z, real = torch.randn(k, 7), torch.randn(12, 7)
    complexity = torch.rand(12)
    result = assign(z, real, complexity, config)
    assert len(result.indices.unique()) == k
    expected = torch.full((k,), 1/m, dtype=torch.float64)
    assert torch.allclose(result.coupling.sum(1), expected, atol=2e-6)
    assert torch.allclose(result.full_coupling.sum(0), torch.full((m,), 1/m, dtype=torch.float64), atol=2e-6)
    if m > k:
        assert result.full_coupling[0].sum().item() == pytest.approx((m-k)/m, abs=2e-6)
    a, b = linear_sum_assignment(result.cost.numpy())
    assert torch.equal(result.indices, result.candidates[torch.tensor(b)])
    zs = torch.nn.functional.normalize(z.double(), dim=1)
    zr = torch.nn.functional.normalize(real.double(), dim=1)[result.candidates]
    structural = minmax(structure_cost(squared_distances(zs), squared_distances(zr), result.coupling))
    expected_cost = minmax(1-zs @ zr.T) + config.complexity_weight*minmax(complexity[result.candidates].double())[None, :]
    torch.testing.assert_close(result.cost, expected_cost+config.structure_weight*structural)


def test_structure_expansion():
    ds, dr, pi = squared_distances(torch.randn(4, 3)), squared_distances(torch.randn(6, 3)), torch.rand(4, 6)
    direct = ((ds[:, None, :, None]-dr[None, :, None, :])**2*pi[None, None]).sum((2, 3))
    torch.testing.assert_close(structure_cost(ds, dr, pi), direct, atol=2e-4, rtol=2e-5)


def test_identical_queries_still_receive_distinct_patches():
    z, real, complexity = torch.ones(4, 3), torch.randn(12, 3), torch.rand(12)
    config = Config(candidates=8)
    assert len(assign(z, real, complexity, config).indices.unique()) == 4
    assert len(assign(z, real, complexity, replace(config, assignment="independent")).indices.unique()) == 1


def test_balanced_barycentric_ablation():
    result = assign(torch.randn(4, 3), torch.randn(12, 3), torch.rand(12), Config(assignment="balanced", candidates=8))
    torch.testing.assert_close(result.coupling.sum(1), torch.full((4,), .25, dtype=torch.float64), atol=2e-6, rtol=0)
    torch.testing.assert_close(result.weights.sum(1), torch.ones(4, dtype=torch.float64))
    assert (result.weights > 0).all()


def test_constant_normalization_and_complexity():
    assert minmax(torch.ones(2, 3)).eq(0).all()
    assert patch_complexity(torch.ones(2, 3, 16, 16)).eq(0).all()
    assert (patch_complexity(torch.rand(2, 3, 16, 16)) > 0).all()


def test_capacity_leaves_complexity_effective():
    z, real = torch.tensor([[1., 0.]]), torch.tensor([[1., 0.]]).repeat(3, 1)
    complexity = torch.tensor([1., .5, 0.])
    config = Config(candidates=3, structure_weight=0, complexity_weight=0)
    assert assign(z, real, complexity, config).indices.item() == 0
    assert assign(z, real, complexity, replace(config, complexity_weight=.1)).indices.item() == 2
