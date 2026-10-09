import pytest
import torch
from retapp.config import Config
from retapp.diagnostics import hard_betti
from retapp.topology import ClassTopology, _HeatTrace, incidence, operators, response


def test_boundary_orientation():
    b1, b2, _, _ = incidence(5)
    assert (b1 @ b2).eq(0).all()


@pytest.mark.parametrize("scale,expected", [(.5, [4., 0.]), (1.5, [1., 1.]), (2.5, [1., 0.])])
def test_corrected_hard_betti_empty_cycle_filled(scale, expected):
    z = torch.tensor([[0., 0.], [1., 0.], [1., 1.], [0., 1.]], dtype=torch.float64)
    result = hard_betti(z, [scale])[0]
    torch.testing.assert_close(result, torch.tensor(expected, dtype=torch.float64))
    smooth = response(z, torch.tensor([scale]), torch.tensor(.0001), torch.full((1, 2), .001))
    torch.testing.assert_close(smooth, result, atol=1e-7, rtol=0)


def test_heat_trace_gradient_at_repeated_eigenvalues():
    matrix = torch.eye(3, dtype=torch.double, requires_grad=True)
    rho = torch.tensor(.4, dtype=torch.double)
    _HeatTrace.apply(matrix, rho).backward()
    expected = -torch.exp(torch.tensor(-2.5, dtype=torch.double))/.4*torch.eye(3, dtype=torch.double)
    torch.testing.assert_close(matrix.grad, expected)


def test_heat_trace_gradient_matches_matrix_exponential():
    a = torch.randn(4, 4, dtype=torch.double)
    a = (a @ a.T).requires_grad_()
    rho = torch.tensor(.8, dtype=torch.double)
    x = torch.autograd.grad(_HeatTrace.apply(a, rho), a)[0]
    y = torch.autograd.grad(torch.matrix_exp(-a/rho).trace(), a)[0]
    torch.testing.assert_close(x, y)


def test_reference_and_active_cache_gradients():
    config = Config(hfta_scales=3, reference_samples=2, cloud_cap=4)
    real = torch.nn.functional.normalize(torch.randn(8, 5), dim=1)
    cache = torch.randn(7, 5, requires_grad=True)
    active = torch.randn(2, 5, requires_grad=True)
    topology = ClassTopology(real, 7, config, 42)
    topology.refresh(cache)
    cloud = topology.cloud(active, torch.tensor([1, 6]))
    assert cloud.shape == (4, 5)
    torch.testing.assert_close(cloud[:2], active)
    assert not topology.cache.requires_grad
    loss = topology.loss(active, torch.tensor([1, 6]))
    loss.backward()
    assert torch.isfinite(active.grad).all() and active.grad.abs().sum() > 0
    assert cache.grad is None
    assert topology.reference["cardinality"] == 4
    assert topology.reference["rhos"].shape == (3, 2)


def test_response_pixel_distance_gradient():
    z = torch.randn(4, 3, dtype=torch.double, requires_grad=True)
    scales = torch.tensor([.4, 1.5], dtype=torch.double)
    rhos = torch.full((2, 2), .2, dtype=torch.double)
    assert torch.autograd.gradcheck(lambda x: response(x, scales, torch.tensor(.3), rhos), (z,), atol=2e-4)
