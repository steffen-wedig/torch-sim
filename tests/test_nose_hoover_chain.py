"""The Nose-Hoover chain half-step: the pure function against the original per-substep
algorithm, and compiled against eager."""

import pytest
import torch

from torch_sim.integrators.md import (
    SUZUKI_YOSHIDA_WEIGHTS,
    NoseHooverChain,
    construct_nose_hoover_chain,
)


DEVICES = [torch.device("cpu")] + (
    [torch.device("cuda")] if torch.cuda.is_available() else []
)


def _reference_half_step(
    momenta: torch.Tensor,
    chain: NoseHooverChain,
    kT: torch.Tensor,
    system_idx: torch.Tensor,
    dt: torch.Tensor,
    chain_steps: int,
    sy_steps: int,
) -> tuple[torch.Tensor, NoseHooverChain]:
    """The torch-sim v0.6.2 half-step: in-place column updates, momenta rescaled in
    every substep."""
    xi, p_xi = chain.positions.clone(), chain.momenta.clone()
    Q, KE, DOF = chain.masses, chain.kinetic_energy, chain.degrees_of_freedom
    M = p_xi.shape[-1] - 1
    weights = SUZUKI_YOSHIDA_WEIGHTS[sy_steps].tolist()
    delta = dt / chain_steps
    for step in range(chain_steps * sy_steps):
        delta_2 = delta * weights[step % sy_steps] / 2.0
        delta_4 = delta_2 / 2.0
        delta_8 = delta_4 / 2.0
        if M > 0:
            G = torch.square(p_xi[:, M - 1]) / Q[:, M - 1] - kT
            p_xi[:, M] += delta_4 * G
        for m in range(M - 1, 0, -1):
            G = torch.square(p_xi[:, m - 1]) / Q[:, m - 1] - kT
            scale = torch.exp(-delta_8 * p_xi[:, m + 1] / Q[:, m + 1])
            p_xi[:, m] = scale * (scale * p_xi[:, m] + delta_4 * G)
        G = 2.0 * KE - DOF * kT
        scale = torch.exp(-delta_8 * p_xi[:, 1] / Q[:, 1]) if M > 0 else 1.0
        p_xi[:, 0] = scale * (scale * p_xi[:, 0] + delta_4 * G)
        scale = torch.exp(-delta_2 * p_xi[:, 0] / Q[:, 0])
        KE = KE * torch.square(scale)
        momenta = momenta * scale[system_idx].reshape(-1, *([1] * (momenta.ndim - 1)))
        xi = xi + delta_2 * p_xi / Q
        G = 2.0 * KE - DOF * kT
        for m in range(M):
            scale = torch.exp(-delta_8 * p_xi[:, m + 1] / Q[:, m + 1])
            p_xi[:, m] = scale * (scale * p_xi[:, m] + delta_4 * G)
            G = torch.square(p_xi[:, m]) / Q[:, m] - kT
        p_xi[:, M] += delta_4 * G
    return momenta, NoseHooverChain(xi, p_xi, Q, chain.tau, KE, DOF)


def _chain_inputs(
    chain_length: int, coupled: str, device: torch.device, seed: int = 3
) -> tuple[torch.Tensor, NoseHooverChain, torch.Tensor, torch.Tensor]:
    """Momenta, a chain with non-trivial state, ``kT`` and the system index of three
    systems: atom momenta ``[9, 3]`` (thermostat) or cell momenta ``[3, 1]``
    (barostat)."""
    generator = torch.Generator().manual_seed(seed)
    systems = 3
    if coupled == "atoms":
        system_idx = torch.tensor([0, 0, 0, 1, 1, 1, 1, 2, 2])
        momenta = torch.randn(9, 3, generator=generator, dtype=torch.float64)
    else:
        system_idx = torch.arange(systems)
        momenta = torch.randn(systems, 1, generator=generator, dtype=torch.float64)
    kT = torch.tensor([0.025, 0.03, 0.04], dtype=torch.float64)
    masses = 0.5 + torch.rand(
        systems, chain_length, generator=generator, dtype=torch.float64
    )
    chain = NoseHooverChain(
        positions=torch.randn(
            systems, chain_length, generator=generator, dtype=torch.float64
        ),
        momenta=0.1
        * torch.randn(systems, chain_length, generator=generator, dtype=torch.float64),
        masses=masses,
        tau=torch.full((systems,), 2.0, dtype=torch.float64),
        kinetic_energy=0.5
        + torch.rand(systems, generator=generator, dtype=torch.float64),
        degrees_of_freedom=torch.tensor([6.0, 9.0, 3.0], dtype=torch.float64),
    )
    tensors = (momenta, kT, system_idx)
    momenta, kT, system_idx = (tensor.to(device) for tensor in tensors)
    chain = NoseHooverChain(
        *(
            getattr(chain, name).to(device)
            for name in (
                "positions",
                "momenta",
                "masses",
                "tau",
                "kinetic_energy",
                "degrees_of_freedom",
            )
        )
    )
    return momenta, chain, kT, system_idx


@pytest.mark.parametrize("device", DEVICES, ids=str)
@pytest.mark.parametrize("coupled", ["atoms", "cell"])
@pytest.mark.parametrize("chain_length", [1, 2, 3, 5])
@pytest.mark.parametrize(("chain_steps", "sy_steps"), [(1, 1), (2, 3), (1, 5), (3, 7)])
def test_half_step_equals_the_per_substep_algorithm(
    device: torch.device, coupled: str, chain_length: int, chain_steps: int, sy_steps: int
) -> None:
    """Chain variables and kinetic energy bitwise; the momenta to rounding (one
    accumulated scale instead of one rescale per substep). The input chain is not
    mutated."""
    momenta, chain, kT, system_idx = _chain_inputs(chain_length, coupled, device)
    chain_momenta_before = chain.momenta.clone()
    dt = torch.tensor(0.4, dtype=torch.float64, device=device)
    functions = construct_nose_hoover_chain(dt, chain_length, chain_steps, sy_steps, 1.0)
    actual_momenta, actual = functions.half_step(momenta, chain, kT, system_idx)
    expected_momenta, expected = _reference_half_step(
        momenta, chain, kT, system_idx, dt, chain_steps, sy_steps
    )
    for name in ("positions", "momenta", "kinetic_energy"):
        torch.testing.assert_close(
            getattr(actual, name), getattr(expected, name), rtol=0, atol=0
        )
    torch.testing.assert_close(actual_momenta, expected_momenta, rtol=1e-14, atol=0)
    assert actual_momenta.shape == momenta.shape
    torch.testing.assert_close(chain.momenta, chain_momenta_before, rtol=0, atol=0)


def test_suzuki_yoshida_weights_are_float64_and_sum_to_one() -> None:
    for sy_steps, weights in SUZUKI_YOSHIDA_WEIGHTS.items():
        assert weights.dtype == torch.float64
        assert len(weights) == sy_steps
        assert abs(float(weights.sum()) - 1.0) < 1e-13


@pytest.mark.parametrize("device", DEVICES, ids=str)
def test_compiled_half_step_equals_eager_without_recompiling(
    device: torch.device,
) -> None:
    """The compiled half-step (``fullgraph``: no graph breaks) matches eager over
    repeated calls with changing values, which do not recompile."""
    torch._dynamo.reset()  # noqa: SLF001
    dt = torch.tensor(0.4, dtype=torch.float64, device=device)
    eager = construct_nose_hoover_chain(dt, 3, 2, 3, 1.0)
    compiled = construct_nose_hoover_chain(dt, 3, 2, 3, 1.0, compiled=True)
    momenta, chain, kT, system_idx = _chain_inputs(3, "atoms", device)
    compiled_momenta, compiled_chain = compiled.half_step(momenta, chain, kT, system_idx)
    eager_momenta, eager_chain = eager.half_step(momenta, chain, kT, system_idx)
    with torch._dynamo.config.patch(error_on_recompile=True):  # noqa: SLF001
        for _ in range(5):
            torch.testing.assert_close(
                compiled_momenta, eager_momenta, rtol=1e-12, atol=0
            )
            for name in ("positions", "momenta", "kinetic_energy"):
                torch.testing.assert_close(
                    getattr(compiled_chain, name),
                    getattr(eager_chain, name),
                    rtol=1e-12,
                    atol=1e-15,
                )
            compiled_momenta, compiled_chain = compiled.half_step(
                compiled_momenta, compiled_chain, kT * 1.01, system_idx
            )
            eager_momenta, eager_chain = eager.half_step(
                eager_momenta, eager_chain, kT * 1.01, system_idx
            )
