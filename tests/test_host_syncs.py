"""Per-step quantities of the MD integrators without host-device synchronisation.

A batched MD step on a GPU is only as fast as the host can queue it: every read of a GPU
value back to the host (``.any()``, ``.item()``, ``bincount``, ``unique``, boolean-mask
indexing) drains the queue. These tests pin the per-system reductions the NPT and NVT
integrators run every step as sync-free on CUDA, and their values against the reductions
they replaced.
"""

import pytest
import torch

import torch_sim as ts
from torch_sim.integrators.npt import _npt_nose_hoover_isotropic_compute_cell_force
from torch_sim.state import SimState


def _batch(device: torch.device) -> SimState:
    generator = torch.Generator().manual_seed(3)
    atoms_per_system = [3, 5, 2, 4]
    system_idx = torch.repeat_interleave(
        torch.arange(len(atoms_per_system)), torch.tensor(atoms_per_system)
    )
    atoms = system_idx.shape[0]
    state = SimState(
        positions=torch.rand(atoms, 3, generator=generator, dtype=torch.float64) * 5,
        masses=torch.rand(atoms, generator=generator, dtype=torch.float64) * 30 + 1,
        cell=torch.eye(3, dtype=torch.float64).repeat(len(atoms_per_system), 1, 1) * 5,
        pbc=True,
        atomic_numbers=torch.ones(atoms, dtype=torch.int64),
        system_idx=system_idx,
    )
    return state.to(device)


def _loop_kinetic_energy(
    masses: torch.Tensor, momenta: torch.Tensor, system_idx: torch.Tensor, n_systems: int
) -> torch.Tensor:
    """The kinetic energy per system as ``_npt_nose_hoover_isotropic_compute_cell_force``
    computed it before (a Python loop of masked selections)."""
    kinetic = torch.zeros(n_systems, dtype=momenta.dtype, device=momenta.device)
    for system in range(n_systems):
        mask = system_idx == system
        kinetic[system] = 0.5 * (momenta[mask].square() / masses[mask, None]).sum()
    return kinetic


def test_per_system_quantities_equal_the_reductions_they_replace() -> None:
    state = _batch(torch.device("cpu"))
    momenta = torch.randn(state.n_atoms, 3, dtype=torch.float64)
    assert state.n_systems == torch.unique(state.system_idx).shape[0]
    torch.testing.assert_close(state.n_atoms_per_system, torch.bincount(state.system_idx))
    kinetic = ts.calc_kinetic_energy(
        masses=state.masses,
        momenta=momenta,
        system_idx=state.system_idx,
        n_systems=state.n_systems,
    )
    expected = _loop_kinetic_energy(
        state.masses, momenta, state.system_idx, state.n_systems
    )
    torch.testing.assert_close(kinetic, expected, rtol=1e-14, atol=0)
    # Without n_systems the count is read from system_idx; same values.
    torch.testing.assert_close(
        ts.calc_kinetic_energy(
            masses=state.masses, momenta=momenta, system_idx=state.system_idx
        ),
        expected,
        rtol=1e-14,
        atol=0,
    )
    dof = state.get_number_of_degrees_of_freedom()
    torch.testing.assert_close(
        ts.calc_kT(
            masses=state.masses,
            momenta=momenta,
            system_idx=state.system_idx,
            dof_per_system=dof,
        ),
        2 * expected / dof,
        rtol=1e-14,
        atol=0,
    )
    torch.testing.assert_close(
        ts.calc_kT(masses=state.masses, momenta=momenta, system_idx=state.system_idx),
        2 * expected / (3 * torch.bincount(state.system_idx)),
        rtol=1e-14,
        atol=0,
    )


def test_cell_force_equals_the_per_system_loop() -> None:
    state = _batch(torch.device("cpu"))
    generator = torch.Generator().manual_seed(5)
    momenta = torch.randn(state.n_atoms, 3, generator=generator, dtype=torch.float64)
    volume = torch.rand(state.n_systems, generator=generator, dtype=torch.float64) + 100
    stress = torch.randn(state.n_systems, 3, 3, generator=generator, dtype=torch.float64)
    pressure = torch.rand(state.n_systems, generator=generator, dtype=torch.float64)
    alpha = 1 + 3 / state.get_number_of_degrees_of_freedom()
    force = _npt_nose_hoover_isotropic_compute_cell_force(
        alpha=alpha,
        volume=volume,
        positions=state.positions,
        momenta=momenta,
        masses=state.masses,
        stress=stress,
        external_pressure=pressure,
        system_idx=state.system_idx,
    )
    kinetic = _loop_kinetic_energy(
        state.masses, momenta, state.system_idx, state.n_systems
    )
    trace = torch.diagonal(stress, dim1=-2, dim2=-1).sum(-1)
    expected = alpha * 2 * kinetic - trace * volume - pressure * volume * 3
    torch.testing.assert_close(force, expected, rtol=1e-14, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_per_step_quantities_do_not_synchronise() -> None:
    state = _batch(torch.device("cuda"))
    momenta = torch.randn(state.n_atoms, 3, dtype=torch.float64, device="cuda")
    volume = torch.full((state.n_systems,), 125.0, dtype=torch.float64, device="cuda")
    stress = torch.zeros(state.n_systems, 3, 3, dtype=torch.float64, device="cuda")
    pressure = torch.zeros(state.n_systems, dtype=torch.float64, device="cuda")
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        assert state.n_systems == 4
        _ = state.n_atoms_per_system
        dof = state.get_number_of_degrees_of_freedom()
        ts.calc_kinetic_energy(
            masses=state.masses,
            momenta=momenta,
            system_idx=state.system_idx,
            n_systems=state.n_systems,
        )
        ts.calc_kT(
            masses=state.masses,
            momenta=momenta,
            system_idx=state.system_idx,
            dof_per_system=dof,
        )
        _npt_nose_hoover_isotropic_compute_cell_force(
            alpha=1 + 3 / dof,
            volume=volume,
            positions=state.positions,
            momenta=momenta,
            masses=state.masses,
            stress=stress,
            external_pressure=pressure,
            system_idx=state.system_idx,
        )
    finally:
        torch.cuda.set_sync_debug_mode("default")
