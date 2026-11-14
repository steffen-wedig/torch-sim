"""Core molecular dynamics state and operations."""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated

import torch

from torch_sim._duecredit import dcite
from torch_sim.models.interface import ModelInterface
from torch_sim.quantities import calc_kT
from torch_sim.state import SimState
from torch_sim.units import MetalUnits, UnitAnnotation


logger = logging.getLogger(__name__)


# The metadata is the factor that converts the kwarg to internal units
TemperatureArg = Annotated[float | torch.Tensor, UnitAnnotation(MetalUnits.temperature)]
EnergyArg = Annotated[float | torch.Tensor, UnitAnnotation(MetalUnits.energy)]
TimeArg = Annotated[float | torch.Tensor, UnitAnnotation(MetalUnits.time)]
InverseTimeArg = Annotated[float | torch.Tensor, UnitAnnotation(1 / MetalUnits.time)]
PressureArg = Annotated[float | torch.Tensor, UnitAnnotation(MetalUnits.pressure)]
InversePressureArg = Annotated[
    float | torch.Tensor, UnitAnnotation(1 / MetalUnits.pressure)
]


@dataclass(kw_only=True)
class MDState(SimState):
    """State information for molecular dynamics simulations.

    This class represents the complete state of a molecular system being integrated
    with molecular dynamics. It extends the base SimState class to include additional
    attributes required for MD simulations, such as momenta, energy, and forces.
    The class also provides computed properties like velocities.

    Attributes:
        positions (torch.Tensor): Particle positions [n_particles, n_dim]
        masses (torch.Tensor): Particle masses [n_particles]
        cell (torch.Tensor): Simulation cell matrix [n_systems, n_dim, n_dim]
        pbc (bool): Whether to use periodic boundary conditions
        system_idx (torch.Tensor): System indices [n_particles]
        atomic_numbers (torch.Tensor): Atomic numbers [n_particles]
        momenta (torch.Tensor): Particle momenta [n_particles, n_dim]
        energy (torch.Tensor): Potential energy of the system [n_systems]
        forces (torch.Tensor): Forces on particles [n_particles, n_dim]
        rng (torch.Generator): RNG used by stochastic integrators (lazily
            initialised via the ``rng`` property on ``SimState``).  Stored on
            the state so that the random stream advances consistently across
            steps and can be serialised for reproducibility.

    Properties:
        velocities (torch.Tensor): Particle velocities [n_particles, n_dim]
        n_systems (int): Number of independent systems in the batch
        device (torch.device): Device on which tensors are stored
        dtype (torch.dtype): Data type of tensors
    """

    momenta: torch.Tensor
    energy: torch.Tensor
    forces: torch.Tensor

    _atom_attributes = (
        SimState._atom_attributes | {"momenta", "forces"}  # noqa: SLF001
    )
    _system_attributes = (
        SimState._system_attributes | {"energy"}  # noqa: SLF001
    )

    @property
    def velocities(self) -> torch.Tensor:
        """Velocities calculated from momenta and masses with shape
        [n_particles, n_dimensions].
        """
        return self.momenta / self.masses.unsqueeze(-1)

    def set_constrained_momenta(self, new_momenta: torch.Tensor) -> None:
        """Set new momenta, applying any constraints as needed."""
        for constraint in self.constraints:
            constraint.adjust_momenta(self, new_momenta)
        self.momenta = new_momenta

    def calc_temperature(
        self, units: MetalUnits = MetalUnits.temperature
    ) -> torch.Tensor:
        """Calculate temperature from momenta, masses, and system indices.

        Args:
            units (MetalUnits): Units to return the temperature in

        Returns:
            torch.Tensor: Calculated temperature
        """
        return self.calc_kT() / units.temperature

    def calc_kT(self) -> torch.Tensor:  # noqa: N802
        """Calculate kT from momenta, masses, and system indices.

        Returns:
            torch.Tensor: Calculated kT in energy units
        """
        return calc_kT(
            masses=self.masses,
            momenta=self.momenta,
            system_idx=self.system_idx,
            dof_per_system=self.get_number_of_degrees_of_freedom(),
        )


def initialize_momenta(
    positions: torch.Tensor,
    masses: torch.Tensor,
    system_idx: torch.Tensor,
    kT: torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Initialize particle momenta based on temperature.

    Generates random momenta for particles following the Maxwell-Boltzmann
    distribution at the specified temperature. The center of mass motion
    is removed to prevent system drift.

    Args:
        positions (torch.Tensor): Particle positions [n_particles, n_dim]
        masses (torch.Tensor): Particle masses [n_particles]
        system_idx (torch.Tensor): System indices [n_particles]
        kT (torch.Tensor): Temperature in energy units [n_systems]
        generator: Optional ``torch.Generator`` for reproducibility.

    Returns:
        torch.Tensor: Initialized momenta [n_particles, n_dim]
    """
    device = positions.device
    dtype = positions.dtype

    if kT.ndim > 0:
        kT = kT[system_idx]

    # Generate random momenta from normal distribution
    momenta = torch.randn(
        positions.shape, device=device, dtype=dtype, generator=generator
    ) * torch.sqrt(masses * kT).unsqueeze(-1)

    systemwise_momenta = torch.zeros(
        size=(int(system_idx[-1]) + 1, momenta.shape[1]), device=device, dtype=dtype
    )

    # create 3 copies of system_idx
    system_idx_3 = system_idx.view(-1, 1).repeat(1, 3)
    bincount = torch.bincount(system_idx)
    mean_momenta = torch.scatter_reduce(
        systemwise_momenta,
        dim=0,
        index=system_idx_3,
        src=momenta,
        reduce="sum",
    ) / bincount.view(-1, 1)

    return torch.where(
        torch.repeat_interleave(bincount > 1, bincount).view(-1, 1),
        momenta - mean_momenta[system_idx],
        momenta,
    )


def momentum_step[T: MDState](state: T, dt: TimeArg) -> T:
    """Update particle momenta using current forces.

    This function performs the momentum update step of velocity Verlet integration
    by applying forces over the timestep dt. It implements the equation:
    p(t+dt) = p(t) + F(t) * dt

    Args:
        state (MDState): Current system state containing forces and momenta
        dt (torch.Tensor): Integration timestep, either scalar or with shape [n_systems]

    Returns:
        MDState: Updated state with new momenta after force application

    """
    dt = torch.as_tensor(dt, device=state.device, dtype=state.dtype)
    if dt.ndim > 0:
        dt = dt[state.system_idx].unsqueeze(-1)
    new_momenta = state.momenta + state.forces * dt
    state.set_constrained_momenta(new_momenta)
    return state


def position_step[T: MDState](state: T, dt: TimeArg) -> T:
    """Update particle positions using current velocities.

    This function performs the position update step of velocity Verlet integration
    by propagating particles according to their velocities over timestep dt.
    It implements the equation: r(t+dt) = r(t) + v(t) * dt

    Args:
        state (MDState): Current system state containing positions and velocities
        dt (torch.Tensor): Integration timestep, either scalar or with shape [n_systems]

    Returns:
        MDState: Updated state with new positions after propagation

    """
    dt = torch.as_tensor(dt, device=state.device, dtype=state.dtype)
    if dt.ndim > 0:
        dt = dt[state.system_idx].unsqueeze(-1)
    new_positions = state.positions + state.velocities * dt
    state.set_constrained_positions(new_positions)
    return state


def velocity_verlet_step[T: MDState](state: T, dt: TimeArg, model: ModelInterface) -> T:
    """Perform one complete velocity Verlet integration step.

    This function implements the velocity Verlet algorithm, which provides
    time-reversible integration of the equations of motion. The integration
    sequence is:
    1. Half momentum update
    2. Full position update
    3. Force update
    4. Half momentum update

    Args:
        state: Current system state containing positions, momenta, forces
        dt: Integration timestep
        model: Neural network model that computes energies and forces

    Returns:
        Updated state after one complete velocity Verlet step

    Notes:
        - Time-reversible and symplectic integrator of second order accuracy
        - Conserves energy in the absence of numerical errors
        - Handles periodic boundary conditions if enabled in state
    """
    dt = torch.as_tensor(dt, device=state.device, dtype=state.dtype)
    dt_2 = dt / 2
    state = momentum_step(state, dt_2)
    state = position_step(state, dt)

    model_output = model(state)

    state.energy = model_output["energy"]
    state.forces = model_output["forces"]
    state.store_model_extras(model_output)
    return momentum_step(state, dt_2)


@dataclass
class NoseHooverChain:
    """State information for a Nose-Hoover chain thermostat.

    The Nose-Hoover chain is a deterministic thermostat that maintains constant
    temperature by coupling the system to a chain of thermostats. Each thermostat
    in the chain has its own positions, momenta, and masses.

    Attributes:
        positions: Positions of the chain thermostats. Shape: [n_systems, chain_length]
        momenta: Momenta of the chain thermostats. Shape: [n_systems, chain_length]
        masses: Masses of the chain thermostats. Shape: [n_systems, chain_length]
        tau: Thermostat relaxation time. Longer values give better stability
            but worse temperature control. Shape: [n_systems] or scalar
        kinetic_energy: Current kinetic energy of the coupled system. Shape: [n_systems]
        degrees_of_freedom: Number of degrees of freedom per system. Shape: [n_systems]
    """

    positions: torch.Tensor
    momenta: torch.Tensor
    masses: torch.Tensor
    tau: torch.Tensor
    kinetic_energy: torch.Tensor
    degrees_of_freedom: torch.Tensor
    system_idx: torch.Tensor | None = None


@dataclass
class NoseHooverChainFns:
    """Collection of functions for operating on a Nose-Hoover chain.

    Attributes:
        initialize (Callable): Function to initialize the chain state
        half_step (Callable): Function to perform half-step integration of chain
        update_mass (Callable): Function to update the chain masses
    """

    initialize: Callable
    half_step: Callable
    update_mass: Callable


#: Suzuki-Yoshida composition weights for higher-order symplectic integrators.
#:
#: These coefficients are used to construct high-order operator-splitting
#: schemes (Suzuki-Yoshida compositions) in molecular dynamics and Hamiltonian
#: simulations.
#:
#: The coefficients define how lower-order symplectic integrators (e.g., leapfrog)
#: can be recursively composed to achieve higher-order accuracy while preserving
#: symplectic structure.
#:
#: References:
#:     - M. Suzuki, *General Decomposition Theory of Ordered Exponentials*,
#:       Proc. Japan Acad. 69, 161 (1993).
#:     - H. Yoshida, *Construction of higher order symplectic integrators*,
#:       Phys. Lett. A 150, 262-268 (1990).
#:     - M. Tuckerman, *Statistical Mechanics: Theory and Molecular Simulation*,
#:       Oxford University Press (2010). Section 4.11
#:
#: :type: dict[int, torch.Tensor]
SUZUKI_YOSHIDA_WEIGHTS = {
    1: torch.tensor([1.0], dtype=torch.float64),
    3: torch.tensor(
        [0.828981543588751, -0.657963087177502, 0.828981543588751], dtype=torch.float64
    ),
    5: torch.tensor(
        [
            0.2967324292201065,
            0.2967324292201065,
            -0.186929716880426,
            0.2967324292201065,
            0.2967324292201065,
        ],
        dtype=torch.float64,
    ),
    7: torch.tensor(
        [
            0.784513610477560,
            0.235573213359357,
            -1.17767998417887,
            1.31518632068391,
            -1.17767998417887,
            0.235573213359357,
            0.784513610477560,
        ],
        dtype=torch.float64,
    ),
}


def nose_hoover_chain_half_step(
    momenta: torch.Tensor,
    system_idx: torch.Tensor,
    chain_positions: torch.Tensor,
    chain_momenta: torch.Tensor,
    chain_masses: torch.Tensor,
    kinetic_energy: torch.Tensor,
    degrees_of_freedom: torch.Tensor,
    kT: torch.Tensor,
    dt: float | torch.Tensor,
    chain_steps: int,
    weights: tuple[float, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Half a timestep of a Nose-Hoover chain coupled to ``momenta`` (pure function).

    ``chain_steps * len(weights)`` substeps of length ``dt / chain_steps * weight``
    (Suzuki-Yoshida), each the Martyna et al. (1996) update: chain momenta backwards,
    the coupled momenta rescaled, chain positions, chain momenta forwards. The chain
    tensors have shape ``[n_systems, chain_length]``.

    The coupled momenta are only ever rescaled per system, so the substeps accumulate
    one scale per system and ``momenta`` is rescaled once at the end. Every argument is
    a tensor or a Python constant and nothing is mutated, so the function compiles as
    one graph (``torch.compile(fullgraph=True)``) on any backend.

    Args:
        momenta: Momenta rescaled by the chain, ``[n, ...]`` with ``n`` atoms (the
            thermostat) or systems (the barostat)
        system_idx: System of each row of ``momenta``, shape ``[n]``
        chain_positions: Chain positions, ``[n_systems, chain_length]``
        chain_momenta: Chain momenta, ``[n_systems, chain_length]``
        chain_masses: Chain masses, ``[n_systems, chain_length]``
        kinetic_energy: Kinetic energy of ``momenta`` per system, ``[n_systems]``
        degrees_of_freedom: Degrees of freedom per system, ``[n_systems]``
        kT: Target temperature, scalar or ``[n_systems]``
        dt: Timestep
        chain_steps: Number of outer substeps
        weights: Suzuki-Yoshida weights

    Returns:
        Rescaled momenta, chain positions, chain momenta, kinetic energy per system
    """
    last = chain_momenta.shape[-1] - 1
    # Columns of the chain: elementwise updates on [n_systems] tensors, no copies into
    # slices of the chain tensors.
    momentum = list(chain_momenta.unbind(-1))
    mass = list(chain_masses.unbind(-1))
    momentum_scale = torch.ones_like(kinetic_energy)
    delta = dt / chain_steps
    for step in range(chain_steps * len(weights)):
        delta_2 = delta * weights[step % len(weights)] / 2.0
        delta_4 = delta_2 / 2.0
        delta_8 = delta_4 / 2.0

        # Chain momenta backwards
        if last > 0:
            force = torch.square(momentum[last - 1]) / mass[last - 1] - kT
            momentum[last] = momentum[last] + delta_4 * force
        for index in range(last - 1, 0, -1):
            force = torch.square(momentum[index - 1]) / mass[index - 1] - kT
            scale = torch.exp(-delta_8 * momentum[index + 1] / mass[index + 1])
            momentum[index] = scale * (scale * momentum[index] + delta_4 * force)

        # Coupling to the system
        force = 2.0 * kinetic_energy - degrees_of_freedom * kT
        scale = torch.exp(-delta_8 * momentum[1] / mass[1]) if last > 0 else 1.0
        momentum[0] = scale * (scale * momentum[0] + delta_4 * force)

        # Rescale the system momenta (accumulated, applied once below)
        scale = torch.exp(-delta_2 * momentum[0] / mass[0])
        kinetic_energy = kinetic_energy * torch.square(scale)
        momentum_scale = momentum_scale * scale

        # Chain positions
        chain_positions = (
            chain_positions + delta_2 * torch.stack(momentum, dim=-1) / chain_masses
        )

        # Chain momenta forwards
        force = 2.0 * kinetic_energy - degrees_of_freedom * kT
        for index in range(last):
            scale = torch.exp(-delta_8 * momentum[index + 1] / mass[index + 1])
            momentum[index] = scale * (scale * momentum[index] + delta_4 * force)
            force = torch.square(momentum[index]) / mass[index] - kT
        momentum[last] = momentum[last] + delta_4 * force

    scale_per_row = momentum_scale[system_idx]
    scale_per_row = scale_per_row.reshape(-1, *([1] * (momenta.ndim - 1)))
    return (
        momenta * scale_per_row,
        chain_positions,
        torch.stack(momentum, dim=-1),
        kinetic_energy,
    )


@dcite("10.1063/1.463940")
@dcite("10.2183/pjab.69.161")
@dcite("10.1016/0375-9601(90)90092-3")
def construct_nose_hoover_chain(
    dt: TimeArg,
    chain_length: int,
    chain_steps: int,
    sy_steps: int,
    tau: TimeArg,
    *,
    compiled: bool = False,
) -> NoseHooverChainFns:
    """Creates functions to simulate a Nose-Hoover Chain thermostat.

    Implements the direct translation method from Martyna et al. for thermal ensemble
    sampling using Nose-Hoover chains. The chains are updated using a symmetric
    splitting scheme with two half-steps per simulation step.

    The integration uses a multi-timestep approach with Suzuki-Yoshida (SY) splitting:
    - The chain evolution is split into nc substeps (chain_steps)
    - Each substep is further split into sy_steps
    - Each SY step has length δi = Δt*wi/nc where wi are the SY weights

    Args:
        dt: Simulation timestep
        chain_length: Number of thermostats in the chain
        chain_steps: Number of outer substeps for chain integration
        sy_steps: Number of Suzuki-Yoshida steps (must be 1, 3, 5, or 7)
        tau: Temperature equilibration timescale (in units of dt)
            Larger values give better stability but slower equilibration
        compiled: Run the half-step through ``torch.compile`` (one graph per chain
            setting; any backend Inductor supports). The chain works on
            ``[n_systems, chain_length]`` tensors, so eagerly it is launch-bound:
            dozens of tiny operations per substep.

    Returns:
        NoseHooverChainFns containing:
        - initialize: Function to create initial chain state
        - half_step: Function to evolve chain for half timestep
        - update_mass: Function to update chain masses

    References:
        Martyna et al. "Nose-Hoover chains: the canonical ensemble via
            continuous dynamics"
        J. Chem. Phys. 97, 2635 (1992)
    """

    def init_fn(
        degrees_of_freedom: torch.Tensor, KE: torch.Tensor, kT: torch.Tensor
    ) -> NoseHooverChain:
        """Initialize a Nose-Hoover chain state.

        Args:
            degrees_of_freedom: Number of degrees of freedom per system, shape [n_systems]
            KE: Initial kinetic energy per system, shape [n_systems]
            kT: Target temperature in energy units, shape [n_systems] or scalar

        Returns:
            Initial NoseHooverChain state
        """
        device = KE.device
        dtype = KE.dtype

        # Ensure n_systems is determined from KE shape
        n_systems = KE.shape[0] if KE.dim() > 0 else 1

        # Initialize chain variables with proper batch dimensions
        xi = torch.zeros((n_systems, chain_length), dtype=dtype, device=device)
        p_xi = torch.zeros((n_systems, chain_length), dtype=dtype, device=device)

        # Broadcast tau to match n_systems
        tau_batched = torch.as_tensor(tau, device=device, dtype=dtype)
        if tau_batched.ndim == 0:
            tau_batched = tau_batched.expand(n_systems)

        # Ensure kT has proper batch dimension
        kT_batched = torch.as_tensor(kT, device=device, dtype=dtype)
        if kT_batched.ndim == 0:
            kT_batched = kT_batched.expand(n_systems)

        Q = (
            kT_batched.unsqueeze(-1)
            * torch.square(tau_batched).unsqueeze(-1)
            * torch.ones((n_systems, chain_length), dtype=dtype, device=device)
        )
        Q[:, 0] *= degrees_of_freedom

        return NoseHooverChain(xi, p_xi, Q, tau_batched, KE, degrees_of_freedom)

    weights = tuple(SUZUKI_YOSHIDA_WEIGHTS[sy_steps].tolist())
    half_step_kernel = (
        torch.compile(nose_hoover_chain_half_step, fullgraph=True)
        if compiled
        else nose_hoover_chain_half_step
    )

    def half_step_chain_fn(
        P: torch.Tensor,
        state: NoseHooverChain,
        kT: torch.Tensor,
        system_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, NoseHooverChain]:
        """Evolve chain for half timestep using multi-timestep integration.

        Args:
            P: System momenta to be rescaled
            state: Current chain state
            kT: Target temperature
            system_idx: Index of the system being evolved

        Returns:
            Tuple of (rescaled momenta, updated chain state)
        """
        timestep = (
            dt.to(device=P.device, dtype=P.dtype) if isinstance(dt, torch.Tensor) else dt
        )
        P, positions, momenta, kinetic_energy = half_step_kernel(
            P,
            system_idx,
            state.positions,
            state.momenta,
            state.masses,
            state.kinetic_energy,
            state.degrees_of_freedom,
            torch.as_tensor(kT, device=P.device, dtype=P.dtype),
            timestep,
            chain_steps,
            weights,
        )
        return P, NoseHooverChain(
            positions,
            momenta,
            state.masses,
            state.tau,
            kinetic_energy,
            state.degrees_of_freedom,
        )

    def update_chain_mass_fn(
        chain_state: NoseHooverChain, kT: float | torch.Tensor
    ) -> NoseHooverChain:
        """Update chain masses to maintain target oscillation period.

        Args:
            chain_state: Current chain state
            kT: Target temperature

        Returns:
            Updated chain state with new masses
        """
        device = chain_state.positions.device
        dtype = chain_state.positions.dtype

        # Get number of systems
        n_systems = chain_state.kinetic_energy.shape[0]

        # Ensure kT has proper batch dimension
        kT_batched = torch.as_tensor(kT, device=device, dtype=dtype)
        if kT_batched.ndim == 0:
            kT_batched = kT_batched.expand(n_systems)

        Q = (
            kT_batched.unsqueeze(-1)
            * torch.square(chain_state.tau).unsqueeze(-1)
            * torch.ones((n_systems, chain_length), dtype=dtype, device=device)
        )
        Q[:, 0] *= chain_state.degrees_of_freedom

        return NoseHooverChain(
            chain_state.positions,
            chain_state.momenta,
            Q,
            chain_state.tau,
            chain_state.kinetic_energy,
            chain_state.degrees_of_freedom,
        )

    return NoseHooverChainFns(init_fn, half_step_chain_fn, update_chain_mass_fn)
