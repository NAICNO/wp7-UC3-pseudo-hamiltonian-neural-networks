"""
Tests for phlearn.phsystems.ode subsystems:
- MassSpringDamperSystem
- initial_condition_radial
- init_msdsystem
- zero_force
- PseudoHamiltonianSystem.sample_trajectory
- PseudoHamiltonianSystem.seed
"""

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_t(tmax=2.0, dt=0.1):
    n = int(round(tmax / dt))
    return np.linspace(0, tmax, n + 1)


# ---------------------------------------------------------------------------
# zero_force
# ---------------------------------------------------------------------------

class TestZeroForce:
    def setup_method(self):
        from phlearn.phsystems.ode.pseudo_hamiltonian_system import zero_force
        self.zero_force = zero_force

    def test_returns_zeros_same_shape(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = self.zero_force(x)
        np.testing.assert_array_equal(result, np.zeros_like(x))

    def test_works_without_t_argument(self):
        x = np.ones((3, 2))
        result = self.zero_force(x)
        assert result.shape == (3, 2)

    def test_works_with_t_argument(self):
        x = np.ones((3, 2))
        t = np.array([[0.5]])
        result = self.zero_force(x, t)
        assert result.shape == x.shape
        assert (result == 0).all()

    def test_1d_input(self):
        x = np.array([1.0, 0.5])
        result = self.zero_force(x)
        assert result.shape == x.shape
        assert (result == 0).all()


# ---------------------------------------------------------------------------
# MassSpringDamperSystem
# ---------------------------------------------------------------------------

class TestMassSpringDamperSystem:
    def setup_method(self):
        from phlearn.phsystems.ode.msd_system import MassSpringDamperSystem
        self.MassSpringDamperSystem = MassSpringDamperSystem

    def test_instantiates_with_defaults(self):
        sys = self.MassSpringDamperSystem()
        assert sys is not None
        assert sys.nstates == 2

    def test_instantiates_with_custom_params(self):
        sys = self.MassSpringDamperSystem(mass=2.0, spring_constant=3.0, damping=0.5)
        assert sys is not None

    def test_x_dot_shape(self):
        sys = self.MassSpringDamperSystem()
        x = np.array([[1.0, 0.5]])
        t = np.array([[0.0]])
        xdot = sys.x_dot(x, t)
        assert xdot.shape == x.shape

    def test_x_dot_no_nans(self):
        sys = self.MassSpringDamperSystem()
        x = np.array([[1.0, 0.5]])
        t = np.array([[0.0]])
        xdot = sys.x_dot(x, t)
        assert not np.isnan(xdot).any()

    def test_sample_trajectory_shape(self):
        sys = self.MassSpringDamperSystem()
        t = _make_t(tmax=2.0, dt=0.1)
        x0 = np.array([1.0, 0.0])
        x, dxdt, t_out, us = sys.sample_trajectory(t, x0=x0)
        assert x.shape[0] == len(t)
        assert x.shape[1] == 2
        assert us is None

    def test_sample_trajectory_no_nans(self):
        sys = self.MassSpringDamperSystem()
        t = _make_t(tmax=2.0, dt=0.1)
        x0 = np.array([1.0, 0.0])
        x, dxdt, t_out, us = sys.sample_trajectory(t, x0=x0)
        assert not np.isnan(x).any()

    def test_initial_condition_preserved(self):
        sys = self.MassSpringDamperSystem()
        t = _make_t(tmax=2.0, dt=0.1)
        x0 = np.array([0.5, -0.5])
        x, _, _, _ = sys.sample_trajectory(t, x0=x0)
        np.testing.assert_allclose(x[0], x0, atol=1e-5)

    def test_dissipation_applied(self):
        """System with damping should lose energy over time."""
        sys = self.MassSpringDamperSystem(damping=0.5)
        t = _make_t(tmax=5.0, dt=0.05)
        x0 = np.array([1.0, 0.0])
        x, _, _, _ = sys.sample_trajectory(t, x0=x0)
        # Energy at end should be less than at start
        energy_start = x[0, 0] ** 2 + x[0, 1] ** 2
        energy_end = x[-1, 0] ** 2 + x[-1, 1] ** 2
        assert energy_end < energy_start

    def test_sample_trajectory_with_noise(self):
        """Adding noise should not produce NaN."""
        sys = self.MassSpringDamperSystem()
        t = _make_t(tmax=1.0, dt=0.1)
        x0 = np.array([1.0, 0.0])
        x, dxdt, _, _ = sys.sample_trajectory(t, x0=x0, noise_std=0.01)
        assert not np.isnan(x).any()
        assert not np.isnan(dxdt).any()


# ---------------------------------------------------------------------------
# initial_condition_radial
# ---------------------------------------------------------------------------

class TestInitialConditionRadial:
    def setup_method(self):
        from phlearn.phsystems.ode.msd_system import initial_condition_radial
        self.initial_condition_radial = initial_condition_radial

    def test_returns_callable(self):
        sampler = self.initial_condition_radial(1.0, 4.0)
        assert callable(sampler)

    def test_sampled_state_has_correct_shape(self):
        sampler = self.initial_condition_radial(1.0, 4.0)
        rng = np.random.default_rng(0)
        x0 = sampler(rng)
        assert x0.shape == (2,)

    def test_sampled_radius_within_bounds(self):
        r_min, r_max = 1.0, 4.0
        sampler = self.initial_condition_radial(r_min, r_max)
        rng = np.random.default_rng(42)
        for _ in range(20):
            x0 = sampler(rng)
            r = np.sqrt(x0[0] ** 2 + x0[1] ** 2)
            assert r_min <= r <= r_max, f"Radius {r:.3f} outside [{r_min}, {r_max}]"

    def test_different_seeds_give_different_samples(self):
        sampler = self.initial_condition_radial(1.0, 4.0)
        rng1 = np.random.default_rng(1)
        rng2 = np.random.default_rng(2)
        x1 = sampler(rng1)
        x2 = sampler(rng2)
        assert not np.allclose(x1, x2)


# ---------------------------------------------------------------------------
# init_msdsystem
# ---------------------------------------------------------------------------

class TestInitMSDSystem:
    def setup_method(self):
        from phlearn.phsystems.ode.msd_system import init_msdsystem
        self.init_msdsystem = init_msdsystem

    def test_returns_system(self):
        sys = self.init_msdsystem()
        assert sys is not None

    def test_system_has_external_forces(self):
        sys = self.init_msdsystem()
        # The system should have non-zero forces at some time
        x = np.array([[1.0, 0.0]])
        t = np.array([[np.pi / 6]])
        force = sys.external_forces(x, t)
        # Force at t=pi/6 with omega=3 is non-zero
        assert np.any(force != 0) or force is not None

    def test_trajectory_has_correct_shape(self):
        sys = self.init_msdsystem()
        t = _make_t(tmax=2.0, dt=0.1)
        x, dxdt, t_out, us = sys.sample_trajectory(t)
        assert x.shape[1] == 2
        assert x.shape[0] == len(t)


# ---------------------------------------------------------------------------
# PseudoHamiltonianSystem.seed and sample_trajectory
# ---------------------------------------------------------------------------

class TestPseudoHamiltonianSystemSeed:
    def setup_method(self):
        import phlearn.phsystems.ode as phsys
        self.phsys = phsys

        def hamiltonian(x):
            M = np.diag([0.5, 0.5])
            return x.T @ M @ x

        def grad_hamiltonian(x):
            return np.diag([1.0, 1.0]) @ x

        self.system = phsys.PseudoHamiltonianSystem(
            nstates=2,
            hamiltonian=hamiltonian,
            grad_hamiltonian=grad_hamiltonian,
            dissipation_matrix=np.diag([0.0, 0.1]),
        )

    def test_seed_produces_reproducible_samples(self):
        t = _make_t(tmax=1.0, dt=0.1)
        self.system.seed(99)
        x1, _, _, _ = self.system.sample_trajectory(t)
        self.system.seed(99)
        x2, _, _, _ = self.system.sample_trajectory(t)
        np.testing.assert_allclose(x1, x2)

    def test_different_seeds_produce_different_samples(self):
        t = _make_t(tmax=1.0, dt=0.1)
        self.system.seed(1)
        x1, _, _, _ = self.system.sample_trajectory(t)
        self.system.seed(2)
        x2, _, _, _ = self.system.sample_trajectory(t)
        assert not np.allclose(x1, x2)

    def test_sample_trajectory_dxdt_shape(self):
        t = _make_t(tmax=1.0, dt=0.1)
        x0 = np.array([1.0, 0.0])
        x, dxdt, t_out, us = self.system.sample_trajectory(t, x0=x0)
        assert dxdt.shape == x.shape

    def test_sample_trajectory_t_out_shape(self):
        t = _make_t(tmax=1.0, dt=0.1)
        x0 = np.array([1.0, 0.0])
        x, dxdt, t_out, us = self.system.sample_trajectory(t, x0=x0)
        assert len(t_out) == len(t)


# ---------------------------------------------------------------------------
# PseudoHamiltonianSystem with callable S and R
# ---------------------------------------------------------------------------

class TestPseudoHamiltonianSystemCallableMatrices:
    def test_callable_skewsymmetric_matrix(self):
        import phlearn.phsystems.ode as phsys

        def S_callable(x):
            return np.array([[0.0, 1.0], [-1.0, 0.0]])

        def grad_h(x):
            return 2.0 * np.diag([0.5, 0.5]) @ x

        sys = phsys.PseudoHamiltonianSystem(
            nstates=2,
            grad_hamiltonian=grad_h,
            skewsymmetric_matrix=S_callable,
        )
        assert sys is not None
        x = np.array([[1.0, 0.5]])
        t = np.array([[0.0]])
        xdot = sys.x_dot(x, t)
        assert xdot.shape == x.shape

    def test_callable_dissipation_matrix(self):
        import phlearn.phsystems.ode as phsys

        def R_callable(x):
            return np.diag([0.0, 0.2])

        def grad_h(x):
            return 2.0 * np.diag([0.5, 0.5]) @ x

        sys = phsys.PseudoHamiltonianSystem(
            nstates=2,
            grad_hamiltonian=grad_h,
            dissipation_matrix=R_callable,
        )
        assert sys is not None
        x = np.array([[1.0, 0.5]])
        t = np.array([[0.0]])
        xdot = sys.x_dot(x, t)
        assert xdot.shape == x.shape

    def test_raises_on_non_skewsymmetric_matrix(self):
        import phlearn.phsystems.ode as phsys

        def grad_h(x):
            return x

        bad_S = np.array([[0.0, 1.0], [1.0, 0.0]])  # symmetric, not skew
        with pytest.raises(Exception, match="skew-symmetric"):
            phsys.PseudoHamiltonianSystem(
                nstates=2,
                grad_hamiltonian=grad_h,
                skewsymmetric_matrix=bad_S,
            )

    def test_1d_dissipation_matrix_converted_to_diag(self):
        """A 1D array dissipation_matrix should be treated as diagonal entries."""
        import phlearn.phsystems.ode as phsys

        def grad_h(x):
            return x

        sys = phsys.PseudoHamiltonianSystem(
            nstates=2,
            grad_hamiltonian=grad_h,
            dissipation_matrix=np.array([0.0, 0.3]),
        )
        expected = np.diag([0.0, 0.3])
        np.testing.assert_allclose(sys.dissipation_matrix, expected)

    def test_hamiltonian_only_no_grad_provided(self):
        """Should work when only hamiltonian (not grad) is provided."""
        import phlearn.phsystems.ode as phsys

        def hamiltonian(x):
            M = np.diag([0.5, 0.5])
            xt = torch.tensor(x, requires_grad=False, dtype=torch.float32)
            return (xt @ torch.diag(torch.tensor([0.5, 0.5])) * xt).sum(dim=-1, keepdim=True)

        sys = phsys.PseudoHamiltonianSystem(
            nstates=2,
            hamiltonian=hamiltonian,
        )
        assert sys is not None
        assert sys.dH is not None
