"""
Tests for DynamicSystemNN.simulate_trajectories and additional coverage
for dynamic_system_neural_network.py.
"""

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trained_phnn():
    import phlearn.phnns as phnn
    import phlearn.phsystems.ode as phsys

    M = np.diag([0.5, 0.5])

    def hamiltonian(x):
        M_t = torch.tensor(M, dtype=x.dtype)
        return (x @ M_t * x).sum(dim=-1, keepdim=True)

    def grad_h(x):
        return 2.0 * M @ x

    system = phsys.PseudoHamiltonianSystem(
        nstates=2,
        hamiltonian=hamiltonian,
        grad_hamiltonian=grad_h,
        dissipation_matrix=np.diag([0.0, 0.2]),
    )
    t_axis = np.linspace(0, 3.0, 31)
    traindata, dxdt = phnn.generate_dataset(
        system, ntrajectories=5, t_sample=t_axis, seed=50
    )
    torch.manual_seed(10)
    model = phnn.PseudoHamiltonianNN(
        2,
        dissipation_est=phnn.R_estimator([False, True]),
    )
    phnn.train(
        model,
        integrator='midpoint',
        traindata=(traindata, dxdt),
        epochs=3,
        batch_size=32,
        verbose=False,
    )
    return model


# ---------------------------------------------------------------------------
# DynamicSystemNN.simulate_trajectory additional scenarios
# ---------------------------------------------------------------------------

class TestDynamicSystemNNSimulateTrajectory:
    def setup_method(self):
        self.model = _make_trained_phnn()

    def test_simulate_with_noise_no_nans(self):
        t = np.linspace(0, 2, 21)
        x0 = np.array([1.0, 0.0])
        xs, _ = self.model.simulate_trajectory(
            integrator=False,
            t_sample=t,
            x0=x0,
            noise_std=0.01,
        )
        assert not np.isnan(xs).any()

    def test_simulate_without_x0_uses_sampler(self):
        t = np.linspace(0, 1, 11)
        # When x0 is None, internal sampler is used
        xs, _ = self.model.simulate_trajectory(
            integrator=False,
            t_sample=t,
            x0=None,
        )
        assert xs is not None
        assert xs.shape == (11, 2)

    def test_simulate_with_euler_integrator(self):
        T = 21
        t = np.linspace(0, 2, T)
        x0 = np.array([1.0, 0.0])
        xs, _ = self.model.simulate_trajectory(
            integrator='euler',
            t_sample=t,
            x0=x0,
        )
        assert xs.shape == (T, 2)
        assert not np.isnan(xs).any()

    def test_simulate_returns_none_us_without_controller(self):
        t = np.linspace(0, 1, 11)
        x0 = np.array([0.5, 0.5])
        _, us = self.model.simulate_trajectory(
            integrator=False,
            t_sample=t,
            x0=x0,
        )
        assert us is None


# ---------------------------------------------------------------------------
# generate_dataset with nsamples limit
# ---------------------------------------------------------------------------

class TestGenerateDatasetWithNsamples:
    def setup_method(self):
        import phlearn.phnns as phnn
        import phlearn.phsystems.ode as phsys

        M = np.diag([0.5, 0.5])

        def hamiltonian(x):
            M_t = torch.tensor(M, dtype=x.dtype)
            return (x @ M_t * x).sum(dim=-1, keepdim=True)

        def grad_h(x):
            return 2.0 * M @ x

        self.system = phsys.PseudoHamiltonianSystem(
            nstates=2,
            hamiltonian=hamiltonian,
            grad_hamiltonian=grad_h,
            dissipation_matrix=np.diag([0.0, 0.2]),
        )
        self.phnn = phnn
        self.t_axis = np.linspace(0, 2.0, 21)

    def test_nsamples_limits_dataset_size(self):
        nsamples = 10
        traindata, dxdt = self.phnn.generate_dataset(
            self.system, ntrajectories=5, t_sample=self.t_axis,
            nsamples=nsamples, seed=60
        )
        assert traindata[0].shape[0] == nsamples
        assert dxdt.shape[0] == nsamples

    def test_true_derivatives_flag(self):
        """true_derivatives=True should return actual x_dot values."""
        traindata, dxdt = self.phnn.generate_dataset(
            self.system, ntrajectories=3, t_sample=self.t_axis,
            true_derivatives=True, seed=61
        )
        assert traindata is not None
        assert dxdt is not None
        # dxdt should still have the right shape
        expected_rows = 3 * (len(self.t_axis) - 1)
        assert dxdt.shape == (expected_rows, 2)


# ---------------------------------------------------------------------------
# PseudoHamiltonianSystem.set_controller
# ---------------------------------------------------------------------------

class TestPseudoHamiltonianSystemSetController:
    def setup_method(self):
        import phlearn.phsystems.ode as phsys

        def grad_h(x):
            return x

        self.system = phsys.PseudoHamiltonianSystem(
            nstates=2,
            grad_hamiltonian=grad_h,
        )

    def test_set_controller_stores_reference(self):
        dummy = object()
        self.system.set_controller(dummy)
        assert self.system.controller is dummy

    def test_set_controller_to_none(self):
        self.system.set_controller(None)
        assert self.system.controller is None


# ---------------------------------------------------------------------------
# PseudoHamiltonianSystem._dH using autograd
# ---------------------------------------------------------------------------

class TestPseudoHamiltonianSystemAutograd:
    def test_dH_computed_via_autograd(self):
        """When only hamiltonian is provided, _dH uses torch autograd.
        The system passes x.T to dH, so for x shape (1,2) it gets (2,1).
        The hamiltonian must handle this shape."""
        import phlearn.phsystems.ode as phsys
        import torch

        def hamiltonian(x):
            # x arrives as (2, 1) after the .T call in x_dot
            # Use element-wise ops compatible with any shape
            return (x ** 2).sum(dim=0, keepdim=True) * 0.5

        sys = phsys.PseudoHamiltonianSystem(
            nstates=2,
            hamiltonian=hamiltonian,
        )
        x = np.array([[1.0, 0.5]])
        # _dH receives x.T of shape (2, 1)
        dh = sys.dH(x.T)
        assert dh is not None
        assert not np.isnan(np.atleast_1d(dh)).any()
