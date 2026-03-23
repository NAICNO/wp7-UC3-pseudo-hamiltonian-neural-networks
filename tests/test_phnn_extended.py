"""
Extended tests for PseudoHamiltonianNN:
- True Hamiltonian provided (with and without grad)
- True dissipation provided (callable and static)
- True external forces provided
- Custom skewsymmetric_matrix
- x_dot output correctness
- simulate_trajectory with midpoint integrator
- PseudoHamiltonianNN with custom HamiltonianNN estimator
"""

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _canonical_S(nstates=2):
    npos = nstates // 2
    return np.block([
        [np.zeros([npos, npos]), np.eye(npos)],
        [-np.eye(npos), np.zeros([npos, npos])],
    ])


def _make_hamiltonian_torch():
    M = torch.tensor([[0.5, 0.0], [0.0, 0.5]])

    def H(x):
        return (x @ M * x).sum(dim=-1, keepdim=True)

    return H


def _make_grad_hamiltonian_torch():
    def dH(x):
        return x  # grad of 0.5 * ||x||^2

    return dH


def _make_t_axis(tmax=3.0, dt=0.1):
    return np.linspace(0, tmax, int(round(tmax / dt)) + 1)


# ---------------------------------------------------------------------------
# PseudoHamiltonianNN with true Hamiltonian
# ---------------------------------------------------------------------------

class TestPHNNWithTrueHamiltonian:
    def setup_method(self):
        import phlearn.phnns as phnn
        self.phnn = phnn

    def test_instantiate_with_true_hamiltonian(self):
        H_true = _make_hamiltonian_torch()
        model = self.phnn.PseudoHamiltonianNN(
            nstates=2,
            hamiltonian_true=H_true,
            dissipation_est=self.phnn.R_estimator([False, True]),
        )
        assert model is not None
        assert model.hamiltonian_provided is True

    def test_instantiate_with_true_hamiltonian_and_grad(self):
        H_true = _make_hamiltonian_torch()
        dH_true = _make_grad_hamiltonian_torch()
        model = self.phnn.PseudoHamiltonianNN(
            nstates=2,
            hamiltonian_true=H_true,
            grad_hamiltonian_true=dH_true,
            dissipation_est=self.phnn.R_estimator([False, True]),
        )
        assert model is not None
        assert model.grad_hamiltonian_provided is True

    def test_instantiate_with_grad_only(self):
        dH_true = _make_grad_hamiltonian_torch()
        model = self.phnn.PseudoHamiltonianNN(
            nstates=2,
            grad_hamiltonian_true=dH_true,
            dissipation_est=self.phnn.R_estimator([False, True]),
        )
        assert model is not None
        assert model.grad_hamiltonian_provided is True

    def test_x_dot_shape_with_true_hamiltonian(self):
        H_true = _make_hamiltonian_torch()
        model = self.phnn.PseudoHamiltonianNN(
            nstates=2,
            hamiltonian_true=H_true,
            dissipation_est=self.phnn.R_estimator([False, True]),
        )
        x = torch.randn(5, 2)
        t = torch.randn(5, 1)
        result = model._x_dot(x, t)
        assert result.shape == (5, 2)

    def test_x_dot_no_nans(self):
        H_true = _make_hamiltonian_torch()
        model = self.phnn.PseudoHamiltonianNN(
            nstates=2,
            hamiltonian_true=H_true,
            dissipation_est=self.phnn.R_estimator([False, True]),
        )
        x = torch.randn(8, 2)
        t = torch.randn(8, 1)
        result = model._x_dot(x, t)
        assert not torch.isnan(result).any()


# ---------------------------------------------------------------------------
# PseudoHamiltonianNN with true dissipation
# ---------------------------------------------------------------------------

class TestPHNNWithTrueDissipation:
    def setup_method(self):
        import phlearn.phnns as phnn
        self.phnn = phnn

    def test_static_true_dissipation(self):
        R_true = torch.tensor([[0.0, 0.0], [0.0, 0.3]])
        model = self.phnn.PseudoHamiltonianNN(
            nstates=2,
            dissipation_true=R_true,
        )
        assert model is not None
        assert model.dissipation_provided is True

    def test_callable_true_dissipation(self):
        def R_callable(x):
            return torch.tensor([[0.0, 0.0], [0.0, 0.3]]).expand(x.shape[0], 2, 2)

        model = self.phnn.PseudoHamiltonianNN(
            nstates=2,
            dissipation_true=R_callable,
        )
        assert model.dissipation_provided is True

    def test_x_dot_with_true_dissipation(self):
        R_true = torch.tensor([[0.0, 0.0], [0.0, 0.3]])
        model = self.phnn.PseudoHamiltonianNN(
            nstates=2,
            dissipation_true=R_true,
        )
        x = torch.randn(5, 2)
        t = torch.randn(5, 1)
        result = model._x_dot(x, t)
        assert result.shape == (5, 2)
        assert not torch.isnan(result).any()


# ---------------------------------------------------------------------------
# PseudoHamiltonianNN with true external forces
# ---------------------------------------------------------------------------

class TestPHNNWithTrueExternalForces:
    def setup_method(self):
        import phlearn.phnns as phnn
        self.phnn = phnn

    def _make_true_ext_forces(self):
        def F(x, t):
            return torch.zeros_like(x)
        return F

    def test_instantiate_with_true_external_forces(self):
        model = self.phnn.PseudoHamiltonianNN(
            nstates=2,
            external_forces_true=self._make_true_ext_forces(),
        )
        assert model is not None
        assert model.external_forces_provided is True

    def test_x_dot_with_true_external_forces(self):
        model = self.phnn.PseudoHamiltonianNN(
            nstates=2,
            external_forces_true=self._make_true_ext_forces(),
            dissipation_est=self.phnn.R_estimator([False, True]),
        )
        x = torch.randn(4, 2)
        t = torch.randn(4, 1)
        result = model._x_dot(x, t)
        assert result.shape == (4, 2)


# ---------------------------------------------------------------------------
# PseudoHamiltonianNN with custom skewsymmetric_matrix
# ---------------------------------------------------------------------------

class TestPHNNWithCustomSkewMatrix:
    def setup_method(self):
        import phlearn.phnns as phnn
        self.phnn = phnn

    def test_custom_skewsymmetric_matrix(self):
        """Should accept a pre-built skew-symmetric S matrix."""
        S = torch.tensor([[0.0, 2.0], [-2.0, 0.0]])
        model = self.phnn.PseudoHamiltonianNN(
            nstates=2,
            skewsymmetric_matrix=S,
        )
        assert model is not None

    def test_callable_skewsymmetric_matrix(self):
        def S_fn(x):
            return torch.tensor([[0.0, 1.0], [-1.0, 0.0]])

        model = self.phnn.PseudoHamiltonianNN(
            nstates=2,
            skewsymmetric_matrix=S_fn,
            dissipation_est=self.phnn.R_estimator([False, True]),
        )
        x = torch.randn(3, 2)
        t = torch.randn(3, 1)
        result = model._x_dot(x, t)
        assert result.shape == (3, 2)

    def test_raises_with_odd_nstates_no_custom_S(self):
        with pytest.raises(Exception):
            self.phnn.PseudoHamiltonianNN(nstates=3)


# ---------------------------------------------------------------------------
# PseudoHamiltonianNN with custom HamiltonianNN estimator
# ---------------------------------------------------------------------------

class TestPHNNWithCustomHamiltonianEst:
    def setup_method(self):
        import phlearn.phnns as phnn
        self.phnn = phnn

    def test_custom_hamiltonian_estimator(self):
        H_est = self.phnn.HamiltonianNN(nstates=2, hidden_dim=32)
        model = self.phnn.PseudoHamiltonianNN(
            nstates=2,
            hamiltonian_est=H_est,
            dissipation_est=self.phnn.R_estimator([False, True]),
        )
        assert model is not None
        x = torch.randn(5, 2)
        t = torch.randn(5, 1)
        result = model._x_dot(x, t)
        assert result.shape == (5, 2)


# ---------------------------------------------------------------------------
# PseudoHamiltonianNN simulate_trajectory with midpoint integrator
# ---------------------------------------------------------------------------

class TestPHNNSimulateWithMidpoint:
    def setup_method(self):
        import phlearn.phnns as phnn
        self.phnn = phnn

        system_module = __import__('phlearn.phsystems.ode', fromlist=['PseudoHamiltonianSystem'])
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
        t_axis = _make_t_axis(tmax=3.0, dt=0.1)
        traindata, dxdt = phnn.generate_dataset(
            system, ntrajectories=5, t_sample=t_axis, seed=15
        )

        torch.manual_seed(5)
        self.model = phnn.PseudoHamiltonianNN(
            2,
            dissipation_est=phnn.R_estimator([False, True]),
        )
        phnn.train(
            self.model,
            integrator='midpoint',
            traindata=(traindata, dxdt),
            epochs=5,
            batch_size=32,
            verbose=False,
        )

    def test_simulate_with_midpoint_returns_correct_length(self):
        T = 31
        t_sample = np.linspace(0, 3, T)
        x0 = np.array([1.0, 0.0])
        # midpoint is not a valid inference integrator; falls back to rk4
        xs, _ = self.model.simulate_trajectory(
            integrator='midpoint',
            t_sample=t_sample,
            x0=x0,
        )
        assert xs.shape[0] == T

    def test_simulate_with_midpoint_no_nans(self):
        T = 21
        t_sample = np.linspace(0, 2, T)
        x0 = np.array([1.0, 0.0])
        xs, _ = self.model.simulate_trajectory(
            integrator='midpoint',
            t_sample=t_sample,
            x0=x0,
        )
        assert not np.isnan(xs).any()
