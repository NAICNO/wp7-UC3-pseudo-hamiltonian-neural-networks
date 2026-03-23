"""
Tests for the UC3 pseudo-Hamiltonian neural networks demonstrator.

Covers:
1. phlearn imports work correctly
2. PseudoHamiltonianSystem creates valid systems
3. Training data generation produces correct shapes
4. PHNN model trains and loss converges
5. Baseline model trains
6. PHNN recovers damping constant within tolerance
7. External force function handles both (1,1) and (T,) shaped t inputs
8. Forced system data generation works
9. PHNN with external forces trains successfully
10. Long-horizon trajectory simulation does not crash or produce NaN
11. simulate_trajectory produces correct output shapes
"""

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------

def _make_M(spring_constant=1.0, mass=1.0):
    return np.diag([spring_constant / 2.0, 1.0 / (2.0 * mass)])


def make_hamiltonian(spring_constant=1.0, mass=1.0):
    """Return a quadratic Hamiltonian callable (torch tensor in, tensor out)."""
    M = _make_M(spring_constant, mass)

    def hamiltonian(x):
        M_t = torch.tensor(M, dtype=x.dtype)
        return (x @ M_t * x).sum(dim=-1, keepdim=True)

    return hamiltonian


def make_grad_hamiltonian(spring_constant=1.0, mass=1.0):
    """Return gradient of Hamiltonian callable (ndarray (2,N) in, ndarray out)."""
    M = _make_M(spring_constant, mass)

    def grad_hamiltonian(x):
        return 2.0 * M @ x

    return grad_hamiltonian


def make_dissipation(damping=0.3):
    return np.diag([0.0, damping])


def make_t_axis(tmax=5.0, dt=0.1):
    """Return a 1D time axis required by generate_dataset / sample_trajectory."""
    n = int(round(tmax / dt))
    return np.linspace(0, tmax, n + 1)   # shape (T,)


def make_system(damping=0.3, external_forces=None):
    """Convenience: build a canonical PseudoHamiltonianSystem."""
    import phlearn.phsystems.ode as phsys
    return phsys.PseudoHamiltonianSystem(
        nstates=2,
        hamiltonian=make_hamiltonian(),
        grad_hamiltonian=make_grad_hamiltonian(),
        dissipation_matrix=make_dissipation(damping),
        external_forces=external_forces,
    )


def external_force(x, t):
    """
    Sinusoidal external force applied to the momentum state only.
    Handles both (1, 1) shaped t (during ODE integration) and
    (T,) shaped t (during batch operations).
    """
    f0 = 0.5
    omega = np.pi / 2.0
    t_arr = np.asarray(t).flatten()
    force_val = f0 * np.sin(omega * t_arr)
    result = np.zeros_like(np.atleast_2d(x))
    if result.ndim > 1:
        result[..., 1] = force_val.flatten()[: result.shape[0]]
    else:
        result[1] = force_val
    return result.squeeze() if np.asarray(x).ndim < 2 else result


# ---------------------------------------------------------------------------
# 1. Import tests
# ---------------------------------------------------------------------------

class TestImports:
    def test_phsystems_ode_module_importable(self):
        import phlearn.phsystems.ode as phsys  # noqa: F401
        assert phsys is not None

    def test_phnns_module_importable(self):
        import phlearn.phnns as phnn  # noqa: F401
        assert phnn is not None

    def test_pseudo_hamiltonian_system_class_available(self):
        import phlearn.phsystems.ode as phsys
        assert hasattr(phsys, "PseudoHamiltonianSystem")

    def test_generate_dataset_available(self):
        import phlearn.phnns as phnn
        assert hasattr(phnn, "generate_dataset")

    def test_phnn_class_available(self):
        import phlearn.phnns as phnn
        assert hasattr(phnn, "PseudoHamiltonianNN")

    def test_train_function_available(self):
        import phlearn.phnns as phnn
        assert hasattr(phnn, "train")

    def test_baseline_nn_available(self):
        import phlearn.phnns as phnn
        assert hasattr(phnn, "BaselineNN")

    def test_dynamic_system_nn_available(self):
        import phlearn.phnns as phnn
        assert hasattr(phnn, "DynamicSystemNN")

    def test_external_forces_nn_available(self):
        import phlearn.phnns as phnn
        assert hasattr(phnn, "ExternalForcesNN")

    def test_r_estimator_available(self):
        import phlearn.phnns as phnn
        assert hasattr(phnn, "R_estimator")


# ---------------------------------------------------------------------------
# 2. PseudoHamiltonianSystem construction
# ---------------------------------------------------------------------------

class TestPseudoHamiltonianSystemConstruction:
    def setup_method(self):
        import phlearn.phsystems.ode as phsys
        self.phsys = phsys

    def test_creates_system_with_hamiltonian_and_grad(self):
        system = make_system()
        assert system is not None
        assert system.nstates == 2

    def test_system_has_callable_dissipation(self):
        system = make_system()
        assert callable(system.R)

    def test_system_has_callable_hamiltonian(self):
        system = make_system()
        assert callable(system.H)

    def test_system_has_callable_grad_hamiltonian(self):
        system = make_system()
        assert callable(system.dH)

    def test_raises_without_hamiltonian_or_grad(self):
        with pytest.raises(Exception, match="hamiltonian"):
            self.phsys.PseudoHamiltonianSystem(nstates=2)

    def test_raises_with_odd_nstates_and_no_skew_matrix(self):
        with pytest.raises(Exception):
            self.phsys.PseudoHamiltonianSystem(
                nstates=3,
                grad_hamiltonian=make_grad_hamiltonian(),
            )

    def test_creates_system_with_external_forces(self):
        system = make_system(external_forces=external_force)
        assert system.external_forces is not None

    def test_dissipation_matrix_stored_correctly(self):
        damping = 0.3
        R = make_dissipation(damping)
        system = make_system(damping)
        np.testing.assert_allclose(system.dissipation_matrix, R)

    def test_default_initial_condition_sampler_works(self):
        system = make_system()
        x0 = system._initial_condition_sampler()
        assert x0.shape == (2,)

    def test_x_dot_output_shape(self):
        system = make_system()
        x = np.array([[1.0, 0.5]])
        t = np.array([[0.0]])
        xdot = system.x_dot(x, t)
        assert xdot.shape == x.shape


# ---------------------------------------------------------------------------
# 3. generate_dataset output shapes
# ---------------------------------------------------------------------------

class TestGenerateDataset:
    def setup_method(self):
        import phlearn.phnns as phnn
        self.phnn = phnn
        self.system = make_system()
        self.ntrajectories = 3
        self.t_axis = make_t_axis(tmax=2.0, dt=0.1)   # shape (21,) 1D
        self.T = len(self.t_axis)

    def test_returns_tuple_of_two_elements(self):
        result = self.phnn.generate_dataset(
            self.system, self.ntrajectories, self.t_axis
        )
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_traindata_is_tuple_of_six_tensors(self):
        traindata, _ = self.phnn.generate_dataset(
            self.system, self.ntrajectories, self.t_axis
        )
        assert isinstance(traindata, tuple)
        assert len(traindata) == 6

    def test_x_start_shape(self):
        traindata, _ = self.phnn.generate_dataset(
            self.system, self.ntrajectories, self.t_axis
        )
        x_start = traindata[0]
        expected_rows = self.ntrajectories * (self.T - 1)
        assert x_start.shape == (expected_rows, 2)

    def test_x_end_shape(self):
        traindata, _ = self.phnn.generate_dataset(
            self.system, self.ntrajectories, self.t_axis
        )
        x_end = traindata[1]
        expected_rows = self.ntrajectories * (self.T - 1)
        assert x_end.shape == (expected_rows, 2)

    def test_t_start_shape(self):
        traindata, _ = self.phnn.generate_dataset(
            self.system, self.ntrajectories, self.t_axis
        )
        t_start = traindata[2]
        expected_rows = self.ntrajectories * (self.T - 1)
        assert t_start.shape == (expected_rows, 1)

    def test_dxdt_shape(self):
        traindata, dxdt = self.phnn.generate_dataset(
            self.system, self.ntrajectories, self.t_axis
        )
        expected_rows = self.ntrajectories * (self.T - 1)
        assert dxdt.shape == (expected_rows, 2)

    def test_returns_tensors(self):
        traindata, dxdt = self.phnn.generate_dataset(
            self.system, self.ntrajectories, self.t_axis
        )
        assert isinstance(traindata[0], torch.Tensor)
        assert isinstance(dxdt, torch.Tensor)

    def test_no_nans_in_dataset(self):
        traindata, dxdt = self.phnn.generate_dataset(
            self.system, self.ntrajectories, self.t_axis, seed=42
        )
        for i, tensor in enumerate(traindata):
            assert not torch.isnan(tensor).any(), f"NaN in traindata[{i}]"
        assert not torch.isnan(dxdt).any(), "NaN in dxdt"

    def test_zero_trajectories_returns_none(self):
        result = self.phnn.generate_dataset(
            self.system, 0, self.t_axis
        )
        assert result is None

    def test_dt_tensor_has_correct_shape(self):
        traindata, _ = self.phnn.generate_dataset(
            self.system, self.ntrajectories, self.t_axis
        )
        dt = traindata[4]
        expected_rows = self.ntrajectories * (self.T - 1)
        assert dt.shape == (expected_rows, 1)


# ---------------------------------------------------------------------------
# 4. PHNN model trains and loss converges
# ---------------------------------------------------------------------------

class TestPHNNTraining:
    def setup_method(self):
        import phlearn.phnns as phnn
        self.phnn = phnn
        self.system = make_system()
        self.t_axis = make_t_axis(tmax=5.0, dt=0.1)
        self.traindata, self.dxdt = phnn.generate_dataset(
            self.system, ntrajectories=5, t_sample=self.t_axis, seed=1
        )
        self.nstates = 2
        self.states_dampened = [False, True]

    def test_phnn_instantiates(self):
        model = self.phnn.PseudoHamiltonianNN(
            self.nstates,
            dissipation_est=self.phnn.R_estimator(self.states_dampened),
        )
        assert model is not None

    def test_train_returns_model_and_loss(self):
        model = self.phnn.PseudoHamiltonianNN(
            self.nstates,
            dissipation_est=self.phnn.R_estimator(self.states_dampened),
        )
        result = self.phnn.train(
            model,
            integrator="midpoint",
            traindata=(self.traindata, self.dxdt),
            epochs=2,
            batch_size=32,
            verbose=False,
        )
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_train_loss_decreases_over_epochs(self):
        """Loss after 30 epochs should be lower than after 1 epoch."""
        traindata, dxdt = self.phnn.generate_dataset(
            self.system, ntrajectories=10, t_sample=self.t_axis, seed=2
        )

        torch.manual_seed(0)
        model_few = self.phnn.PseudoHamiltonianNN(
            self.nstates,
            dissipation_est=self.phnn.R_estimator(self.states_dampened),
        )
        self.phnn.train(
            model_few,
            integrator="midpoint",
            traindata=(traindata, dxdt),
            epochs=1,
            batch_size=64,
            verbose=False,
        )

        torch.manual_seed(0)
        model_many = self.phnn.PseudoHamiltonianNN(
            self.nstates,
            dissipation_est=self.phnn.R_estimator(self.states_dampened),
        )
        self.phnn.train(
            model_many,
            integrator="midpoint",
            traindata=(traindata, dxdt),
            epochs=30,
            batch_size=64,
            verbose=False,
        )

        loss_few = self.phnn.compute_validation_loss(
            model_few, "midpoint", (traindata, dxdt)
        )
        loss_many = self.phnn.compute_validation_loss(
            model_many, "midpoint", (traindata, dxdt)
        )
        assert loss_many < loss_few, (
            f"Expected loss to decrease: {loss_many:.6f} should be < {loss_few:.6f}"
        )

    def test_trained_model_has_parameters(self):
        model = self.phnn.PseudoHamiltonianNN(
            self.nstates,
            dissipation_est=self.phnn.R_estimator(self.states_dampened),
        )
        params = list(model.parameters())
        assert len(params) > 0

    def test_r_matrix_is_retrievable_after_training(self):
        model = self.phnn.PseudoHamiltonianNN(
            self.nstates,
            dissipation_est=self.phnn.R_estimator(self.states_dampened),
        )
        self.phnn.train(
            model,
            integrator="midpoint",
            traindata=(self.traindata, self.dxdt),
            epochs=5,
            batch_size=32,
            verbose=False,
        )
        R_matrix = model.R().detach().numpy()
        assert R_matrix.shape == (self.nstates, self.nstates)


# ---------------------------------------------------------------------------
# 5. Baseline model trains
# ---------------------------------------------------------------------------

class TestBaselineTraining:
    def setup_method(self):
        import phlearn.phnns as phnn
        self.phnn = phnn
        self.system = make_system()
        t_axis = make_t_axis(tmax=5.0, dt=0.1)
        self.traindata, self.dxdt = phnn.generate_dataset(
            self.system, ntrajectories=5, t_sample=t_axis, seed=3
        )
        self.nstates = 2

    def test_baseline_nn_instantiates(self):
        baseline_nn = self.phnn.BaselineNN(self.nstates, hidden_dim=50)
        assert baseline_nn is not None

    def test_dynamic_system_nn_with_baseline_instantiates(self):
        baseline_nn = self.phnn.BaselineNN(self.nstates, hidden_dim=50)
        model = self.phnn.DynamicSystemNN(self.nstates, rhs_model=baseline_nn)
        assert model is not None

    def test_baseline_model_trains_without_error(self):
        baseline_nn = self.phnn.BaselineNN(self.nstates, hidden_dim=50)
        model = self.phnn.DynamicSystemNN(self.nstates, rhs_model=baseline_nn)
        result = self.phnn.train(
            model,
            integrator=False,
            traindata=(self.traindata, self.dxdt),
            epochs=5,
            batch_size=32,
            verbose=False,
        )
        assert result is not None

    def test_baseline_train_returns_tuple(self):
        baseline_nn = self.phnn.BaselineNN(self.nstates, hidden_dim=50)
        model = self.phnn.DynamicSystemNN(self.nstates, rhs_model=baseline_nn)
        result = self.phnn.train(
            model,
            integrator=False,
            traindata=(self.traindata, self.dxdt),
            epochs=3,
            batch_size=32,
            verbose=False,
        )
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_baseline_nn_output_shape(self):
        baseline_nn = self.phnn.BaselineNN(self.nstates, hidden_dim=50)
        baseline_nn.eval()
        x = torch.randn(10, self.nstates)
        t = torch.randn(10, 1)
        out = baseline_nn(x=x, t=t)
        assert out.shape == (10, self.nstates)


# ---------------------------------------------------------------------------
# 6. PHNN recovers damping constant within tolerance
# ---------------------------------------------------------------------------

class TestDampingRecovery:
    """After sufficient training the learned R[1,1] should be within 0.1 of 0.3."""

    def test_phnn_recovers_damping_within_tolerance(self):
        import phlearn.phnns as phnn

        true_damping = 0.3
        tolerance = 0.1

        system = make_system(damping=true_damping)
        t_axis = make_t_axis(tmax=10.0, dt=0.1)
        traindata, dxdt = phnn.generate_dataset(
            system, ntrajectories=20, t_sample=t_axis, seed=7
        )

        states_dampened = [False, True]
        torch.manual_seed(42)
        model = phnn.PseudoHamiltonianNN(
            2,
            dissipation_est=phnn.R_estimator(states_dampened),
        )
        phnn.train(
            model,
            integrator="midpoint",
            traindata=(traindata, dxdt),
            epochs=100,
            batch_size=128,
            verbose=False,
        )

        learned_R = model.R().detach().numpy()
        learned_damping = learned_R[1, 1]
        assert abs(learned_damping - true_damping) < tolerance, (
            f"Learned damping {learned_damping:.4f} is not within {tolerance} "
            f"of true damping {true_damping}"
        )


# ---------------------------------------------------------------------------
# 7. External force function handles both (1,1) and (T,) shaped t inputs
# ---------------------------------------------------------------------------

class TestExternalForceShapes:
    def test_handles_1x1_t_shape(self):
        """During ODE integration, t arrives as shape (1, 1)."""
        x = np.array([[0.5, 0.3]])   # (1, 2)
        t = np.array([[1.0]])         # (1, 1)
        result = external_force(x, t)
        assert result is not None
        assert not np.isnan(result).any()

    def test_handles_flat_t_shape(self):
        """During batch processing, t arrives as shape (T,)."""
        T = 10
        x = np.random.randn(T, 2)
        t = np.linspace(0, 1, T)     # (T,)
        result = external_force(x, t)
        assert result is not None
        assert not np.isnan(result).any()

    def test_1x1_result_no_nans(self):
        x = np.array([[0.5, 0.3]])
        t = np.array([[1.0]])
        result = external_force(x, t)
        assert not np.isnan(np.asarray(result)).any()

    def test_force_is_zero_in_position_state(self):
        """Force should only affect momentum state (index 1), not position (index 0)."""
        x = np.array([[0.5, 0.3]])
        t = np.array([[1.0]])
        result = external_force(x, t)
        result_arr = np.asarray(result).flatten()
        assert result_arr[0] == pytest.approx(0.0)

    def test_force_matches_expected_sine_value(self):
        """f(x, t) at t=1.0 should equal 0.5*sin(pi/2 * 1.0) = 0.5."""
        x = np.array([[0.0, 0.0]])
        t = np.array([[1.0]])
        result = external_force(x, t)
        expected_momentum_force = 0.5 * np.sin(np.pi / 2.0 * 1.0)
        result_arr = np.asarray(result).flatten()
        assert result_arr[1] == pytest.approx(expected_momentum_force, abs=1e-6)

    def test_batch_t_result_has_correct_rows(self):
        T = 5
        x = np.zeros((T, 2))
        t = np.linspace(0, 1, T)
        result = external_force(x, t)
        assert np.asarray(result).shape[0] == T

    def test_handles_scalar_t(self):
        """Scalar t should not crash."""
        x = np.array([[0.5, 0.3]])
        t = 1.0
        result = external_force(x, t)
        assert result is not None
        assert not np.isnan(np.asarray(result)).any()


# ---------------------------------------------------------------------------
# 8. Forced system data generation works
# ---------------------------------------------------------------------------

class TestForcedSystemDataGeneration:
    def setup_method(self):
        import phlearn.phnns as phnn
        self.phnn = phnn
        self.system = make_system(external_forces=external_force)
        self.t_axis = make_t_axis(tmax=3.0, dt=0.1)   # 1D
        self.ntrajectories = 3
        self.T = len(self.t_axis)

    def test_forced_dataset_generates_without_error(self):
        result = self.phnn.generate_dataset(
            self.system, self.ntrajectories, self.t_axis, seed=5
        )
        assert result is not None

    def test_forced_dataset_has_correct_structure(self):
        traindata, dxdt = self.phnn.generate_dataset(
            self.system, self.ntrajectories, self.t_axis, seed=5
        )
        assert len(traindata) == 6

    def test_forced_dataset_x_start_shape(self):
        traindata, dxdt = self.phnn.generate_dataset(
            self.system, self.ntrajectories, self.t_axis, seed=5
        )
        expected_rows = self.ntrajectories * (self.T - 1)
        assert traindata[0].shape == (expected_rows, 2)

    def test_forced_dataset_no_nans(self):
        traindata, dxdt = self.phnn.generate_dataset(
            self.system, self.ntrajectories, self.t_axis, seed=5
        )
        for i, tensor in enumerate(traindata):
            assert not torch.isnan(tensor).any(), f"NaN in traindata[{i}]"
        assert not torch.isnan(dxdt).any(), "NaN in dxdt"

    def test_forced_dataset_dxdt_shape(self):
        traindata, dxdt = self.phnn.generate_dataset(
            self.system, self.ntrajectories, self.t_axis, seed=5
        )
        expected_rows = self.ntrajectories * (self.T - 1)
        assert dxdt.shape == (expected_rows, 2)


# ---------------------------------------------------------------------------
# 9. PHNN with external forces trains successfully
# ---------------------------------------------------------------------------

class TestPHNNWithExternalForces:
    def setup_method(self):
        import phlearn.phnns as phnn
        self.phnn = phnn
        self.system = make_system(external_forces=external_force)
        t_axis = make_t_axis(tmax=5.0, dt=0.1)
        self.traindata, self.dxdt = phnn.generate_dataset(
            self.system, ntrajectories=5, t_sample=t_axis, seed=6
        )
        self.nstates = 2

    def _make_model(self):
        states_dampened = [False, True]
        ext_forces_nn = self.phnn.ExternalForcesNN(
            nstates=2,
            noutputs=1,
            external_forces_filter=[0, 1],
            hidden_dim=50,
            timedependent=True,
            statedependent=False,
        )
        return self.phnn.PseudoHamiltonianNN(
            self.nstates,
            dissipation_est=self.phnn.R_estimator(states_dampened),
            external_forces_est=ext_forces_nn,
        )

    def test_phnn_with_external_forces_instantiates(self):
        model = self._make_model()
        assert model is not None

    def test_phnn_with_external_forces_trains_without_error(self):
        model = self._make_model()
        result = self.phnn.train(
            model,
            integrator="midpoint",
            traindata=(self.traindata, self.dxdt),
            epochs=5,
            batch_size=32,
            verbose=False,
        )
        assert result is not None

    def test_phnn_with_external_forces_train_returns_tuple(self):
        model = self._make_model()
        result = self.phnn.train(
            model,
            integrator="midpoint",
            traindata=(self.traindata, self.dxdt),
            epochs=3,
            batch_size=32,
            verbose=False,
        )
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_external_forces_nn_instantiates_with_filter(self):
        ext_forces_nn = self.phnn.ExternalForcesNN(
            nstates=2,
            noutputs=1,
            external_forces_filter=[0, 1],
            hidden_dim=50,
            timedependent=True,
            statedependent=False,
        )
        assert ext_forces_nn is not None


# ---------------------------------------------------------------------------
# 10. Long-horizon trajectory simulation does not crash or produce NaN
# ---------------------------------------------------------------------------

class TestLongHorizonSimulation:
    def setup_method(self):
        import phlearn.phnns as phnn
        self.phnn = phnn

        system = make_system()
        t_axis_train = make_t_axis(tmax=5.0, dt=0.1)
        traindata, dxdt = phnn.generate_dataset(
            system, ntrajectories=10, t_sample=t_axis_train, seed=8
        )

        states_dampened = [False, True]
        torch.manual_seed(0)
        self.model = phnn.PseudoHamiltonianNN(
            2,
            dissipation_est=phnn.R_estimator(states_dampened),
        )
        phnn.train(
            self.model,
            integrator="midpoint",
            traindata=(traindata, dxdt),
            epochs=10,
            batch_size=64,
            verbose=False,
        )
        self.t_long = np.linspace(0, 50, 501)
        self.x0 = np.array([1.0, 0.0])

    def test_long_simulation_does_not_raise(self):
        xs, _ = self.model.simulate_trajectory(
            integrator=False,
            t_sample=self.t_long,
            x0=self.x0,
        )
        assert xs is not None

    def test_long_simulation_no_nans(self):
        xs, _ = self.model.simulate_trajectory(
            integrator=False,
            t_sample=self.t_long,
            x0=self.x0,
        )
        assert not np.isnan(xs).any(), "Long-horizon simulation produced NaN values"

    def test_long_simulation_no_infs(self):
        xs, _ = self.model.simulate_trajectory(
            integrator=False,
            t_sample=self.t_long,
            x0=self.x0,
        )
        assert not np.isinf(xs).any(), "Long-horizon simulation produced Inf values"

    def test_long_simulation_correct_length(self):
        T = len(self.t_long)
        xs, _ = self.model.simulate_trajectory(
            integrator=False,
            t_sample=self.t_long,
            x0=self.x0,
        )
        assert xs.shape[0] == T


# ---------------------------------------------------------------------------
# 11. simulate_trajectory output shapes
# ---------------------------------------------------------------------------

class TestSimulateTrajectoryOutputShapes:
    def setup_method(self):
        import phlearn.phnns as phnn
        self.phnn = phnn

        system = make_system()
        t_axis_train = make_t_axis(tmax=5.0, dt=0.1)
        traindata, dxdt = phnn.generate_dataset(
            system, ntrajectories=5, t_sample=t_axis_train, seed=9
        )

        states_dampened = [False, True]
        torch.manual_seed(1)
        self.model = phnn.PseudoHamiltonianNN(
            2,
            dissipation_est=phnn.R_estimator(states_dampened),
        )
        phnn.train(
            self.model,
            integrator="midpoint",
            traindata=(traindata, dxdt),
            epochs=5,
            batch_size=64,
            verbose=False,
        )
        self.nstates = 2

    def test_simulate_trajectory_with_scipy_returns_two_elements(self):
        t_sample = np.linspace(0, 5, 51)
        x0 = np.array([1.0, 0.0])
        result = self.model.simulate_trajectory(
            integrator=False,
            t_sample=t_sample,
            x0=x0,
        )
        assert len(result) == 2

    def test_simulate_trajectory_x_shape_scipy(self):
        T = 51
        t_sample = np.linspace(0, 5, T)
        x0 = np.array([1.0, 0.0])
        xs, _ = self.model.simulate_trajectory(
            integrator=False,
            t_sample=t_sample,
            x0=x0,
        )
        assert xs.shape == (T, self.nstates)

    def test_simulate_trajectory_x_shape_rk4(self):
        T = 51
        t_sample = np.linspace(0, 5, T)
        x0 = np.array([1.0, 0.0])
        xs, _ = self.model.simulate_trajectory(
            integrator="rk4",
            t_sample=t_sample,
            x0=x0,
        )
        assert xs.shape == (T, self.nstates)

    def test_simulate_trajectory_x_shape_euler(self):
        T = 51
        t_sample = np.linspace(0, 5, T)
        x0 = np.array([1.0, 0.0])
        xs, _ = self.model.simulate_trajectory(
            integrator="euler",
            t_sample=t_sample,
            x0=x0,
        )
        assert xs.shape == (T, self.nstates)

    def test_simulate_trajectory_us_is_none_without_controller(self):
        t_sample = np.linspace(0, 2, 21)
        x0 = np.array([1.0, 0.0])
        _, us = self.model.simulate_trajectory(
            integrator=False,
            t_sample=t_sample,
            x0=x0,
        )
        assert us is None

    def test_simulate_trajectory_no_nans(self):
        t_sample = np.linspace(0, 5, 51)
        x0 = np.array([1.0, 0.0])
        xs, _ = self.model.simulate_trajectory(
            integrator=False,
            t_sample=t_sample,
            x0=x0,
        )
        assert not np.isnan(xs).any()

    def test_initial_condition_preserved_scipy(self):
        """First row of xs should be very close to x0 when using solve_ivp."""
        t_sample = np.linspace(0, 5, 51)
        x0 = np.array([1.0, 0.0])
        xs, _ = self.model.simulate_trajectory(
            integrator=False,
            t_sample=t_sample,
            x0=x0,
        )
        np.testing.assert_allclose(xs[0], x0, atol=1e-5)

    def test_nstates_dimension_is_correct(self):
        T = 31
        t_sample = np.linspace(0, 3, T)
        x0 = np.array([0.5, -0.5])
        xs, _ = self.model.simulate_trajectory(
            integrator=False,
            t_sample=t_sample,
            x0=x0,
        )
        assert xs.shape[1] == self.nstates
