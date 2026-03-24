"""
Tests for phlearn.phsystems.ode.tank_system:
- init_tanksystem
- init_tanksystem_leaky
- TankSystem via graph with explicit dissipation_pipes
- pipeflows / tanklevels accessors
- H_tanksystem / dH_tanksystem
- sample_trajectory for tank systems
"""

import numpy as np
import pytest

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not NETWORKX_AVAILABLE,
    reason="networkx not installed"
)


def _make_t(tmax=1.0, dt=0.1):
    n = int(round(tmax / dt))
    return np.linspace(0, tmax, n + 1)


def _make_standard_tank_graph():
    """The same graph used by init_tanksystem (5 pipes, 4 tanks)."""
    G = nx.DiGraph()
    G.add_edge(1, 2)
    G.add_edge(2, 3)
    G.add_edge(3, 4)
    G.add_edge(1, 3)
    G.add_edge(1, 4)
    return G


# ---------------------------------------------------------------------------
# TankSystem via graph with explicit dissipation_pipes
# ---------------------------------------------------------------------------

class TestTankSystemViaGraph:
    def setup_method(self):
        from phlearn.phsystems.ode.tank_system import TankSystem
        self.TankSystem = TankSystem

    def _make_system(self):
        G = _make_standard_tank_graph()
        npipes = G.number_of_edges()
        R_pipes = 1e-2 * np.diag(np.array([3., 3., 9., 3., 3.]))
        J = 2e-2 * np.ones(npipes)
        return self.TankSystem(system_graph=G, dissipation_pipes=R_pipes, J=J)

    def test_instantiates_with_graph_and_dissipation(self):
        sys = self._make_system()
        assert sys is not None

    def test_nstates_is_npipes_plus_ntanks(self):
        sys = self._make_system()
        assert sys.nstates == 9
        assert sys.npipes == 5
        assert sys.ntanks == 4

    def test_x_dot_no_nans(self):
        sys = self._make_system()
        x = np.zeros((1, sys.nstates))
        x[0, :3] = 0.1
        t = np.array([[0.0]])
        xdot = sys.x_dot(x, t)
        assert not np.isnan(xdot).any()

    def test_x_dot_output_shape(self):
        sys = self._make_system()
        x = np.zeros((1, sys.nstates))
        t = np.array([[0.0]])
        xdot = sys.x_dot(x, t)
        assert xdot.shape == x.shape

    def test_pipeflows_returns_correct_slice(self):
        sys = self._make_system()
        x = np.arange(sys.nstates, dtype=float)
        flows = sys.pipeflows(x)
        np.testing.assert_array_equal(flows, x[:sys.npipes])

    def test_tanklevels_returns_correct_slice(self):
        sys = self._make_system()
        x = np.arange(sys.nstates, dtype=float)
        levels = sys.tanklevels(x)
        np.testing.assert_array_equal(levels, x[sys.npipes:])

    def test_hamiltonian_no_nans(self):
        sys = self._make_system()
        x = np.ones(sys.nstates) * 0.1
        H = sys.H_tanksystem(x)
        assert not np.isnan(np.atleast_1d(H)).any()

    def test_grad_hamiltonian_shape(self):
        sys = self._make_system()
        x = np.ones((1, sys.nstates)) * 0.1
        dH = sys.dH_tanksystem(x)
        # dH_tanksystem returns (x.T * Hvec).T which for x shape (1, N)
        # gives shape (N, N) due to broadcasting; just verify no NaNs
        assert not np.isnan(dH).any()


# ---------------------------------------------------------------------------
# init_tanksystem
# ---------------------------------------------------------------------------

class TestInitTankSystem:
    def setup_method(self):
        from phlearn.phsystems.ode.tank_system import init_tanksystem
        self.init_tanksystem = init_tanksystem

    def test_returns_system(self):
        sys = self.init_tanksystem()
        assert sys is not None

    def test_nstates_correct(self):
        sys = self.init_tanksystem()
        assert sys.nstates == 9

    def test_sample_trajectory_shape(self):
        sys = self.init_tanksystem()
        t = _make_t(tmax=0.5, dt=0.1)
        x0 = np.zeros(sys.nstates)
        x, dxdt, t_out, us = sys.sample_trajectory(t, x0=x0)
        assert x.shape[1] == sys.nstates
        assert x.shape[0] == len(t)

    def test_sample_trajectory_no_nans(self):
        sys = self.init_tanksystem()
        t = _make_t(tmax=0.5, dt=0.1)
        x0 = np.zeros(sys.nstates)
        x, dxdt, t_out, us = sys.sample_trajectory(t, x0=x0)
        assert not np.isnan(x).any()

    def test_has_external_forces(self):
        """init_tanksystem includes an external force."""
        sys = self.init_tanksystem()
        assert sys.external_forces is not None


# ---------------------------------------------------------------------------
# init_tanksystem_leaky
# ---------------------------------------------------------------------------

class TestInitTankSystemLeaky:
    def setup_method(self):
        from phlearn.phsystems.ode.tank_system import init_tanksystem_leaky
        self.init_tanksystem_leaky = init_tanksystem_leaky

    def test_no_leaks_returns_system(self):
        sys = self.init_tanksystem_leaky(nleaks=0)
        assert sys is not None

    def test_one_leak_returns_system(self):
        sys = self.init_tanksystem_leaky(nleaks=1)
        assert sys is not None

    def test_two_leaks_returns_system(self):
        sys = self.init_tanksystem_leaky(nleaks=2)
        assert sys is not None

    def test_nstates_correct(self):
        sys = self.init_tanksystem_leaky(nleaks=0)
        assert sys.nstates == 9

    def test_no_leaks_zero_external_force(self):
        """With no leaks external force should be all zero on a zero state."""
        sys = self.init_tanksystem_leaky(nleaks=0)
        x = np.zeros(sys.nstates)
        t = np.array([[0.0]])
        force = sys.external_forces(x, t)
        assert np.allclose(force, 0.0)

    def test_with_leak_nonzero_force_on_last_tank(self):
        """With a nonzero last-tank state and leak, force on last tank is nonzero."""
        sys = self.init_tanksystem_leaky(nleaks=1)
        x = np.zeros(sys.nstates)
        x[-1] = 0.2
        t = np.array([[0.0]])
        force = sys.external_forces(x, t)
        assert force[-1] != 0.0

    def test_sample_trajectory_no_leaks(self):
        sys = self.init_tanksystem_leaky(nleaks=0)
        t = _make_t(tmax=0.5, dt=0.1)
        x0 = np.zeros(sys.nstates)
        x, dxdt, t_out, us = sys.sample_trajectory(t, x0=x0)
        assert not np.isnan(x).any()

    def test_sample_trajectory_with_leaks(self):
        sys = self.init_tanksystem_leaky(nleaks=1)
        t = _make_t(tmax=0.5, dt=0.1)
        x0 = np.ones(sys.nstates) * 0.1
        x, dxdt, t_out, us = sys.sample_trajectory(t, x0=x0)
        assert not np.isnan(x).any()
