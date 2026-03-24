"""
Tests for phlearn.utils: to_tensor, midpoint_method, and time_derivative.
"""

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Tests for phlearn.utils.utils.to_tensor
# ---------------------------------------------------------------------------

class TestToTensor:
    """Unit tests for the to_tensor utility function."""

    def setup_method(self):
        from phlearn.utils.utils import to_tensor
        self.to_tensor = to_tensor

    def test_returns_none_for_none_input(self):
        result = self.to_tensor(None)
        assert result is None

    def test_converts_list_to_tensor(self):
        result = self.to_tensor([1.0, 2.0, 3.0])
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3,)

    def test_converts_numpy_array_to_tensor(self):
        arr = np.array([1.0, 2.0])
        result = self.to_tensor(arr)
        assert isinstance(result, torch.Tensor)

    def test_passthrough_existing_tensor(self):
        t = torch.tensor([1.0, 2.0])
        result = self.to_tensor(t)
        assert result is t

    def test_default_dtype_is_float32(self):
        result = self.to_tensor([1.0, 2.0])
        assert result.dtype == torch.float32

    def test_custom_dtype_float64(self):
        result = self.to_tensor([1.0, 2.0], ttype=torch.float64)
        assert result.dtype == torch.float64

    def test_2d_list_converts_correctly(self):
        data = [[1.0, 2.0], [3.0, 4.0]]
        result = self.to_tensor(data)
        assert result.shape == (2, 2)
        assert isinstance(result, torch.Tensor)

    def test_scalar_value_converts_to_tensor(self):
        result = self.to_tensor(3.14)
        assert isinstance(result, torch.Tensor)

    def test_values_preserved_after_conversion(self):
        arr = np.array([1.5, 2.5, 3.5])
        result = self.to_tensor(arr)
        np.testing.assert_allclose(result.numpy(), arr, atol=1e-6)


# ---------------------------------------------------------------------------
# Tests for phlearn.utils.utils.midpoint_method
# ---------------------------------------------------------------------------

class TestMidpointMethod:
    """Unit tests for the midpoint_method Newton solver."""

    def setup_method(self):
        from phlearn.utils.utils import midpoint_method
        self.midpoint_method = midpoint_method

    def _simple_ode(self, u, t):
        """dx/dt = -x  (exponential decay)."""
        return -u

    def _simple_ode_jacobian(self, u, t):
        M = len(u)
        return -np.eye(M)

    def test_returns_array_of_correct_shape(self):
        u = np.array([1.0, 0.0])
        un = u.copy()
        dt = 0.01
        result = self.midpoint_method(
            u, un, 0.0, self._simple_ode, self._simple_ode_jacobian, dt, M=2
        )
        assert result.shape == (2,)

    def test_decay_decreases_value(self):
        """Integrating dx/dt = -x with x0 = 1 should decrease x."""
        u = np.array([1.0])
        un = u.copy()
        dt = 0.1
        result = self.midpoint_method(
            u, un, 0.0, self._simple_ode, self._simple_ode_jacobian, dt, M=1
        )
        assert result[0] < u[0]

    def test_exponential_decay_accuracy(self):
        """Midpoint method should approximate exp(-dt) well for small dt."""
        u = np.array([1.0])
        dt = 0.01
        result = self.midpoint_method(
            u, u.copy(), 0.0, self._simple_ode, self._simple_ode_jacobian, dt, M=1
        )
        expected = np.exp(-dt)
        assert abs(result[0] - expected) < 1e-6

    def test_zero_initial_state_stays_zero(self):
        """For dx/dt = -x with x0 = 0, x should remain 0."""
        u = np.array([0.0])
        result = self.midpoint_method(
            u, u.copy(), 0.0, self._simple_ode, self._simple_ode_jacobian, 0.1, M=1
        )
        assert abs(result[0]) < 1e-12

    def test_max_iter_respected(self):
        """Should return after max_iter even with loose tolerance."""
        u = np.array([1.0])
        result = self.midpoint_method(
            u, u.copy(), 0.0,
            self._simple_ode,
            self._simple_ode_jacobian,
            0.1, M=1, tol=1e-100, max_iter=2
        )
        assert result is not None
        assert result.shape == (1,)


# ---------------------------------------------------------------------------
# Tests for phlearn.utils.derivatives.time_derivative
# ---------------------------------------------------------------------------

class TestTimeDerivative:
    """Tests for the time_derivative dispatcher."""

    def setup_method(self):
        from phlearn.utils.derivatives import time_derivative
        self.time_derivative = time_derivative

    def _make_x_dot(self):
        """Simple x_dot: returns -x (linear decay)."""
        def x_dot(x, t, u=None, xspatial=None):
            return -x
        return x_dot

    def test_euler_integrator_returns_correct_shape(self):
        x = torch.ones(5, 2)
        t = torch.zeros(5, 1)
        result = self.time_derivative('euler', self._make_x_dot(), x, x, t, t, 0.1)
        assert result.shape == x.shape

    def test_false_integrator_same_as_euler(self):
        x = torch.ones(5, 2)
        t = torch.zeros(5, 1)
        xdot_euler = self.time_derivative('euler', self._make_x_dot(), x, x, t, t, 0.1)
        xdot_false = self.time_derivative(False, self._make_x_dot(), x, x, t, t, 0.1)
        torch.testing.assert_close(xdot_euler, xdot_false)

    def test_midpoint_integrator_returns_correct_shape(self):
        x = torch.ones(5, 2)
        t = torch.zeros(5, 1)
        result = self.time_derivative('midpoint', self._make_x_dot(), x, x, t, t, 0.1)
        assert result.shape == x.shape

    def test_rk4_integrator_returns_correct_shape(self):
        x = torch.ones(5, 2)
        t = torch.zeros(5, 1)
        result = self.time_derivative('rk4', self._make_x_dot(), x, x, t, t, 0.1)
        assert result.shape == x.shape

    def test_srk4_integrator_returns_correct_shape(self):
        x = torch.ones(5, 2)
        t = torch.zeros(5, 1)
        result = self.time_derivative('srk4', self._make_x_dot(), x, x, t, t, 0.1)
        assert result.shape == x.shape

    def test_unknown_integrator_raises_value_error(self):
        x = torch.ones(5, 2)
        t = torch.zeros(5, 1)
        with pytest.raises(ValueError, match="Unknown integrator"):
            self.time_derivative('bogus', self._make_x_dot(), x, x, t, t, 0.1)

    def test_midpoint_uses_midpoint_of_x(self):
        """For midpoint, x_dot is evaluated at midpoint of x_start and x_end."""
        calls = []

        def x_dot_recorder(x, t, u=None, xspatial=None):
            calls.append(x.clone() if isinstance(x, torch.Tensor) else x.copy())
            return torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)

        x_start = torch.tensor([[0.0, 0.0]])
        x_end = torch.tensor([[2.0, 2.0]])
        t_start = torch.zeros(1, 1)
        t_end = torch.ones(1, 1)
        self.time_derivative('midpoint', x_dot_recorder, x_start, x_end, t_start, t_end, 1.0)
        assert len(calls) == 1
        expected_mid = torch.tensor([[1.0, 1.0]])
        torch.testing.assert_close(calls[0], expected_mid)

    def test_rk4_no_nans_produced(self):
        x = torch.randn(10, 2)
        t = torch.randn(10, 1)
        result = self.time_derivative('rk4', self._make_x_dot(), x, x, t, t, 0.01)
        assert not torch.isnan(result).any()

    def test_integrator_case_insensitive(self):
        """Integrator string should be case-insensitive."""
        x = torch.ones(3, 2)
        t = torch.zeros(3, 1)
        result = self.time_derivative('RK4', self._make_x_dot(), x, x, t, t, 0.1)
        assert result.shape == x.shape

    def test_cm4_integrator_returns_correct_shape(self):
        x = torch.ones(5, 2)
        t = torch.zeros(5, 1)
        result = self.time_derivative('cm4', self._make_x_dot(), x, x, t, t, 0.1)
        assert result.shape == x.shape

    def test_cs6_integrator_returns_correct_shape(self):
        x = torch.ones(5, 2)
        t = torch.zeros(5, 1)
        result = self.time_derivative('cs6', self._make_x_dot(), x, x, t, t, 0.1)
        assert result.shape == x.shape

    def test_cm4_no_nans(self):
        x = torch.randn(4, 2)
        t = torch.randn(4, 1)
        result = self.time_derivative('cm4', self._make_x_dot(), x, x, t, t, 0.01)
        assert not torch.isnan(result).any()

    def test_cs6_no_nans(self):
        x = torch.randn(4, 2)
        t = torch.randn(4, 1)
        result = self.time_derivative('cs6', self._make_x_dot(), x, x, t, t, 0.01)
        assert not torch.isnan(result).any()
