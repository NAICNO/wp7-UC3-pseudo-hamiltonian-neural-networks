"""
Tests for phlearn.phnns.ode_models:
- BaseNN
- HamiltonianNN
- BaselineSplitNN
- R_NN
- R_estimator.get_parameters
- ExternalForcesNN edge cases
"""

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# BaseNN
# ---------------------------------------------------------------------------

class TestBaseNN:
    def setup_method(self):
        from phlearn.phnns.ode_models import BaseNN
        self.BaseNN = BaseNN

    def test_state_and_time_dependent_forward(self):
        net = self.BaseNN(nstates=4, noutputs=2, hidden_dim=16,
                          timedependent=True, statedependent=True)
        x = torch.randn(10, 4)
        t = torch.randn(10, 1)
        out = net(x=x, t=t)
        assert out.shape == (10, 2)

    def test_time_only_dependent_forward(self):
        net = self.BaseNN(nstates=4, noutputs=2, hidden_dim=16,
                          timedependent=True, statedependent=False)
        t = torch.randn(10, 1)
        out = net(x=None, t=t)
        assert out.shape == (10, 2)

    def test_state_only_dependent_forward(self):
        net = self.BaseNN(nstates=4, noutputs=2, hidden_dim=16,
                          timedependent=False, statedependent=True)
        x = torch.randn(10, 4)
        out = net(x=x, t=None)
        assert out.shape == (10, 2)

    def test_neither_state_nor_time_dependent_forward(self):
        """When neither state nor time is used, returns a trainable parameter."""
        net = self.BaseNN(nstates=4, noutputs=3, hidden_dim=16,
                          timedependent=False, statedependent=False)
        out = net(x=None, t=None)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (3,)

    def test_output_no_nans(self):
        net = self.BaseNN(nstates=2, noutputs=2, hidden_dim=32,
                          timedependent=True, statedependent=True)
        x = torch.randn(20, 2)
        t = torch.randn(20, 1)
        out = net(x=x, t=t)
        assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# HamiltonianNN
# ---------------------------------------------------------------------------

class TestHamiltonianNN:
    def setup_method(self):
        from phlearn.phnns.ode_models import HamiltonianNN
        self.HamiltonianNN = HamiltonianNN

    def test_output_shape_is_nsamples_by_1(self):
        net = self.HamiltonianNN(nstates=4)
        x = torch.randn(8, 4)
        out = net(x=x, t=None)
        assert out.shape == (8, 1)

    def test_custom_hidden_dim(self):
        net = self.HamiltonianNN(nstates=2, hidden_dim=50)
        x = torch.randn(5, 2)
        out = net(x=x)
        assert out.shape == (5, 1)

    def test_output_is_differentiable(self):
        net = self.HamiltonianNN(nstates=2)
        x = torch.randn(4, 2, requires_grad=True)
        H = net(x=x)
        grad = torch.autograd.grad(H.sum(), x)[0]
        assert grad.shape == x.shape

    def test_no_nans_in_output(self):
        net = self.HamiltonianNN(nstates=4, hidden_dim=64)
        x = torch.randn(10, 4)
        out = net(x=x)
        assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# R_NN
# ---------------------------------------------------------------------------

class TestRNN:
    """
    R_NN returns flat outputs:
    - diagonal=False: (nsamples, nstates**2)
    - diagonal=True:  (nsamples, nstates)
    These are intended to be reshaped by the calling code when needed.
    """

    def setup_method(self):
        from phlearn.phnns.ode_models import R_NN
        self.R_NN = R_NN

    def test_full_matrix_output_shape(self):
        nstates = 3
        net = self.R_NN(nstates=nstates, hidden_dim=32, diagonal=False)
        x = torch.randn(5, nstates)
        out = net(x)
        # Returns flat (nsamples, nstates**2)
        assert out.shape == (5, nstates ** 2)

    def test_diagonal_output_shape(self):
        nstates = 3
        net = self.R_NN(nstates=nstates, hidden_dim=32, diagonal=True)
        x = torch.randn(5, nstates)
        out = net(x)
        # Returns (nsamples, nstates) for diagonal mode
        assert out.shape == (5, nstates)

    def test_full_matrix_batch_size_1(self):
        nstates = 2
        net = self.R_NN(nstates=nstates, hidden_dim=16, diagonal=False)
        x = torch.randn(1, nstates)
        out = net(x)
        assert out.shape == (1, nstates ** 2)

    def test_no_nans_in_output_full(self):
        net = self.R_NN(nstates=4, hidden_dim=64, diagonal=False)
        x = torch.randn(8, 4)
        out = net(x)
        assert not torch.isnan(out).any()

    def test_no_nans_in_output_diagonal(self):
        net = self.R_NN(nstates=4, hidden_dim=64, diagonal=True)
        x = torch.randn(8, 4)
        out = net(x)
        assert not torch.isnan(out).any()

    def test_has_trainable_parameters(self):
        net = self.R_NN(nstates=2, hidden_dim=16, diagonal=False)
        params = list(net.parameters())
        assert len(params) > 0


# ---------------------------------------------------------------------------
# R_estimator
# ---------------------------------------------------------------------------

class TestREstimator:
    def setup_method(self):
        from phlearn.phnns.ode_models import R_estimator
        self.R_estimator = R_estimator

    def test_all_states_dampened(self):
        est = self.R_estimator([True, True])
        R = est()
        assert R.shape == (2, 2)

    def test_no_states_dampened(self):
        """All states undamped: R matrix should be all zeros."""
        est = self.R_estimator([False, False])
        R = est()
        assert torch.allclose(R, torch.zeros(2, 2))

    def test_get_parameters_shape(self):
        est = self.R_estimator([False, True])
        params = est.get_parameters()
        assert params.shape == (1,)

    def test_get_parameters_all_dampened(self):
        est = self.R_estimator([True, True, True])
        params = est.get_parameters()
        assert params.shape == (3,)

    def test_forward_returns_nstates_by_nstates(self):
        nstates = 4
        est = self.R_estimator([False, True, False, True])
        R = est()
        assert R.shape == (nstates, nstates)

    def test_r_estimator_trainable(self):
        """Parameters should be trainable (require grad)."""
        est = self.R_estimator([False, True])
        params = list(est.parameters())
        assert len(params) > 0
        assert all(p.requires_grad for p in params)


# ---------------------------------------------------------------------------
# BaselineSplitNN
# ---------------------------------------------------------------------------

class TestBaselineSplitNN:
    def setup_method(self):
        from phlearn.phnns.ode_models import BaselineSplitNN
        self.BaselineSplitNN = BaselineSplitNN

    def test_instantiates_with_default_outputs(self):
        net = self.BaselineSplitNN(nstates=4, hidden_dim=32)
        assert net is not None

    def test_output_shape(self):
        net = self.BaselineSplitNN(nstates=4, hidden_dim=32)
        x = torch.randn(6, 4)
        t = torch.randn(6, 1)
        out = net(x, t)
        assert out.shape == (6, 4)

    def test_output_is_sum_of_two_networks(self):
        """Output should equal sum of state network and time network."""
        net = self.BaselineSplitNN(nstates=2, hidden_dim=16)
        x = torch.randn(4, 2)
        t = torch.randn(4, 1)
        out = net(x, t)
        out_x = net.network_x(x, t)
        out_t = net.network_t(x, t)
        torch.testing.assert_close(out, out_x + out_t)

    def test_no_nans_in_output(self):
        net = self.BaselineSplitNN(nstates=3, hidden_dim=32)
        x = torch.randn(8, 3)
        t = torch.randn(8, 1)
        out = net(x, t)
        assert not torch.isnan(out).any()

    def test_custom_noutputs_with_filter(self):
        net = self.BaselineSplitNN(
            nstates=4, hidden_dim=16,
            noutputs_x=2,
            external_forces_filter_x=[1, 0, 1, 0],
        )
        x = torch.randn(5, 4)
        t = torch.randn(5, 1)
        out = net(x, t)
        assert out.shape == (5, 4)


# ---------------------------------------------------------------------------
# ExternalForcesNN edge cases
# ---------------------------------------------------------------------------

class TestExternalForcesNNEdgeCases:
    def setup_method(self):
        from phlearn.phnns.ode_models import ExternalForcesNN
        self.ExternalForcesNN = ExternalForcesNN

    def test_filter_as_2d_matrix(self):
        """external_forces_filter can be a 2D (nstates, noutputs) matrix."""
        filter_matrix = np.array([[1, 0], [0, 0], [0, 1]])  # (3, 2)
        net = self.ExternalForcesNN(
            nstates=3, noutputs=2, hidden_dim=16,
            timedependent=True, statedependent=False,
            external_forces_filter=filter_matrix,
        )
        t = torch.randn(5, 1)
        out = net(x=None, t=t)
        assert out.shape == (5, 3)

    def test_no_filter_noutputs_equals_nstates(self):
        """When filter is None, noutputs must equal nstates."""
        net = self.ExternalForcesNN(
            nstates=2, noutputs=2, hidden_dim=16,
            timedependent=True, statedependent=False,
            external_forces_filter=None,
        )
        t = torch.randn(4, 1)
        out = net(x=None, t=t)
        assert out.shape == (4, 2)

    def test_filter_vector_selects_states(self):
        """A vector filter [0, 1, 0] means force only on state 1."""
        net = self.ExternalForcesNN(
            nstates=3, noutputs=1, hidden_dim=16,
            timedependent=True, statedependent=False,
            external_forces_filter=[0, 1, 0],
        )
        t = torch.randn(4, 1)
        out = net(x=None, t=t)
        assert out.shape == (4, 3)
        # States 0 and 2 should always be zero
        assert torch.allclose(out[:, 0], torch.zeros(4))
        assert torch.allclose(out[:, 2], torch.zeros(4))

    def test_neither_state_nor_time_dependent(self):
        """No-input variant should return correct shape."""
        net = self.ExternalForcesNN(
            nstates=2, noutputs=2, hidden_dim=16,
            timedependent=False, statedependent=False,
            external_forces_filter=None,
        )
        out = net(x=None, t=None)
        # Should return a (2,) parameter mapped through identity filter -> (2,)
        assert out is not None
