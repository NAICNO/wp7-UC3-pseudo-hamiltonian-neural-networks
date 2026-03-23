"""
Tests for phlearn.control.reference:
- ConstantReference
- StepReference
- FixedReference

Note: the phlearn.control module requires optional dependencies (casadi).
All tests are skipped if the module is not available.
"""

import numpy as np
import pytest

# Check whether the control module is available
try:
    from phlearn.control.reference import (
        ConstantReference,
        StepReference,
        FixedReference,
    )
    CONTROL_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    CONTROL_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not CONTROL_AVAILABLE,
    reason="phlearn.control module not installed (requires 'pip install phlearn[control]')"
)


# ---------------------------------------------------------------------------
# ConstantReference
# ---------------------------------------------------------------------------

class TestConstantReference:
    def test_returns_constant_value(self):
        ref = ConstantReference(low=0.0, high=1.0, value=0.5)
        assert ref(0.0) == pytest.approx(0.5)
        assert ref(1.0) == pytest.approx(0.5)
        assert ref(100.0) == pytest.approx(0.5)

    def test_reset_resamples_value_in_range(self):
        ref = ConstantReference(low=2.0, high=5.0, seed=0)
        ref.reset()
        val = ref(0.0)
        assert 2.0 <= val <= 5.0

    def test_reset_with_explicit_value(self):
        ref = ConstantReference(low=0.0, high=1.0, seed=0)
        ref.reset(value=0.7)
        assert ref(0.0) == pytest.approx(0.7)

    def test_history_records_calls(self):
        ref = ConstantReference(low=0.0, high=1.0, value=0.3)
        ref(0.0)
        ref(1.0)
        ref(2.0)
        assert len(ref.history['t']) == 3
        assert len(ref.history['r']) == 3

    def test_reset_clears_history(self):
        ref = ConstantReference(low=0.0, high=1.0, value=0.3)
        ref(0.0)
        ref(1.0)
        ref.reset()
        assert len(ref.history['t']) == 0

    def test_get_reference_data_without_ts(self):
        ref = ConstantReference(low=0.0, high=1.0, value=0.4)
        ref(0.5)
        ref(1.0)
        values, times = ref.get_reference_data()
        assert len(values) == 2
        assert len(times) == 2

    def test_get_reference_data_with_ts(self):
        ref = ConstantReference(low=0.0, high=1.0, value=0.4)
        ts = [0.0, 1.0, 2.0]
        values, returned_ts = ref.get_reference_data(ts=ts)
        assert len(values) == 3
        # All values should equal the constant 0.4
        assert all(v == pytest.approx(0.4) for v in values)


# ---------------------------------------------------------------------------
# StepReference
# ---------------------------------------------------------------------------

class TestStepReference:
    def test_returns_value_within_range(self):
        ref = StepReference(low=0.0, high=1.0, step_interval=1.0, seed=42)
        val = ref(0.5)
        assert 0.0 <= val <= 1.0

    def test_same_interval_returns_same_value(self):
        ref = StepReference(low=0.0, high=1.0, step_interval=1.0, seed=0)
        v1 = ref(0.1)
        v2 = ref(0.9)
        # Both calls are within the first step interval [0, 1)
        assert v1 == pytest.approx(v2)

    def test_different_intervals_do_not_crash(self):
        ref = StepReference(low=0.0, high=1.0, step_interval=1.0, seed=0)
        v1 = ref(0.5)
        v2 = ref(1.5)
        assert v1 is not None
        assert v2 is not None

    def test_reset_clears_history(self):
        ref = StepReference(low=0.0, high=1.0, step_interval=1.0, seed=0)
        ref(0.5)
        ref.reset()
        assert len(ref.history['t']) == 0

    def test_beyond_initial_values_extends(self):
        """Calling at t beyond initial values should not crash."""
        ref = StepReference(low=0.0, high=1.0, step_interval=1.0, seed=1)
        for t in [0.5, 1.5, 2.5, 3.5, 4.5]:
            val = ref(t)
            assert 0.0 <= val <= 1.0


# ---------------------------------------------------------------------------
# FixedReference
# ---------------------------------------------------------------------------

class TestFixedReference:
    def test_returns_closest_value(self):
        ref = FixedReference(
            values=[10.0, 20.0, 30.0],
            timestamps=[0.0, 1.0, 2.0],
        )
        # t=0.1 is closest to timestamp 0.0
        assert ref(0.1) == pytest.approx(10.0)

    def test_exact_timestamp_returns_exact_value(self):
        ref = FixedReference(
            values=[5.0, 15.0],
            timestamps=[0.0, 2.0],
        )
        assert ref(0.0) == pytest.approx(5.0)
        assert ref(2.0) == pytest.approx(15.0)

    def test_midpoint_returns_one_of_two(self):
        ref = FixedReference(
            values=[1.0, 2.0],
            timestamps=[0.0, 2.0],
        )
        # t=1.0 is equidistant, argmin returns first
        val = ref(1.0)
        assert val in (1.0, 2.0)

    def test_history_recorded(self):
        ref = FixedReference(
            values=[1.0, 2.0, 3.0],
            timestamps=[0.0, 1.0, 2.0],
        )
        ref(0.0)
        ref(1.0)
        assert len(ref.history['t']) == 2

    def test_get_reference_data_with_timestamps(self):
        ref = FixedReference(
            values=[100.0, 200.0],
            timestamps=[0.0, 5.0],
        )
        values, ts = ref.get_reference_data(ts=[0.0, 5.0])
        assert values[0] == pytest.approx(100.0)
        assert values[1] == pytest.approx(200.0)
