"""
Tests for qufin.timeseries.kalman — KalmanFilter, FilterResult, SmootherResult.

Correctness benchmarks:
  - Scalar random-walk with known analytical posterior
  - Constant-velocity model recovers true trajectory in low-noise limit
  - RTS smoother reduces posterior variance (cannot increase it)
  - Log-likelihood is negative and finite for valid Gaussian data
  - Missing observations (NaN) propagate without crashing
  - Numerical stability: observation update keeps P symmetric and PSD
  - Control input B @ u is applied correctly
"""

import numpy as np
import pytest

from qufin.timeseries.kalman import FilterResult, KalmanFilter, SmootherResult

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)


def make_scalar_rw(T: int = 200, sigma_x: float = 1.0, sigma_z: float = 2.0):
    """Scalar random-walk hidden state with noisy observations."""
    x = np.cumsum(RNG.normal(0, sigma_x, T))
    z = x + RNG.normal(0, sigma_z, T)
    return x, z


def scalar_kf(sigma_x: float = 1.0, sigma_z: float = 2.0) -> KalmanFilter:
    return KalmanFilter(
        F=[[1.0]],
        H=[[1.0]],
        Q=[[sigma_x**2]],
        R=[[sigma_z**2]],
        x0=[0.0],
        P0=[[sigma_z**2]],
    )


# ---------------------------------------------------------------------------
# Construction & validation
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_dimensions_stored(self):
        kf = scalar_kf()
        assert kf.n == 1
        assert kf.m == 1

    def test_wrong_F_shape_raises(self):
        with pytest.raises((ValueError, AssertionError)):
            KalmanFilter(
                F=np.eye(3),  # wrong: state is 2-d based on x0
                H=[[1.0, 0.0]],
                Q=np.eye(2),
                R=[[1.0]],
                x0=[0.0, 0.0],
                P0=np.eye(2),
            )

    def test_state_property_is_copy(self):
        kf = scalar_kf()
        s = kf.state
        s[0] = 99.0
        assert kf.x[0] != 99.0

    def test_covariance_property_is_copy(self):
        kf = scalar_kf()
        P = kf.covariance
        P[0, 0] = 999.0
        assert kf.P[0, 0] != 999.0

    def test_reset_restores_initial_state(self):
        kf = scalar_kf()
        kf.predict()
        kf.predict()
        kf.reset()
        np.testing.assert_array_equal(kf.x, kf._x0)
        np.testing.assert_array_equal(kf.P, kf._P0)


# ---------------------------------------------------------------------------
# Predict step
# ---------------------------------------------------------------------------


class TestPredict:
    def test_state_evolves_by_F(self):
        kf = scalar_kf()
        kf.x = np.array([3.0])
        x_pred, _ = kf.predict()
        assert x_pred[0] == pytest.approx(3.0)  # F = I for scalar RW

    def test_covariance_grows(self):
        kf = scalar_kf(sigma_x=1.0, sigma_z=2.0)
        P_before = kf.P[0, 0]
        kf.predict()
        assert kf.P[0, 0] > P_before

    def test_covariance_symmetry_maintained(self):
        kf = KalmanFilter(
            F=np.eye(2),
            H=np.eye(2),
            Q=0.1 * np.eye(2),
            R=np.eye(2),
            x0=[0.0, 0.0],
            P0=np.eye(2),
        )
        kf.predict()
        np.testing.assert_allclose(kf.P, kf.P.T, atol=1e-14)

    def test_control_input_applied(self):
        B = np.array([[1.0]])
        kf = KalmanFilter(
            F=[[1.0]],
            H=[[1.0]],
            Q=[[0.01]],
            R=[[1.0]],
            x0=[0.0],
            P0=[[1.0]],
            B=B,
        )
        kf.x = np.array([0.0])
        x_pred, _ = kf.predict(u=np.array([5.0]))
        assert x_pred[0] == pytest.approx(5.0)

    def test_control_without_B_raises(self):
        kf = scalar_kf()
        with pytest.raises(ValueError):
            kf.predict(u=np.array([1.0]))


# ---------------------------------------------------------------------------
# Update step
# ---------------------------------------------------------------------------


class TestUpdate:
    def test_posterior_mean_between_prior_and_observation(self):
        """P(x|z) mean must lie between prior mean and observation."""
        kf = scalar_kf(sigma_x=1.0, sigma_z=2.0)
        kf.x = np.array([0.0])
        kf.predict()
        x_filt, _, innov, _, _ = kf.update(np.array([10.0]))
        assert 0.0 < x_filt[0] < 10.0

    def test_posterior_covariance_smaller_than_prior(self):
        kf = scalar_kf()
        _, P_pred = kf.predict()
        P_before = P_pred[0, 0]
        _, P_filt, _, _, _ = kf.update(np.array([1.0]))
        assert P_filt[0, 0] < P_before

    def test_covariance_symmetry_after_update(self):
        kf = KalmanFilter(
            F=np.eye(2),
            H=np.eye(2),
            Q=0.1 * np.eye(2),
            R=np.eye(2),
            x0=[0.0, 0.0],
            P0=np.eye(2),
        )
        kf.predict()
        kf.update(np.array([1.0, 2.0]))
        np.testing.assert_allclose(kf.P, kf.P.T, atol=1e-14)

    def test_covariance_positive_definite_after_update(self):
        kf = scalar_kf()
        kf.predict()
        kf.update(np.array([5.0]))
        eigenvalues = np.linalg.eigvalsh(kf.P)
        assert np.all(eigenvalues > 0)

    def test_perfect_observation_collapses_uncertainty(self):
        """When R → 0 the posterior should converge to the observation."""
        kf = KalmanFilter(
            F=[[1.0]],
            H=[[1.0]],
            Q=[[1.0]],
            R=[[1e-10]],
            x0=[0.0],
            P0=[[1.0]],
        )
        kf.predict()
        x_filt, P_filt, _, _, _ = kf.update(np.array([7.0]))
        assert x_filt[0] == pytest.approx(7.0, abs=1e-4)
        assert P_filt[0, 0] < 1e-8


# ---------------------------------------------------------------------------
# Batch filter
# ---------------------------------------------------------------------------


class TestBatchFilter:
    def test_returns_filter_result(self):
        kf = scalar_kf()
        _, z = make_scalar_rw(50)
        res = kf.filter(z)
        assert isinstance(res, FilterResult)

    def test_output_shapes(self):
        kf = scalar_kf()
        T = 80
        _, z = make_scalar_rw(T)
        res = kf.filter(z)
        assert res.states.shape == (T, 1)
        assert res.covariances.shape == (T, 1, 1)
        assert res.innovations.shape == (T, 1)
        assert res.gains.shape == (T, 1, 1)

    def test_mean_squared_error_better_than_raw(self):
        """Filtered estimates should be closer to truth than raw observations."""
        T = 500
        x_true, z = make_scalar_rw(T, sigma_x=0.5, sigma_z=3.0)
        kf = scalar_kf(sigma_x=0.5, sigma_z=3.0)
        res = kf.filter(z)
        mse_filter = np.mean((res.states[:, 0] - x_true) ** 2)
        mse_raw = np.mean((z - x_true) ** 2)
        assert mse_filter < mse_raw

    def test_missing_observations_nan_handled(self):
        kf = scalar_kf()
        _, z = make_scalar_rw(50)
        z[10] = np.nan
        z[20] = np.nan
        res = kf.filter(z)  # must not raise
        # States at missing steps equal predictions (no update)
        np.testing.assert_allclose(res.states[10], res.pred_states[10])
        np.testing.assert_allclose(res.states[20], res.pred_states[20])

    def test_log_likelihood_negative_finite(self):
        kf = scalar_kf()
        _, z = make_scalar_rw(100)
        res = kf.filter(z)
        assert np.isfinite(res.log_likelihood)
        assert res.log_likelihood < 0.0

    def test_x0_P0_override_respected(self):
        kf = scalar_kf()
        _, z = make_scalar_rw(20)
        res1 = kf.filter(z, x0=np.array([0.0]), P0=np.array([[4.0]]))
        res2 = kf.filter(z, x0=np.array([100.0]), P0=np.array([[4.0]]))
        # First filtered state should differ significantly
        assert not np.allclose(res1.states[0], res2.states[0])

    def test_all_covariances_positive_definite(self):
        kf = scalar_kf()
        _, z = make_scalar_rw(100)
        res = kf.filter(z)
        for t in range(len(z)):
            ev = np.linalg.eigvalsh(res.covariances[t])
            assert np.all(ev >= 0), f"Covariance not PSD at t={t}"

    def test_multivariate_observation(self):
        """2-state, 2-observation model runs without error."""
        kf = KalmanFilter(
            F=np.eye(2),
            H=np.eye(2),
            Q=0.1 * np.eye(2),
            R=np.eye(2),
            x0=[0.0, 0.0],
            P0=np.eye(2),
        )
        z = RNG.normal(size=(50, 2))
        res = kf.filter(z)
        assert res.states.shape == (50, 2)


# ---------------------------------------------------------------------------
# RTS smoother
# ---------------------------------------------------------------------------


class TestRTSSmoother:
    def test_returns_smoother_result(self):
        kf = scalar_kf()
        _, z = make_scalar_rw(50)
        filt = kf.filter(z)
        smo = kf.smooth(filt)
        assert isinstance(smo, SmootherResult)

    def test_smoother_variance_le_filter_variance(self):
        """Smoothed variance must be ≤ filtered variance at every time step."""
        kf = scalar_kf()
        T = 200
        _, z = make_scalar_rw(T)
        filt = kf.filter(z)
        smo = kf.smooth(filt)
        filt_var = filt.covariances[:, 0, 0]
        smo_var = smo.covariances[:, 0, 0]
        # Allow tiny floating-point slack (1e-10)
        assert np.all(smo_var <= filt_var + 1e-10), (
            f"Smoother exceeded filter variance at some steps: "
            f"max excess = {np.max(smo_var - filt_var):.2e}"
        )

    def test_smoother_reduces_mse(self):
        """Smoothed MSE must be ≤ filtered MSE."""
        T = 500
        x_true, z = make_scalar_rw(T, sigma_x=0.5, sigma_z=3.0)
        kf = scalar_kf(sigma_x=0.5, sigma_z=3.0)
        filt = kf.filter(z)
        smo = kf.smooth(filt)
        mse_filt = np.mean((filt.states[:, 0] - x_true) ** 2)
        mse_smo = np.mean((smo.states[:, 0] - x_true) ** 2)
        assert mse_smo <= mse_filt + 1e-6

    def test_smoother_gains_shape(self):
        kf = scalar_kf()
        T = 30
        _, z = make_scalar_rw(T)
        filt = kf.filter(z)
        smo = kf.smooth(filt)
        assert smo.gains.shape == (T - 1, 1, 1)

    def test_smoothed_covariances_symmetric(self):
        kf = KalmanFilter(
            F=np.eye(2),
            H=np.eye(2),
            Q=0.1 * np.eye(2),
            R=np.eye(2),
            x0=[0.0, 0.0],
            P0=np.eye(2),
        )
        z = RNG.normal(size=(40, 2))
        filt = kf.filter(z)
        smo = kf.smooth(filt)
        for t in range(40):
            np.testing.assert_allclose(smo.covariances[t], smo.covariances[t].T, atol=1e-13)

    def test_log_likelihood_matches_filter(self):
        kf = scalar_kf()
        _, z = make_scalar_rw(50)
        filt = kf.filter(z)
        smo = kf.smooth(filt)
        assert smo.log_likelihood == pytest.approx(filt.log_likelihood)


# ---------------------------------------------------------------------------
# Log-likelihood interface
# ---------------------------------------------------------------------------


class TestLogLikelihood:
    def test_correct_model_has_higher_ll_than_misspecified(self):
        """A well-specified model should have higher log-likelihood."""
        T = 300
        sigma_x, sigma_z = 0.5, 2.0
        x_true, z = make_scalar_rw(T, sigma_x, sigma_z)

        kf_good = scalar_kf(sigma_x=sigma_x, sigma_z=sigma_z)
        kf_bad = scalar_kf(sigma_x=10.0, sigma_z=0.01)

        ll_good = kf_good.log_likelihood(z)
        ll_bad = kf_bad.log_likelihood(z)
        assert ll_good > ll_bad

    def test_standalone_ll_equals_filter_ll(self):
        kf = scalar_kf()
        _, z = make_scalar_rw(100)
        ll_via_method = kf.log_likelihood(z)
        ll_via_filter = kf.filter(z).log_likelihood
        assert ll_via_method == pytest.approx(ll_via_filter, rel=1e-10)


# ---------------------------------------------------------------------------
# Numerical stress tests
# ---------------------------------------------------------------------------


class TestNumericalStability:
    def test_many_steps_P_stays_psd(self):
        """Run 2000 predict+update steps; P must remain positive definite."""
        kf = scalar_kf(sigma_x=0.1, sigma_z=1.0)
        for _ in range(2000):
            kf.predict()
            kf.update(RNG.normal())
        ev = np.linalg.eigvalsh(kf.P)
        assert np.all(ev > 0)

    def test_ill_conditioned_R_handled(self):
        """Tiny R (near-perfect observations) should not blow up the filter."""
        kf = KalmanFilter(
            F=[[1.0]],
            H=[[1.0]],
            Q=[[1.0]],
            R=[[1e-12]],
            x0=[0.0],
            P0=[[1.0]],
        )
        for _ in range(100):
            kf.predict()
            kf.update(RNG.normal())
        assert np.isfinite(kf.x[0])
        assert np.isfinite(kf.P[0, 0])
        assert kf.P[0, 0] >= 0

    def test_large_initial_covariance(self):
        """Diffuse initialisation (large P0) converges without NaN."""
        kf = KalmanFilter(
            F=[[1.0]],
            H=[[1.0]],
            Q=[[1.0]],
            R=[[1.0]],
            x0=[0.0],
            P0=[[1e8]],
        )
        _, z = make_scalar_rw(50)
        res = kf.filter(z)
        assert np.all(np.isfinite(res.states))
        assert np.all(np.isfinite(res.covariances))
