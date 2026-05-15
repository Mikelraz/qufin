"""
Ornstein-Uhlenbeck (OU) process for financial time series.

Continuous-time SDE
-------------------
    dX_t = θ(μ - X_t) dt + σ dW_t

    θ > 0   mean-reversion speed
    μ       long-run mean
    σ > 0   diffusion coefficient

Discrete-time exact solution (no Euler approximation error)
------------------------------------------------------------
    X_{t+Δ} = μ + (X_t - μ) e^{-θΔ} + ε_t

    ε_t ~ N(0, σ²_ε),   σ²_ε = σ² (1 - e^{-2θΔ}) / (2θ)

This is an AR(1) process:  X_{t+1} = a + b X_t + ε_t
    b = e^{-θΔ},   a = μ(1 - b)

Applications
------------
* Pairs-trading spread modelling
* Interest-rate short-rate models (Vasicek)
* Mean-reversion signal generation
* Parameter estimation via OLS or exact MLE
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class OUFitResult:
    """
    Summary of a completed parameter fit.

    Attributes
    ----------
    theta       Mean-reversion speed  (per unit time)
    mu          Long-run mean
    sigma       Diffusion coefficient
    half_life   ln(2) / θ  — time to revert half the gap to μ
    sigma_eq    σ / √(2θ)  — stationary (equilibrium) std
    log_lik     Log-likelihood of the fitted model on the training series
    method      Estimation method used ('ols' or 'mle')
    n_obs       Number of observations used (= len(series) - 1 transitions)
    """

    theta: float
    mu: float
    sigma: float
    half_life: float
    sigma_eq: float
    log_lik: float
    method: str
    n_obs: int

    def __str__(self) -> str:
        lines = [
            "Ornstein-Uhlenbeck Fit",
            "=" * 35,
            f"  method      : {self.method}",
            f"  n_obs       : {self.n_obs}",
            f"  θ  (speed)  : {self.theta:.6f}",
            f"  μ  (mean)   : {self.mu:.6f}",
            f"  σ  (vol)    : {self.sigma:.6f}",
            f"  half-life   : {self.half_life:.4f}",
            f"  σ_eq        : {self.sigma_eq:.6f}",
            f"  log-lik     : {self.log_lik:.4f}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core model
# ---------------------------------------------------------------------------

class OrnsteinUhlenbeck:
    """
    Ornstein-Uhlenbeck process:  dX_t = θ(μ - X_t)dt + σ dW_t

    Parameters can be supplied at construction *or* estimated from data via
    ``fit()``.  All methods require that the model is parameterised (either
    manually or via ``fit()``) before they are called.

    Parameters
    ----------
    theta : float, optional
        Mean-reversion speed (must be > 0).
    mu : float, optional
        Long-run mean.
    sigma : float, optional
        Diffusion coefficient (must be > 0).
    dt : float
        Sampling interval in consistent time units (default 1.0 for
        daily data where one step = one day).
    """

    def __init__(
        self,
        theta: float | None = None,
        mu: float | None = None,
        sigma: float | None = None,
        dt: float = 1.0,
    ) -> None:
        if dt <= 0:
            raise ValueError("dt must be positive.")
        self.dt = float(dt)
        self._theta: float | None = None
        self._mu: float | None = None
        self._sigma: float | None = None
        self._fit_result: OUFitResult | None = None

        if theta is not None:
            self.theta = theta
        if mu is not None:
            self.mu = mu
        if sigma is not None:
            self.sigma = sigma

    # ------------------------------------------------------------------
    # Parameter setters (with validation)
    # ------------------------------------------------------------------

    @property
    def theta(self) -> float:
        self._require_param("theta")
        return self._theta  # type: ignore[return-value]

    @theta.setter
    def theta(self, value: float) -> None:
        value = float(value)
        if value <= 0:
            raise ValueError(f"theta must be > 0, got {value}.")
        self._theta = value

    @property
    def mu(self) -> float:
        self._require_param("mu")
        return self._mu  # type: ignore[return-value]

    @mu.setter
    def mu(self, value: float) -> None:
        self._mu = float(value)

    @property
    def sigma(self) -> float:
        self._require_param("sigma")
        return self._sigma  # type: ignore[return-value]

    @sigma.setter
    def sigma(self, value: float) -> None:
        value = float(value)
        if value <= 0:
            raise ValueError(f"sigma must be > 0, got {value}.")
        self._sigma = value

    def _require_param(self, name: str) -> None:
        if getattr(self, f"_{name}") is None:
            raise RuntimeError(
                f"Parameter '{name}' is not set. Call fit() or set it manually."
            )

    def _require_fitted(self) -> None:
        for p in ("theta", "mu", "sigma"):
            self._require_param(p)

    # ------------------------------------------------------------------
    # Derived quantities (all require the model to be parameterised)
    # ------------------------------------------------------------------

    @property
    def half_life(self) -> float:
        """Time for the gap to the mean to shrink by 50 %. ln(2) / θ."""
        return np.log(2.0) / self.theta

    @property
    def stationary_var(self) -> float:
        """Variance of the stationary (equilibrium) distribution: σ² / (2θ)."""
        return self.sigma ** 2 / (2.0 * self.theta)

    @property
    def stationary_std(self) -> float:
        """Standard deviation of the stationary distribution: σ / √(2θ)."""
        return self.sigma / np.sqrt(2.0 * self.theta)

    @property
    def _b(self) -> float:
        """AR(1) coefficient b = e^{-θ Δt}."""
        return np.exp(-self.theta * self.dt)

    @property
    def _a(self) -> float:
        """AR(1) intercept a = μ (1 - b)."""
        return self.mu * (1.0 - self._b)

    @property
    def _sigma_eps(self) -> float:
        """Conditional std of the exact discrete transition."""
        b = self._b
        return self.sigma * np.sqrt((1.0 - b ** 2) / (2.0 * self.theta))

    def autocorrelation(self, lag: int) -> float:
        """
        Theoretical autocorrelation at integer lag k:  ρ(k) = e^{-θ k Δt}.

        Parameters
        ----------
        lag : int  (>= 0)
        """
        if lag < 0:
            raise ValueError("lag must be >= 0.")
        return float(np.exp(-self.theta * lag * self.dt))

    # ------------------------------------------------------------------
    # Parameter estimation
    # ------------------------------------------------------------------

    def fit(
        self,
        x: np.ndarray,
        method: str = "ols",
    ) -> OUFitResult:
        """
        Estimate θ, μ, σ from an observed time series.

        Parameters
        ----------
        x : array_like, shape (T,)
            Observed path sampled at intervals of ``self.dt``.
        method : {'ols', 'mle'}
            * ``'ols'`` — Exact conditional MLE via OLS regression on the
              equivalent AR(1) form.  Analytical, O(T).  Recommended default.
            * ``'mle'`` — Numerical maximisation of the full conditional
              log-likelihood.  Useful when ``dt`` is large or when you want
              guaranteed convergence on the (θ, μ, σ) simplex.

        Returns
        -------
        OUFitResult
        """
        x = np.asarray(x, dtype=float)
        if x.ndim != 1 or len(x) < 3:
            raise ValueError("x must be a 1-D array with at least 3 observations.")

        if method == "ols":
            return self._fit_ols(x)
        elif method == "mle":
            return self._fit_mle(x)
        else:
            raise ValueError(f"Unknown method '{method}'. Choose 'ols' or 'mle'.")

    def _fit_ols(self, x: np.ndarray) -> OUFitResult:
        """
        OLS regression of X_{t+1} on [1, X_t].

        The OLS estimator is identical to the exact conditional MLE for a
        Gaussian OU process when σ is profiled out analytically.
        """
        y   = x[1:]   # X_{t+1}
        x_t = x[:-1]  # X_t
        T   = len(y)

        # Design matrix [1, X_t]
        A = np.column_stack([np.ones(T), x_t])
        # Least squares: [a, b] = (A'A)^{-1} A'y
        coeffs, residuals_ss, _, _ = np.linalg.lstsq(A, y, rcond=None)
        a, b = float(coeffs[0]), float(coeffs[1])

        eps = y - (a + b * x_t)
        sigma_eps_sq = float(np.dot(eps, eps) / T)  # MLE denominator

        theta, mu, sigma = self._ar1_to_ou(a, b, sigma_eps_sq)
        self._theta, self._mu, self._sigma = theta, mu, sigma

        ll = self._log_likelihood_from_eps(eps, sigma_eps_sq, T)
        self._fit_result = OUFitResult(
            theta=theta, mu=mu, sigma=sigma,
            half_life=self.half_life, sigma_eq=self.stationary_std,
            log_lik=ll, method="ols", n_obs=T,
        )
        return self._fit_result

    def _fit_mle(self, x: np.ndarray) -> OUFitResult:
        """
        Numerical maximisation of the conditional log-likelihood.

        Parameterises as (log θ, μ, log σ) for unconstrained optimisation.
        Uses the OLS estimates as starting point.
        """
        # Warm-start from OLS
        ols = self._fit_ols(x)
        theta0, mu0, sigma0 = ols.theta, ols.mu, ols.sigma

        def neg_ll(params: np.ndarray) -> float:
            log_theta, mu_p, log_sigma = params
            theta_p = np.exp(log_theta)
            sigma_p = np.exp(log_sigma)
            return -self._conditional_log_likelihood(x, theta_p, mu_p, sigma_p)

        result = optimize.minimize(
            neg_ll,
            x0=[np.log(theta0), mu0, np.log(sigma0)],
            method="L-BFGS-B",
            options={"maxiter": 1000, "ftol": 1e-14, "gtol": 1e-8},
        )

        theta, mu, sigma = (
            float(np.exp(result.x[0])),
            float(result.x[1]),
            float(np.exp(result.x[2])),
        )
        self._theta, self._mu, self._sigma = theta, mu, sigma

        ll = self._conditional_log_likelihood(x, theta, mu, sigma)
        self._fit_result = OUFitResult(
            theta=theta, mu=mu, sigma=sigma,
            half_life=self.half_life, sigma_eq=self.stationary_std,
            log_lik=ll, method="mle", n_obs=len(x) - 1,
        )
        return self._fit_result

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate(
        self,
        n_steps: int,
        x0: float | None = None,
        n_paths: int = 1,
        seed: int | None = None,
    ) -> np.ndarray:
        """
        Simulate exact (discretisation-error-free) OU paths.

        Uses the exact conditional Gaussian transition:
            X_{t+Δ} = μ + (X_t - μ) e^{-θΔ} + ε_t,   ε_t ~ N(0, σ²_ε)

        Parameters
        ----------
        n_steps : int
            Number of time steps to simulate (output length = n_steps + 1).
        x0 : float, optional
            Starting value.  Defaults to the long-run mean μ.
        n_paths : int
            Number of independent sample paths.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        np.ndarray, shape (n_steps + 1,) if n_paths == 1,
                           (n_paths, n_steps + 1) otherwise.
        """
        self._require_fitted()
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1.")
        if n_paths < 1:
            raise ValueError("n_paths must be >= 1.")

        rng = np.random.default_rng(seed)
        x_start = float(self._mu if x0 is None else x0)  # type: ignore[arg-type]

        b         = self._b
        a         = self._a
        sigma_eps = self._sigma_eps

        paths = np.empty((n_paths, n_steps + 1))
        paths[:, 0] = x_start
        noise = rng.standard_normal((n_paths, n_steps))

        for t in range(n_steps):
            paths[:, t + 1] = a + b * paths[:, t] + sigma_eps * noise[:, t]

        return paths[0] if n_paths == 1 else paths

    # ------------------------------------------------------------------
    # Log-likelihood
    # ------------------------------------------------------------------

    def log_likelihood(self, x: np.ndarray) -> float:
        """
        Conditional log-likelihood of a path under the current parameters.

        P(X_1, ..., X_T | X_0) = Π P(X_{t+1} | X_t)

        Parameters
        ----------
        x : array_like, shape (T,)

        Returns
        -------
        float  Total log-likelihood.
        """
        self._require_fitted()
        x = np.asarray(x, dtype=float)
        return self._conditional_log_likelihood(x, self._theta, self._mu, self._sigma)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Trading / signal utilities
    # ------------------------------------------------------------------

    def z_score(self, x: np.ndarray) -> np.ndarray:
        """
        Standardised deviation from the long-run mean.

            z_t = (X_t - μ) / σ_eq

        where σ_eq = σ / √(2θ) is the stationary standard deviation.

        Parameters
        ----------
        x : array_like, shape (T,)
        """
        self._require_fitted()
        return (np.asarray(x, dtype=float) - self._mu) / self.stationary_std  # type: ignore[operator]

    def residuals(self, x: np.ndarray) -> np.ndarray:
        """
        One-step-ahead prediction residuals:  ε_t = X_{t+1} - (a + b X_t).

        Parameters
        ----------
        x : array_like, shape (T,)

        Returns
        -------
        np.ndarray, shape (T - 1,)
        """
        self._require_fitted()
        x = np.asarray(x, dtype=float)
        return x[1:] - (self._a + self._b * x[:-1])

    def band_probability(self, lower: float, upper: float) -> float:
        """
        Stationary probability that X lies in [lower, upper].

        Under stationarity X ~ N(μ, σ²/(2θ)).

        Parameters
        ----------
        lower, upper : float
        """
        self._require_fitted()
        dist = stats.norm(loc=self._mu, scale=self.stationary_std)
        return float(dist.cdf(upper) - dist.cdf(lower))

    def expected_crossing_time(self, x0: float, target: float | None = None) -> float:
        """
        Approximate expected time to reach ``target`` (default: μ) from ``x0``.

        For a Gaussian OU process the exact first-passage-time density does not
        have a closed form in general.  This returns the *expected reversion
        time* under the linear-decay approximation:

            E[τ] ≈ (1/θ) · ln(|x0 - μ| / |target - μ|)

        when target ≠ μ, or 1/θ (the relaxation time) when target = μ.

        Parameters
        ----------
        x0 : float  Starting value.
        target : float, optional  Target value.  Defaults to μ.
        """
        self._require_fitted()
        tgt = float(self._mu if target is None else target)  # type: ignore[arg-type]
        gap0 = abs(x0 - self._mu)  # type: ignore[operator]
        gap1 = abs(tgt - self._mu)  # type: ignore[operator]

        if gap0 == 0.0:
            return 0.0
        if gap1 == 0.0:
            # Crossing to the mean itself.  Use the log-distance approximation:
            #   E[τ → μ | X₀] ≈ (1/θ) ln(1 + |x₀ - μ| / σ_eq)
            # This is monotone in the starting distance and equals the
            # half-life when |x₀ - μ| = σ_eq.
            return (1.0 / self._theta) * np.log(1.0 + gap0 / self.stationary_std)  # type: ignore[operator]
        if gap1 >= gap0:
            # Target is farther from μ than x0 — already past the target.
            return 0.0
        return (1.0 / self._theta) * np.log(gap0 / gap1)  # type: ignore[operator]

    def ljung_box(self, x: np.ndarray, lags: int = 10) -> tuple[float, float]:
        """
        Ljung-Box test for residual autocorrelation.

        Tests H₀: residuals are i.i.d.  A small p-value suggests model
        mis-specification or remaining autocorrelation.

        Parameters
        ----------
        x : array_like  Observed series (residuals are computed internally).
        lags : int  Number of lags to include in the test statistic.

        Returns
        -------
        statistic : float  Q statistic.
        p_value : float
        """
        self._require_fitted()
        eps = self.residuals(x)
        T   = len(eps)
        acf = self._sample_acf(eps, lags)
        Q   = T * (T + 2) * np.sum(acf ** 2 / (T - np.arange(1, lags + 1)))
        p   = float(1.0 - stats.chi2.cdf(Q, df=lags))
        return float(Q), p

    def summary(self) -> str:
        """Return a formatted parameter summary string."""
        if self._fit_result is not None:
            return str(self._fit_result)
        self._require_fitted()
        lines = [
            "Ornstein-Uhlenbeck (manual parameters)",
            "=" * 40,
            f"  θ  (speed)  : {self._theta:.6f}",
            f"  μ  (mean)   : {self._mu:.6f}",
            f"  σ  (vol)    : {self._sigma:.6f}",
            f"  half-life   : {self.half_life:.4f}",
            f"  σ_eq        : {self.stationary_std:.6f}",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ar1_to_ou(
        self, a: float, b: float, sigma_eps_sq: float
    ) -> tuple[float, float, float]:
        """
        Convert AR(1) parameters (a, b, σ²_ε) to OU parameters (θ, μ, σ).

        Handles the degenerate case b ≥ 1 (unit root / explosive) by
        clamping b to a small positive value, emitting a warning.
        """
        import warnings

        dt = self.dt
        if b >= 1.0:
            warnings.warn(
                f"AR(1) coefficient b = {b:.4f} ≥ 1 implies a non-stationary "
                "process. θ has been clamped to a small positive value. "
                "Consider differencing the series.",
                RuntimeWarning,
                stacklevel=3,
            )
            b = max(b, 1e-6)
            b = min(b, 1.0 - 1e-9)

        if b <= 0.0:
            warnings.warn(
                f"AR(1) coefficient b = {b:.4f} ≤ 0. Clamping to small positive.",
                RuntimeWarning,
                stacklevel=3,
            )
            b = 1e-9

        theta = -np.log(b) / dt
        mu    = a / (1.0 - b)

        # σ²_ε = σ² (1 - e^{-2θΔ}) / (2θ)  ⟹  σ² = σ²_ε · 2θ / (1 - b²)
        denom = 1.0 - b ** 2
        if denom <= 0.0:
            denom = 1e-12
        sigma = np.sqrt(sigma_eps_sq * 2.0 * theta / denom)

        return float(theta), float(mu), float(sigma)

    def _conditional_log_likelihood(
        self,
        x: np.ndarray,
        theta: float,
        mu: float,
        sigma: float,
    ) -> float:
        """Full conditional log-likelihood L(θ,μ,σ | X_0,...,X_T)."""
        dt = self.dt
        b  = np.exp(-theta * dt)
        a  = mu * (1.0 - b)
        sigma_eps_sq = sigma ** 2 * (1.0 - b ** 2) / (2.0 * theta)
        if sigma_eps_sq <= 0.0:
            return -np.inf

        T   = len(x) - 1
        eps = x[1:] - (a + b * x[:-1])
        return float(
            -0.5 * T * np.log(2.0 * np.pi * sigma_eps_sq)
            - 0.5 * np.dot(eps, eps) / sigma_eps_sq
        )

    @staticmethod
    def _log_likelihood_from_eps(
        eps: np.ndarray, sigma_eps_sq: float, T: int
    ) -> float:
        return float(
            -0.5 * T * np.log(2.0 * np.pi * sigma_eps_sq)
            - 0.5 * np.dot(eps, eps) / sigma_eps_sq
        )

    @staticmethod
    def _sample_acf(x: np.ndarray, nlags: int) -> np.ndarray:
        """Sample autocorrelation function at lags 1 … nlags."""
        n = len(x)
        x = x - x.mean()
        var = np.dot(x, x) / n
        if var == 0.0:
            return np.zeros(nlags)
        acf = np.array(
            [np.dot(x[: n - k], x[k:]) / (n * var) for k in range(1, nlags + 1)]
        )
        return acf
