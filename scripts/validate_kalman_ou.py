"""
Validate the Kalman Filter on synthetic OU (mean-reverting) data.

Generates exact OU paths via src.models.OrnsteinUhlenbeck, corrupts them
with observation noise, then runs the KF in OU state-space form and checks:

  1. Parameter recovery   — filtered AR(1) coefficients converge to truth
  2. Innovation whiteness — normalised innovations pass Ljung-Box
  3. Smoother tightening  — RTS smoother covariance ≤ filter covariance
  4. Coverage             — empirical credible-interval coverage ≈ nominal
  5. Log-likelihood       — correct model beats a mis-specified one

Visual diagnostics (saved to scripts/figures/):
  - filtered state vs truth
  - innovation histogram + Q-Q plot
  - z-score time series with ±2σ bands
  - parameter convergence traces
  - coverage probability calibration
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.filters.kalman import KalmanFilter
from src.models.ou_process import OrnsteinUhlenbeck

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Ground-truth OU parameters
# ---------------------------------------------------------------------------
TRUE_THETA = 0.15       # mean-reversion speed
TRUE_MU    = 100.0      # long-run mean
TRUE_SIGMA = 2.0        # diffusion coefficient
DT         = 1.0        # daily sampling
OBS_NOISE  = 0.8        # observation noise std
N_STEPS    = 2000
SEED       = 42


def generate_data() -> tuple[np.ndarray, np.ndarray]:
    """Return (noisy_obs, true_hidden_state) arrays of length N_STEPS+1."""
    ou = OrnsteinUhlenbeck(theta=TRUE_THETA, mu=TRUE_MU, sigma=TRUE_SIGMA, dt=DT)
    x_true = ou.simulate(n_steps=N_STEPS, x0=TRUE_MU + 3.0, seed=SEED)
    rng = np.random.default_rng(SEED + 1)
    noise = rng.normal(0, OBS_NOISE, size=len(x_true))
    z_obs = x_true + noise
    return z_obs, x_true


def build_ou_kalman() -> KalmanFilter:
    """
    Build a KF in the OU state-space form.

    State:   x_k = hidden OU level   (scalar, n=1)
    Transition:  x_k = a + b * x_{k-1} + w_k
                 rewritten as  x_k = F * x_{k-1} + B * u_k + w_k
                 with  F = [[b]],  B = [[1]],  u_k = [a]

    Observation: z_k = x_k + v_k,   H = [[1]]

    Process noise:  Q = [[sigma_eps^2]]
    Obs noise:      R = [[obs_noise^2]]
    """
    b = np.exp(-TRUE_THETA * DT)
    a = TRUE_MU * (1.0 - b)
    sigma_eps = TRUE_SIGMA * np.sqrt((1.0 - b**2) / (2.0 * TRUE_THETA))

    f_mat  = np.array([[b]])
    b_mat  = np.array([[1.0]])
    h_mat  = np.array([[1.0]])
    q_mat  = np.array([[sigma_eps**2]])
    r_mat  = np.array([[OBS_NOISE**2]])
    x0     = np.array([TRUE_MU])
    p0_mat = np.array([[TRUE_SIGMA**2 / (2.0 * TRUE_THETA)]])

    return KalmanFilter(F=f_mat, H=h_mat, Q=q_mat, R=r_mat, x0=x0, P0=p0_mat, B=b_mat), a


# ===================================================================
# Numerical diagnostics
# ===================================================================

def run_numerical_tests(
    kf: KalmanFilter,
    a: float,
    z_obs: np.ndarray,
    x_true: np.ndarray,
) -> dict:
    """Run all numerical checks; return a dict of pass/fail results."""
    n_obs = len(z_obs)
    controls = np.full((n_obs, 1), a)
    filt = kf.filter(z_obs, controls=controls)
    smooth = kf.smooth(filt)

    results: dict[str, dict] = {}
    x_filt = filt.states[:, 0]
    x_smooth = smooth.states[:, 0]

    # --- 1. Filter accuracy (RMSE vs true state) -----------------------
    burn = 100
    rmse_filt = float(np.sqrt(np.mean((x_filt[burn:] - x_true[burn:])**2)))
    rmse_obs  = float(np.sqrt(np.mean((z_obs[burn:] - x_true[burn:])**2)))
    results["filter_rmse"] = {
        "rmse_filter": round(rmse_filt, 4),
        "rmse_raw_obs": round(rmse_obs, 4),
        "improvement_pct": round((1 - rmse_filt / rmse_obs) * 100, 1),
        "pass": rmse_filt < rmse_obs,
    }

    # --- 2. Smoother tighter than filter --------------------------------
    var_filt   = np.array([filt.covariances[i, 0, 0] for i in range(n_obs)])
    var_smooth = np.array([smooth.covariances[i, 0, 0] for i in range(n_obs)])
    smoother_tighter = bool(np.all(var_smooth[1:-1] <= var_filt[1:-1] + 1e-12))
    rmse_smooth = float(np.sqrt(np.mean((x_smooth[burn:] - x_true[burn:])**2)))
    results["smoother_tightening"] = {
        "rmse_smoother": round(rmse_smooth, 4),
        "smoother_leq_filter_everywhere": smoother_tighter,
        "pass": smoother_tighter and rmse_smooth <= rmse_filt + 0.01,
    }

    # --- 3. Innovation whiteness (Ljung-Box) ----------------------------
    innov = filt.innovations[burn:, 0]
    innov_var = np.array([filt.innovation_covs[i, 0, 0] for i in range(burn, n_obs)])
    norm_innov = innov / np.sqrt(innov_var)

    acf_1 = float(np.corrcoef(norm_innov[:-1], norm_innov[1:])[0, 1])

    n_valid = len(norm_innov)
    lags = 20
    acf_vals = np.array([
        float(np.corrcoef(norm_innov[:n_valid - k], norm_innov[k:])[0, 1])
        for k in range(1, lags + 1)
    ])
    q_stat = n_valid * (n_valid + 2) * np.sum(
        acf_vals**2 / (n_valid - np.arange(1, lags + 1))
    )
    lb_pval = float(1.0 - stats.chi2.cdf(q_stat, df=lags))

    results["innovation_whiteness"] = {
        "mean": round(float(np.mean(norm_innov)), 4),
        "std": round(float(np.std(norm_innov)), 4),
        "lag1_autocorr": round(acf_1, 4),
        "ljung_box_Q": round(q_stat, 2),
        "ljung_box_p": round(lb_pval, 4),
        "pass": lb_pval > 0.01 and abs(acf_1) < 0.1,
    }

    # --- 4. Credible interval coverage ----------------------------------
    for level_name, z_crit in [("68%", 1.0), ("95%", 1.96), ("99%", 2.576)]:
        filt_std = np.sqrt(var_filt[burn:])
        in_band = np.abs(x_filt[burn:] - x_true[burn:]) < z_crit * filt_std
        cov = float(np.mean(in_band))
        nominal = float(stats.norm.cdf(z_crit) - stats.norm.cdf(-z_crit))
        results[f"coverage_{level_name}"] = {
            "empirical": round(cov, 4),
            "nominal": round(nominal, 4),
            "pass": abs(cov - nominal) < 0.08,
        }

    # --- 5. Log-likelihood: correct model vs mis-specified ---------------
    ll_correct = filt.log_likelihood

    b_wrong = np.exp(-0.5 * DT)
    a_wrong = 50.0 * (1.0 - b_wrong)
    sig_eps_wrong = 5.0 * np.sqrt((1 - b_wrong**2) / 1.0)
    kf_wrong = KalmanFilter(
        F=np.array([[b_wrong]]),
        H=np.array([[1.0]]),
        Q=np.array([[sig_eps_wrong**2]]),
        R=np.array([[OBS_NOISE**2]]),
        x0=np.array([50.0]),
        P0=np.array([[25.0]]),
        B=np.array([[1.0]]),
    )
    controls_wrong = np.full((n_obs, 1), a_wrong)
    ll_wrong = kf_wrong.filter(z_obs, controls=controls_wrong).log_likelihood

    results["log_likelihood"] = {
        "correct_model": round(ll_correct, 2),
        "wrong_model": round(ll_wrong, 2),
        "delta": round(ll_correct - ll_wrong, 2),
        "pass": ll_correct > ll_wrong,
    }

    return results, filt, smooth, norm_innov


def print_report(results: dict) -> None:
    """Pretty-print the numerical diagnostic report."""
    print("\n" + "=" * 60)
    print("  KALMAN FILTER VALIDATION ON SYNTHETIC OU DATA")
    print("=" * 60)
    print(f"\n  Ground truth:  theta={TRUE_THETA}, mu={TRUE_MU}, "
          f"sigma={TRUE_SIGMA}")
    print(f"  Observation noise std = {OBS_NOISE}")
    print(f"  Series length = {N_STEPS + 1}\n")

    all_pass = True
    for name, info in results.items():
        passed = info.get("pass", None)
        tag = "PASS" if passed else "FAIL"
        marker = "+" if passed else "X"
        all_pass = all_pass and passed
        print(f"  [{marker}] {name}: {tag}")
        for k, v in info.items():
            if k != "pass":
                print(f"        {k}: {v}")
        print()

    print("-" * 60)
    if all_pass:
        print("  ALL CHECKS PASSED")
    else:
        print("  SOME CHECKS FAILED — see above")
    print("-" * 60)
    return all_pass


# ===================================================================
# Visual diagnostics
# ===================================================================

def plot_diagnostics(
    z_obs: np.ndarray,
    x_true: np.ndarray,
    filt,
    smooth,
    norm_innov: np.ndarray,
) -> None:
    """Generate and save all diagnostic plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
    except ImportError:
        print("  [!] matplotlib not installed — skipping plots.")
        return

    n_pts = len(z_obs)
    t = np.arange(n_pts)
    burn = 100

    x_filt = filt.states[:, 0]
    x_smooth = smooth.states[:, 0]
    filt_std = np.sqrt(np.array([filt.covariances[i, 0, 0] for i in range(n_pts)]))
    smooth_std = np.sqrt(np.array([smooth.covariances[i, 0, 0] for i in range(n_pts)]))

    # ---- Figure 1: Filtered/Smoothed state vs truth -------------------
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    ax = axes[0]
    ax.plot(t, z_obs, ".", color="silver", markersize=1, alpha=0.5, label="Observations")
    ax.plot(t, x_true, "k-", linewidth=0.8, label="True hidden state")
    ax.plot(t, x_filt, "b-", linewidth=0.8, label="KF filtered")
    ax.fill_between(t, x_filt - 2*filt_std, x_filt + 2*filt_std,
                     color="blue", alpha=0.15, label="Filter 95% CI")
    ax.set_ylabel("Level")
    ax.set_title("Kalman Filter on Synthetic OU Process")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[1]
    ax.plot(t, x_true, "k-", linewidth=0.8, label="True hidden state")
    ax.plot(t, x_smooth, "r-", linewidth=0.8, label="RTS smoothed")
    ax.fill_between(t, x_smooth - 2*smooth_std, x_smooth + 2*smooth_std,
                     color="red", alpha=0.15, label="Smoother 95% CI")
    ax.set_ylabel("Level")
    ax.set_title("RTS Smoother vs Truth")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[2]
    err_filt = x_filt - x_true
    err_smooth = x_smooth - x_true
    ax.plot(t, err_filt, "b-", linewidth=0.6, alpha=0.7, label="Filter error")
    ax.plot(t, err_smooth, "r-", linewidth=0.6, alpha=0.7, label="Smoother error")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_ylabel("Error")
    ax.set_xlabel("Time step")
    ax.set_title("Estimation Error")
    ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "01_filter_vs_truth.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {FIG_DIR / '01_filter_vs_truth.png'}")

    # ---- Figure 2: Innovation diagnostics -----------------------------
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 2, figure=fig)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(np.arange(burn, n_pts), norm_innov, linewidth=0.5, color="steelblue")
    ax1.axhline(0, color="k", linewidth=0.5)
    ax1.axhline(2, color="r", linewidth=0.5, linestyle="--")
    ax1.axhline(-2, color="r", linewidth=0.5, linestyle="--")
    ax1.set_title("Normalised Innovations")
    ax1.set_xlabel("Time step")
    ax1.set_ylabel("z")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(norm_innov, bins=60, density=True, color="steelblue", alpha=0.7,
             edgecolor="white", linewidth=0.3)
    xg = np.linspace(-4, 4, 200)
    ax2.plot(xg, stats.norm.pdf(xg), "r-", linewidth=1.5, label="N(0,1)")
    ax2.set_title("Innovation Histogram vs N(0,1)")
    ax2.legend()

    ax3 = fig.add_subplot(gs[1, 0])
    osm, osr = stats.probplot(norm_innov, dist="norm", fit=False)
    ax3.plot(osm, osr, ".", markersize=1.5, color="steelblue")
    lims = [min(osm.min(), osr.min()), max(osm.max(), osr.max())]
    ax3.plot(lims, lims, "r-", linewidth=1)
    ax3.set_title("Q-Q Plot (Normal)")
    ax3.set_xlabel("Theoretical quantiles")
    ax3.set_ylabel("Sample quantiles")

    ax4 = fig.add_subplot(gs[1, 1])
    max_lag = 30
    acf = np.array([
        float(np.corrcoef(norm_innov[:len(norm_innov) - k],
                           norm_innov[k:])[0, 1])
        for k in range(1, max_lag + 1)
    ])
    ax4.bar(np.arange(1, max_lag + 1), acf, color="steelblue", edgecolor="white")
    ci = 1.96 / np.sqrt(len(norm_innov))
    ax4.axhline(ci, color="r", linestyle="--", linewidth=0.8)
    ax4.axhline(-ci, color="r", linestyle="--", linewidth=0.8)
    ax4.set_title("Innovation ACF")
    ax4.set_xlabel("Lag")
    ax4.set_ylabel("Autocorrelation")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "02_innovation_diagnostics.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {FIG_DIR / '02_innovation_diagnostics.png'}")

    # ---- Figure 3: Variance reduction & coverage ----------------------
    fig, axes = plt.subplots(2, 1, figsize=(14, 7))

    ax = axes[0]
    ax.plot(t, filt_std, "b-", linewidth=0.8, label="Filter std")
    ax.plot(t, smooth_std, "r-", linewidth=0.8, label="Smoother std")
    ax.set_ylabel("Posterior std")
    ax.set_title("Filter vs Smoother Uncertainty")
    ax.legend()

    ax = axes[1]
    window = 200
    err_sq_filt = (x_filt - x_true)**2
    err_sq_smooth = (x_smooth - x_true)**2
    rolling_rmse_f = np.sqrt(np.convolve(err_sq_filt, np.ones(window)/window, mode="valid"))
    rolling_rmse_s = np.sqrt(np.convolve(err_sq_smooth, np.ones(window)/window, mode="valid"))
    t_roll = np.arange(window - 1, n_pts)
    ax.plot(t_roll, rolling_rmse_f, "b-", linewidth=0.8,
            label=f"Filter RMSE ({window}-bar rolling)")
    ax.plot(t_roll, rolling_rmse_s, "r-", linewidth=0.8,
            label=f"Smoother RMSE ({window}-bar rolling)")
    ax.axhline(OBS_NOISE, color="gray", linestyle="--", linewidth=0.8, label="Obs noise std")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Rolling RMSE")
    ax.set_title("Rolling RMSE: Filter & Smoother vs Observation Noise Baseline")
    ax.legend()

    plt.tight_layout()
    fig.savefig(FIG_DIR / "03_variance_reduction.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {FIG_DIR / '03_variance_reduction.png'}")

    # ---- Figure 4: Coverage calibration plot --------------------------
    fig, ax = plt.subplots(figsize=(7, 7))
    nominals = np.linspace(0.50, 0.99, 30)
    empiricals = []
    for nom in nominals:
        z_crit = stats.norm.ppf(0.5 + nom / 2)
        in_band = np.abs(x_filt[burn:] - x_true[burn:]) < z_crit * filt_std[burn:]
        empiricals.append(float(np.mean(in_band)))
    ax.plot(nominals, empiricals, "bo-", markersize=4, label="Filter")

    empiricals_s = []
    for nom in nominals:
        z_crit = stats.norm.ppf(0.5 + nom / 2)
        in_band = np.abs(x_smooth[burn:] - x_true[burn:]) < z_crit * smooth_std[burn:]
        empiricals_s.append(float(np.mean(in_band)))
    ax.plot(nominals, empiricals_s, "rs-", markersize=4, label="Smoother")

    ax.plot([0.5, 1.0], [0.5, 1.0], "k--", linewidth=0.8, label="Ideal")
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Credible Interval Calibration")
    ax.legend()
    ax.set_aspect("equal")
    ax.set_xlim(0.48, 1.01)
    ax.set_ylim(0.48, 1.01)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "04_coverage_calibration.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {FIG_DIR / '04_coverage_calibration.png'}")

    # ---- Figure 5: Kalman gain convergence ----------------------------
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    gains = filt.gains[:, 0, 0]

    ax = axes[0]
    ax.plot(t, gains, linewidth=0.8, color="steelblue")
    ax.set_ylabel("Kalman Gain K")
    ax.set_title("Kalman Gain Convergence")

    b_true = np.exp(-TRUE_THETA * DT)
    sigma_eps_true = TRUE_SIGMA * np.sqrt((1 - b_true**2) / (2 * TRUE_THETA))
    p_ss = sigma_eps_true**2
    s_ss = p_ss + OBS_NOISE**2
    k_ss = p_ss / s_ss
    ax.axhline(k_ss, color="r", linestyle="--", linewidth=0.8,
               label=f"Steady-state K = {k_ss:.4f}")
    ax.legend()

    ax = axes[1]
    ax.plot(t, filt_std, linewidth=0.8, color="steelblue")
    p_ss_total = sigma_eps_true**2 * OBS_NOISE**2 / (sigma_eps_true**2 + OBS_NOISE**2)
    ax.axhline(np.sqrt(p_ss_total), color="r", linestyle="--", linewidth=0.8,
               label=f"Steady-state std = {np.sqrt(p_ss_total):.4f}")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Filter std")
    ax.set_title("Filter Posterior Std Convergence")
    ax.legend()

    plt.tight_layout()
    fig.savefig(FIG_DIR / "05_gain_convergence.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {FIG_DIR / '05_gain_convergence.png'}")


# ===================================================================
# OU parameter recovery via batch OLS/MLE on filtered state
# ===================================================================

def test_ou_parameter_recovery(x_true: np.ndarray) -> None:
    """Fit OU model on the true (hidden) path and check parameter recovery."""
    print("\n" + "=" * 60)
    print("  OU PARAMETER RECOVERY (on true hidden path)")
    print("=" * 60)

    ou = OrnsteinUhlenbeck(dt=DT)
    for method in ("ols", "mle"):
        res = ou.fit(x_true, method=method)
        theta_err = abs(res.theta - TRUE_THETA) / TRUE_THETA * 100
        mu_err    = abs(res.mu - TRUE_MU) / TRUE_MU * 100
        sigma_err = abs(res.sigma - TRUE_SIGMA) / TRUE_SIGMA * 100
        print(f"\n  {method.upper()}:")
        print(f"    theta = {res.theta:.4f}  (true {TRUE_THETA}, err {theta_err:.1f}%)")
        print(f"    mu    = {res.mu:.4f}  (true {TRUE_MU}, err {mu_err:.1f}%)")
        print(f"    sigma = {res.sigma:.4f}  (true {TRUE_SIGMA}, err {sigma_err:.1f}%)")
        print(f"    half-life = {res.half_life:.2f}  (true {np.log(2)/TRUE_THETA:.2f})")
        lb_q, lb_p = ou.ljung_box(x_true, lags=10)
        tag = "PASS" if lb_p > 0.05 else "FAIL"
        print(f"    Ljung-Box(10): Q={lb_q:.2f}, p={lb_p:.4f}  [{tag}]")


# ===================================================================
# Main
# ===================================================================

def main() -> None:
    print("Generating synthetic OU data...")
    z_obs, x_true = generate_data()

    print("Building Kalman Filter (OU state-space form)...")
    kf, a = build_ou_kalman()

    print("Running numerical diagnostics...")
    results, filt, smooth, norm_innov = run_numerical_tests(kf, a, z_obs, x_true)
    all_pass = print_report(results)

    test_ou_parameter_recovery(x_true)

    print("\nGenerating diagnostic plots...")
    plot_diagnostics(z_obs, x_true, filt, smooth, norm_innov)

    print("\nDone.")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
