"""
Market Regime Detection using Gaussian Hidden Markov Models.

This script demonstrates how to use the continuous-emission GHMM to detect
unobservable market regimes (e.g., Bull vs. Bear markets) from financial data.
It fits a 2-state GHMM to daily log returns.
"""

import os
import sys

import numpy as np
from scipy.stats import multivariate_normal

# Ensure we can import qufin if run directly from the scripts directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from qufin.markov import ghmm

try:
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
except ImportError:
    print("This script requires 'matplotlib'. Install it via: pip install matplotlib")
    sys.exit(1)

try:
    import pandas as pd
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    print("Missing 'yfinance' or 'pandas'. Using synthetic financial data instead.")
    HAS_YFINANCE = False


def generate_synthetic_data(n_days=2500):
    """Generates a realistic synthetic price series with 2 regimes."""
    np.random.seed(42)
    # Regime 0: Bull Market (Positive mean, low vol)
    # Regime 1: Bear Market (Negative mean, high vol)
    
    # Transition matrix
    trans_mat = np.array([
        [0.98, 0.02], # Bull tends to stay Bull
        [0.05, 0.95]  # Bear stays Bear for shorter periods
    ])
    
    means = np.array([0.0005, -0.001]) # Daily returns
    vols = np.array([0.005, 0.015])
    
    states = np.zeros(n_days, dtype=int)
    returns = np.zeros(n_days)
    
    states[0] = 0
    for t in range(1, n_days):
        states[t] = np.random.choice([0, 1], p=trans_mat[states[t-1]])
        
    for t in range(n_days):
        returns[t] = np.random.normal(means[states[t]], vols[states[t]])
        
    # Create price series
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Create dummy dates
    import datetime
    dates = [datetime.date(2010, 1, 1) + datetime.timedelta(days=i) for i in range(n_days)]
    
    return dates, prices, returns


def main():
    print("=========================================================")
    print("   MARKET REGIME DETECTION STUDY USING GAUSSIAN HMM      ")
    print("=========================================================\n")

    if HAS_YFINANCE:
        print("Downloading S&P 500 (SPY) data for the last 15 years...")
        ticker = yf.Ticker("SPY")
        data = ticker.history(start="2009-01-01", end="2024-01-01")
        
        if data.empty:
            print("Failed to download data.")
            return

        # Calculate daily log returns
        data['Log_Ret'] = np.log(data['Close'] / data['Close'].shift(1))
        data = data.dropna()
        
        dates = data.index
        prices = data['Close'].values
        obs = data['Log_Ret'].values
    else:
        print("Generating synthetic financial data with volatility clustering...")
        dates, prices, obs = generate_synthetic_data()
    
    print(f"Data loaded: {len(obs)} trading days.")
    print("\nFitting a 2-state Gaussian HMM to the daily log returns...")
    print("This will identify two regimes with different mean and volatility characteristics.")
    
    model = ghmm.fit(
        obs=obs,
        n_states=2,
        n_init=10,        # Use 10 random restarts to find the global optimum
        max_iter=200,
        tol=1e-6,
        verbose=True
    )
    
    print("\n--- Model Fit Complete ---")
    print(f"Log Likelihood: {model.log_likelihood:.2f}")
    
    # Identify which state is the High Volatility (Bear) regime and which is Low Volatility (Bull)
    # The covariances have shape (K, D, D), so for 1D it's (2, 1, 1). We extract the variance.
    vols = np.sqrt(model.covars[:, 0, 0])
    means = model.means[:, 0]
    
    # We define the High Vol regime as the one with the higher variance
    high_vol_state = int(np.argmax(vols))
    low_vol_state = 1 - high_vol_state
    
    # Annualize the parameters (assuming 252 trading days)
    ann_mean_high = means[high_vol_state] * 252
    ann_vol_high = vols[high_vol_state] * np.sqrt(252)
    
    ann_mean_low = means[low_vol_state] * 252
    ann_vol_low = vols[low_vol_state] * np.sqrt(252)
    
    print("\nRegime 'Low Volatility' (Bull Market):")
    print(f"  Annualized Mean Return: {ann_mean_low:+.2%}")
    print(f"  Annualized Volatility:  {ann_vol_low:.2%}")
    
    print("\nRegime 'High Volatility' (Bear Market):")
    print(f"  Annualized Mean Return: {ann_mean_high:+.2%}")
    print(f"  Annualized Volatility:  {ann_vol_high:.2%}")

    # Decode the hidden states (Viterbi path)
    print("\nDecoding historical regime path using Viterbi algorithm...")
    hidden_states = ghmm.decode(obs, model)
    
    # Compute continuous posterior probabilities (gamma) using Forward-Backward
    print("Computing posterior probabilities using Forward-Backward algorithm...")
    log_trans = np.log(np.maximum(model.transition_matrix, 1e-12))
    log_pi = np.log(np.maximum(model.initial_probs, 1e-12))
    
    # Calculate log emission sequence exactly as in ghmm.py
    log_emit_seq = np.empty((len(obs), 2), dtype=np.float64)
    obs_2d = obs[:, np.newaxis]
    for k in range(2):
        log_emit_seq[:, k] = multivariate_normal.logpdf(
            obs_2d, mean=model.means[k], cov=model.covars[k], allow_singular=True
        )
    
    gamma, _, _ = ghmm.posteriors(log_emit_seq, log_trans, log_pi)
    
    # Probability of being in the High Volatility state
    prob_high_vol = gamma[:, high_vol_state]
    
    # --- Plotting ---
    print("\nGenerating plots...")
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Plot 1: Cumulative Return colored by Regime
    axes[0].plot(dates, prices, color='black', alpha=0.5, label='Asset Price')
    
    # Overlay colored dots for regimes
    # We map the high_vol_state to 'red' and low_vol_state to 'green'
    colors = np.where(hidden_states == high_vol_state, 'red', 'green')
    axes[0].scatter(dates, prices, c=colors, s=5, alpha=0.7)
    
    green_patch = mpatches.Patch(color='green', label='Regime: Low Vol (Bull)')
    red_patch = mpatches.Patch(color='red', label='Regime: High Vol (Bear)')
    axes[0].legend(handles=[green_patch, red_patch], loc='upper left')
    axes[0].set_title('Asset Price with Detected Market Regimes', fontsize=14)
    axes[0].set_ylabel('Price')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Posterior Probability of High Volatility Regime
    axes[1].plot(dates, prob_high_vol, color='darkred', lw=1)
    axes[1].fill_between(dates, 0, prob_high_vol, color='red', alpha=0.3)
    axes[1].axhline(0.5, color='black', linestyle='--', alpha=0.5)
    axes[1].set_title('Posterior Probability of High Volatility (Bear) Regime', fontsize=12)
    axes[1].set_ylabel('Probability')
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = "market_regime_study.png"
    plt.savefig(plot_path)
    print(f"Plot saved successfully to {plot_path}")
    
    print("\nDisplaying plot. Close the window to exit the script.")
    try:
        plt.show()
    except Exception as e:
        print(f"Could not display plot interactively: {e}")

if __name__ == "__main__":
    main()
