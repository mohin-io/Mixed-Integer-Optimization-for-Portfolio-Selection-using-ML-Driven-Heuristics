#!/usr/bin/env python
"""
Utility script to run comprehensive portfolio analysis.

Usage:
    python scripts/run_analysis.py --assets 10 --days 1000
    python scripts/run_analysis.py --quick  # Fast demo
    python scripts/run_analysis.py --full   # Complete analysis
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from datetime import datetime


def generate_synthetic_data(n_assets=10, n_days=1000, seed=42):
    """Generate synthetic market data."""
    np.random.seed(seed)

    tickers = [f'ASSET_{i+1}' for i in range(n_assets)]

    # Factor model
    n_factors = 3
    factor_loadings = np.random.randn(n_assets, n_factors) * 0.3
    factor_returns = np.random.randn(n_days, n_factors) * 0.01
    idiosyncratic_returns = np.random.randn(n_days, n_assets) * 0.005

    # Add drift
    drift = np.random.uniform(0.0001, 0.0005, n_assets)
    returns_matrix = factor_returns @ factor_loadings.T + idiosyncratic_returns + drift

    # Create DataFrame
    dates = pd.date_range('2020-01-01', periods=n_days, freq='D')
    returns = pd.DataFrame(returns_matrix, index=dates, columns=tickers)
    prices = (1 + returns).cumprod() * 100

    return prices, returns


def compute_statistics(returns):
    """Compute asset statistics."""
    annual_returns = returns.mean() * 252
    annual_volatility = returns.std() * np.sqrt(252)
    sharpe_ratios = annual_returns / annual_volatility
    cov_matrix = returns.cov() * 252

    return {
        'annual_returns': annual_returns,
        'annual_volatility': annual_volatility,
        'sharpe_ratios': sharpe_ratios,
        'cov_matrix': cov_matrix
    }


def optimize_portfolio(returns, cov_matrix, strategy='max_sharpe', max_assets=None):
    """Optimize portfolio using specified strategy."""
    n_assets = len(returns.columns)
    annual_returns = returns.mean() * 252

    if strategy == 'equal_weight':
        weights = np.ones(n_assets) / n_assets

    elif strategy == 'max_sharpe':
        best_sharpe = -np.inf
        best_weights = None

        for _ in range(10000):
            w = np.random.dirichlet(np.ones(n_assets))
            port_return = (w * annual_returns.values).sum()
            port_vol = np.sqrt(w @ cov_matrix.values @ w)
            sharpe = port_return / port_vol if port_vol > 0 else 0

            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_weights = w

        weights = best_weights

    elif strategy == 'min_variance':
        min_var = np.inf
        min_var_weights = None

        for _ in range(10000):
            w = np.random.dirichlet(np.ones(n_assets))
            port_var = w @ cov_matrix.values @ w

            if port_var < min_var:
                min_var = port_var
                min_var_weights = w

        weights = min_var_weights

    elif strategy == 'concentrated':
        if max_assets is None:
            max_assets = max(3, n_assets // 2)

        sharpe_ratios = annual_returns / (returns.std() * np.sqrt(252))
        top_assets = sharpe_ratios.nlargest(max_assets).index

        best_sharpe = -np.inf
        best_concentrated = None

        for _ in range(10000):
            w = np.zeros(n_assets)
            top_indices = [returns.columns.get_loc(t) for t in top_assets]
            w[top_indices] = np.random.dirichlet(np.ones(max_assets))

            port_return = (w * annual_returns.values).sum()
            port_vol = np.sqrt(w @ cov_matrix.values @ w)
            sharpe = port_return / port_vol if port_vol > 0 else 0

            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_concentrated = w

        weights = best_concentrated

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return pd.Series(weights, index=returns.columns)


def evaluate_portfolio(weights, annual_returns, cov_matrix):
    """Evaluate portfolio metrics."""
    port_return = (weights * annual_returns).sum()
    port_vol = np.sqrt(weights.values @ cov_matrix.values @ weights.values)
    sharpe = port_return / port_vol if port_vol > 0 else 0
    n_assets = (weights > 1e-4).sum()

    return {
        'return': port_return,
        'volatility': port_vol,
        'sharpe': sharpe,
        'n_assets': n_assets
    }


def run_analysis(n_assets=10, n_days=1000, max_concentrated=5, verbose=True):
    """Run complete portfolio analysis."""
    if verbose:
        print("=" * 70)
        print("PORTFOLIO OPTIMIZATION ANALYSIS")
        print("=" * 70)
        print(f"\nConfiguration:")
        print(f"  Assets: {n_assets}")
        print(f"  Days: {n_days}")
        print(f"  Max concentrated assets: {max_concentrated}")
        print()

    # Generate data
    if verbose:
        print("[1/4] Generating synthetic data...")
    prices, returns = generate_synthetic_data(n_assets, n_days)

    # Compute statistics
    if verbose:
        print("[2/4] Computing statistics...")
    stats = compute_statistics(returns)

    # Optimize portfolios
    if verbose:
        print("[3/4] Optimizing portfolios...")

    strategies = {
        'Equal Weight': 'equal_weight',
        'Max Sharpe': 'max_sharpe',
        'Min Variance': 'min_variance',
        'Concentrated': 'concentrated'
    }

    results = {}
    for name, strategy in strategies.items():
        if strategy == 'concentrated':
            weights = optimize_portfolio(returns, stats['cov_matrix'], strategy, max_concentrated)
        else:
            weights = optimize_portfolio(returns, stats['cov_matrix'], strategy)

        metrics = evaluate_portfolio(weights, stats['annual_returns'], stats['cov_matrix'])
        results[name] = metrics

    # Display results
    if verbose:
        print("[4/4] Results:")
        print()
        comparison_df = pd.DataFrame(results).T
        comparison_df.columns = ['Return', 'Volatility', 'Sharpe', 'N Assets']
        print(comparison_df.round(4))
        print()
        print("=" * 70)
        print(f"Best Sharpe Ratio: {comparison_df['Sharpe'].max():.3f} ({comparison_df['Sharpe'].idxmax()})")
        print("=" * 70)

    return results, stats


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run portfolio optimization analysis')
    parser.add_argument('--assets', type=int, default=10, help='Number of assets')
    parser.add_argument('--days', type=int, default=1000, help='Number of trading days')
    parser.add_argument('--concentrated', type=int, default=5, help='Max assets in concentrated portfolio')
    parser.add_argument('--quick', action='store_true', help='Quick demo (5 assets, 250 days)')
    parser.add_argument('--full', action='store_true', help='Full analysis (20 assets, 2000 days)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--quiet', action='store_true', help='Suppress output')

    args = parser.parse_args()

    if args.quick:
        n_assets, n_days = 5, 250
    elif args.full:
        n_assets, n_days = 20, 2000
    else:
        n_assets, n_days = args.assets, args.days

    np.random.seed(args.seed)

    results, stats = run_analysis(
        n_assets=n_assets,
        n_days=n_days,
        max_concentrated=args.concentrated,
        verbose=not args.quiet
    )

    return 0


if __name__ == '__main__':
    sys.exit(main())
