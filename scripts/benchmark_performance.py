#!/usr/bin/env python
"""
Performance Benchmarking Script

Benchmarks optimization algorithms for speed and quality.

Usage:
    python scripts/benchmark_performance.py
    python scripts/benchmark_performance.py --detailed
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import time
import argparse
from collections import defaultdict


class PerformanceBenchmark:
    """Benchmark optimization algorithms."""

    def __init__(self, seed=42):
        """Initialize benchmarker."""
        np.random.seed(seed)
        self.results = defaultdict(list)

    def generate_test_case(self, n_assets, n_days):
        """Generate test data."""
        tickers = [f'A{i}' for i in range(n_assets)]

        n_factors = min(3, n_assets // 2)
        factor_loadings = np.random.randn(n_assets, n_factors) * 0.3
        factor_returns = np.random.randn(n_days, n_factors) * 0.01
        idiosyncratic = np.random.randn(n_days, n_assets) * 0.005
        drift = np.random.uniform(0.0001, 0.0005, n_assets)

        returns_matrix = factor_returns @ factor_loadings.T + idiosyncratic + drift

        dates = pd.date_range('2020-01-01', periods=n_days, freq='D')
        returns = pd.DataFrame(returns_matrix, index=dates, columns=tickers)

        annual_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252

        return returns, annual_returns, cov_matrix

    def benchmark_max_sharpe(self, annual_returns, cov_matrix, iterations=10000):
        """Benchmark max Sharpe optimization."""
        n_assets = len(annual_returns)

        start_time = time.time()

        best_sharpe = -np.inf
        best_weights = None

        for _ in range(iterations):
            w = np.random.dirichlet(np.ones(n_assets))
            port_return = (w * annual_returns.values).sum()
            port_vol = np.sqrt(w @ cov_matrix.values @ w)
            sharpe = port_return / port_vol if port_vol > 0 else 0

            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_weights = w

        elapsed = time.time() - start_time

        return {
            'weights': best_weights,
            'sharpe': best_sharpe,
            'time': elapsed,
            'iterations': iterations,
            'speed': iterations / elapsed
        }

    def benchmark_min_variance(self, cov_matrix, iterations=10000):
        """Benchmark min variance optimization."""
        n_assets = len(cov_matrix)

        start_time = time.time()

        min_var = np.inf
        min_var_weights = None

        for _ in range(iterations):
            w = np.random.dirichlet(np.ones(n_assets))
            port_var = w @ cov_matrix.values @ w

            if port_var < min_var:
                min_var = port_var
                min_var_weights = w

        elapsed = time.time() - start_time

        return {
            'weights': min_var_weights,
            'variance': min_var,
            'volatility': np.sqrt(min_var),
            'time': elapsed,
            'iterations': iterations,
            'speed': iterations / elapsed
        }

    def run_scalability_test(self, asset_counts, iterations=5000):
        """Test how performance scales with number of assets."""
        print("=" * 70)
        print("SCALABILITY TEST")
        print("=" * 70)
        print(f"Iterations per optimization: {iterations}")
        print()

        results = []

        for n_assets in asset_counts:
            print(f"Testing {n_assets} assets...", end=" ")

            # Generate test case
            returns, annual_returns, cov_matrix = self.generate_test_case(n_assets, 1000)

            # Benchmark
            sharpe_result = self.benchmark_max_sharpe(annual_returns, cov_matrix, iterations)
            var_result = self.benchmark_min_variance(cov_matrix, iterations)

            results.append({
                'n_assets': n_assets,
                'max_sharpe_time': sharpe_result['time'],
                'max_sharpe_speed': sharpe_result['speed'],
                'max_sharpe_value': sharpe_result['sharpe'],
                'min_var_time': var_result['time'],
                'min_var_speed': var_result['speed'],
                'min_var_value': var_result['volatility']
            })

            print(f"Done (Sharpe: {sharpe_result['time']:.2f}s, Var: {var_result['time']:.2f}s)")

        df = pd.DataFrame(results)

        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)
        print(df.to_string(index=False))
        print("=" * 70)

        return df

    def run_iteration_test(self, n_assets=10, iteration_counts=[1000, 5000, 10000, 20000]):
        """Test convergence with different iteration counts."""
        print("\n" + "=" * 70)
        print("ITERATION CONVERGENCE TEST")
        print("=" * 70)
        print(f"Assets: {n_assets}")
        print()

        # Generate test case
        returns, annual_returns, cov_matrix = self.generate_test_case(n_assets, 1000)

        results = []

        for iterations in iteration_counts:
            print(f"Testing {iterations} iterations...", end=" ")

            sharpe_result = self.benchmark_max_sharpe(annual_returns, cov_matrix, iterations)

            results.append({
                'iterations': iterations,
                'time': sharpe_result['time'],
                'speed': sharpe_result['speed'],
                'sharpe_ratio': sharpe_result['sharpe']
            })

            print(f"Sharpe: {sharpe_result['sharpe']:.4f}, Time: {sharpe_result['time']:.2f}s")

        df = pd.DataFrame(results)

        print("\n" + "=" * 70)
        print("CONVERGENCE RESULTS")
        print("=" * 70)
        print(df.to_string(index=False))
        print("=" * 70)

        # Calculate marginal improvement
        df['sharpe_improvement'] = df['sharpe_ratio'].diff()
        df['time_cost'] = df['time'].diff()

        print("\nMarginal Analysis:")
        print(df[['iterations', 'sharpe_improvement', 'time_cost']].iloc[1:].to_string(index=False))

        return df


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Benchmark optimization performance')
    parser.add_argument('--detailed', action='store_true', help='Run detailed benchmarks')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    print("=" * 70)
    print("PORTFOLIO OPTIMIZATION PERFORMANCE BENCHMARK")
    print("=" * 70)
    print(f"Seed: {args.seed}")
    print()

    benchmark = PerformanceBenchmark(seed=args.seed)

    # Basic scalability test
    asset_counts = [5, 10, 15, 20] if not args.detailed else [5, 10, 15, 20, 30, 50]
    scalability_results = benchmark.run_scalability_test(asset_counts)

    if args.detailed:
        # Iteration convergence test
        iteration_results = benchmark.run_iteration_test(
            n_assets=10,
            iteration_counts=[1000, 2500, 5000, 10000, 20000, 50000]
        )

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)

    # Summary statistics
    print("\nKey Findings:")
    print(f"  - Max assets tested: {scalability_results['n_assets'].max()}")
    print(f"  - Fastest optimization: {scalability_results['max_sharpe_time'].min():.3f}s")
    print(f"  - Average speed: {scalability_results['max_sharpe_speed'].mean():.0f} iterations/sec")

    print("\nRecommendations:")
    if scalability_results['max_sharpe_time'].max() < 5.0:
        print("  ✓ Performance is excellent for production use")
    elif scalability_results['max_sharpe_time'].max() < 15.0:
        print("  ✓ Performance is good for most applications")
    else:
        print("  ! Consider reducing iterations or using heuristics for large portfolios")

    return 0


if __name__ == '__main__':
    sys.exit(main())
