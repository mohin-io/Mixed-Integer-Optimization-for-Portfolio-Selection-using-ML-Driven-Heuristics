"""
Comprehensive validation script for Streamlit Dashboard.

This script validates that:
1. All imports work
2. All functions execute without errors
3. All strategies produce valid results
4. Visualizations can be generated
5. System is ready for deployment
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from src.visualization.dashboard import (
    generate_synthetic_data,
    optimize_portfolio,
    evaluate_portfolio
)


def print_header(text):
    """Print formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_success(text):
    """Print success message."""
    print(f"[OK] {text}")


def print_metric(name, value):
    """Print metric."""
    print(f"  • {name}: {value}")


def validate_imports():
    """Validate all required imports."""
    print_header("1. Validating Imports")

    try:
        import streamlit
        print_success("Streamlit imported")
    except ImportError:
        print("[ERROR] Streamlit import failed")
        return False

    try:
        import numpy
        print_success("NumPy imported")
    except ImportError:
        print("[ERROR] NumPy import failed")
        return False

    try:
        import pandas
        print_success("Pandas imported")
    except ImportError:
        print("[ERROR] Pandas import failed")
        return False

    try:
        import matplotlib
        print_success("Matplotlib imported")
    except ImportError:
        print("[ERROR] Matplotlib import failed")
        return False

    try:
        import seaborn
        print_success("Seaborn imported")
    except ImportError:
        print("[ERROR] Seaborn import failed")
        return False

    return True


def validate_data_generation():
    """Validate data generation functions."""
    print_header("2. Validating Data Generation")

    try:
        prices, returns = generate_synthetic_data(10, 252, 42)

        print_success(f"Generated data: {prices.shape[0]} days, {prices.shape[1]} assets")
        print_metric("Price range", f"${prices.min().min():.2f} - ${prices.max().max():.2f}")
        print_metric("Return mean", f"{returns.mean().mean():.4f}")
        print_metric("Return std", f"{returns.std().mean():.4f}")

        # Validate properties
        assert prices.shape == (252, 10), "Price shape mismatch"
        assert returns.shape == (252, 10), "Returns shape mismatch"
        assert not prices.isnull().any().any(), "Prices contain NaN"
        assert not returns.isnull().any().any(), "Returns contain NaN"
        assert (prices > 0).all().all(), "Negative prices found"

        print_success("All data validation checks passed")
        return True

    except Exception as e:
        print(f"[ERROR] Data generation failed: {e}")
        return False


def validate_strategies():
    """Validate all optimization strategies."""
    print_header("3. Validating Optimization Strategies")

    prices, returns = generate_synthetic_data(10, 500, 42)

    strategies = [
        ('Equal Weight', None),
        ('Max Sharpe', None),
        ('Min Variance', None),
        ('Concentrated', 5)
    ]

    results = {}

    for strategy_name, max_assets in strategies:
        try:
            if max_assets:
                weights, annual_returns, cov_matrix = optimize_portfolio(
                    returns, strategy_name, max_assets=max_assets
                )
            else:
                weights, annual_returns, cov_matrix = optimize_portfolio(
                    returns, strategy_name
                )

            metrics = evaluate_portfolio(weights, annual_returns, cov_matrix)
            results[strategy_name] = metrics

            print_success(f"{strategy_name} optimization completed")
            print_metric("Expected Return", f"{metrics['return']:.2%}")
            print_metric("Volatility", f"{metrics['volatility']:.2%}")
            print_metric("Sharpe Ratio", f"{metrics['sharpe']:.3f}")
            print_metric("Active Assets", f"{int(metrics['n_assets'])}")

            # Validate
            assert abs(weights.sum() - 1.0) < 1e-6, f"{strategy_name}: weights don't sum to 1"
            assert all(weights >= -1e-10), f"{strategy_name}: negative weights"
            assert metrics['volatility'] > 0, f"{strategy_name}: volatility not positive"

        except Exception as e:
            print(f"[ERROR] {strategy_name} failed: {e}")
            return False

    print_success("All strategies validated successfully")
    return True


def validate_visualizations():
    """Validate that visualizations can be generated."""
    print_header("4. Validating Visualizations")

    try:
        prices, returns = generate_synthetic_data(10, 252, 42)
        weights, annual_returns, cov_matrix = optimize_portfolio(returns, 'Max Sharpe')

        # Test bar chart (weights)
        fig, ax = plt.subplots(figsize=(10, 6))
        active_weights = weights[weights > 1e-4].sort_values(ascending=True)
        ax.barh(range(len(active_weights)), active_weights.values)
        plt.close(fig)
        print_success("Bar chart (weights) generated")

        # Test line chart (prices)
        fig, ax = plt.subplots(figsize=(12, 6))
        for col in prices.columns[:5]:
            ax.plot(prices.index, prices[col], label=col)
        plt.close(fig)
        print_success("Line chart (prices) generated")

        # Test heatmap (correlation)
        fig, ax = plt.subplots(figsize=(10, 8))
        corr_matrix = returns.corr()
        import seaborn as sns
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
        plt.close(fig)
        print_success("Heatmap (correlation) generated")

        # Test performance chart
        fig, ax = plt.subplots(figsize=(12, 6))
        portfolio_returns = (returns * weights.values).sum(axis=1)
        cumulative = (1 + portfolio_returns).cumprod()
        ax.plot(cumulative.index, cumulative.values)
        plt.close(fig)
        print_success("Performance chart generated")

        print_success("All visualizations validated successfully")
        return True

    except Exception as e:
        print(f"[ERROR] Visualization generation failed: {e}")
        return False


def validate_performance():
    """Validate performance metrics."""
    print_header("5. Validating Performance")

    import time

    # Test optimization speed
    prices, returns = generate_synthetic_data(10, 252, 42)

    start = time.time()
    weights, _, _ = optimize_portfolio(returns, 'Max Sharpe')
    elapsed = time.time() - start

    print_metric("Optimization time (Max Sharpe)", f"{elapsed:.2f} seconds")

    if elapsed < 30:
        print_success("Optimization completes in reasonable time")
    else:
        print(f"[WARNING] Optimization took {elapsed:.2f}s (>30s)")

    # Test scalability
    for n_assets in [5, 10, 15, 20]:
        prices, returns = generate_synthetic_data(n_assets, 252, 42)
        start = time.time()
        weights, _, _ = optimize_portfolio(returns, 'Equal Weight')
        elapsed = time.time() - start
        print_metric(f"Equal Weight ({n_assets} assets)", f"{elapsed:.3f} seconds")

    print_success("Performance validation completed")
    return True


def validate_edge_cases():
    """Validate edge cases."""
    print_header("6. Validating Edge Cases")

    # Test single asset
    try:
        returns = pd.DataFrame({'ASSET_1': np.random.randn(100) * 0.01})
        weights, _, _ = optimize_portfolio(returns, 'Equal Weight')
        assert weights['ASSET_1'] == 1.0
        print_success("Single asset case handled")
    except Exception as e:
        print(f"[ERROR] Single asset case failed: {e}")
        return False

    # Test two assets
    try:
        returns = pd.DataFrame({
            'ASSET_1': np.random.randn(100) * 0.01,
            'ASSET_2': np.random.randn(100) * 0.01
        })
        weights, _, _ = optimize_portfolio(returns, 'Equal Weight')
        assert len(weights) == 2
        print_success("Two assets case handled")
    except Exception as e:
        print(f"[ERROR] Two assets case failed: {e}")
        return False

    # Test short time series
    try:
        prices, returns = generate_synthetic_data(5, 50, 42)
        weights, _, _ = optimize_portfolio(returns, 'Equal Weight')
        print_success("Short time series handled")
    except Exception as e:
        print(f"[ERROR] Short time series failed: {e}")
        return False

    # Test long time series
    try:
        prices, returns = generate_synthetic_data(5, 2000, 42)
        weights, _, _ = optimize_portfolio(returns, 'Equal Weight')
        print_success("Long time series handled")
    except Exception as e:
        print(f"[ERROR] Long time series failed: {e}")
        return False

    print_success("All edge cases validated successfully")
    return True


def main():
    """Run all validation tests."""
    print("\n" + "=" * 70)
    print("  STREAMLIT DASHBOARD VALIDATION")
    print("  Portfolio Optimization System - Comprehensive Testing")
    print("=" * 70)

    all_passed = True

    # Run all validations
    all_passed &= validate_imports()
    all_passed &= validate_data_generation()
    all_passed &= validate_strategies()
    all_passed &= validate_visualizations()
    all_passed &= validate_performance()
    all_passed &= validate_edge_cases()

    # Final summary
    print_header("VALIDATION SUMMARY")

    if all_passed:
        print("\n[SUCCESS] ALL VALIDATIONS PASSED")
        print("\n[READY] System is READY FOR DEPLOYMENT")
        print("\nNext steps:")
        print("  1. Run: streamlit run src/visualization/dashboard.py")
        print("  2. Deploy to Streamlit Cloud")
        print("  3. Monitor performance")
        return 0
    else:
        print("\n[FAILED] SOME VALIDATIONS FAILED")
        print("\n[WARNING] System is NOT ready for deployment")
        print("\nPlease review errors above and fix issues.")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
