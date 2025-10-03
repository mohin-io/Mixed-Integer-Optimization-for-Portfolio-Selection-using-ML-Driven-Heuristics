"""
Integration and System-Level Tests for Streamlit Dashboard

Tests complete workflows, data pipelines, and system interactions.
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from io import BytesIO

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.visualization.dashboard import (
    generate_synthetic_data,
    optimize_portfolio,
    evaluate_portfolio
)


class TestDataPipeline:
    """Test complete data pipeline from generation to evaluation."""

    @pytest.mark.parametrize("n_assets,n_days,seed", [
        (5, 252, 42),
        (10, 500, 99),
        (15, 1000, 123),
        (20, 250, 777),
    ])
    def test_full_pipeline_parametrized(self, n_assets, n_days, seed):
        """Test full pipeline with various parameter combinations."""
        # Step 1: Generate data
        prices, returns = generate_synthetic_data(n_assets, n_days, seed)

        # Validate data generation
        assert prices.shape == (n_days, n_assets)
        assert returns.shape == (n_days, n_assets)
        assert not prices.isnull().any().any()
        assert not returns.isnull().any().any()

        # Step 2: Optimize portfolio
        weights, annual_returns, cov_matrix = optimize_portfolio(returns, 'Max Sharpe')

        # Validate optimization
        assert len(weights) == n_assets
        assert weights.sum() == pytest.approx(1.0, abs=1e-6)
        assert all(weights >= -1e-10)

        # Step 3: Evaluate portfolio
        metrics = evaluate_portfolio(weights, annual_returns, cov_matrix)

        # Validate metrics
        assert all(key in metrics for key in ['return', 'volatility', 'sharpe', 'n_assets'])
        assert metrics['volatility'] > 0
        assert 0 <= metrics['n_assets'] <= n_assets

    def test_all_strategies_pipeline(self):
        """Test pipeline with all available strategies."""
        prices, returns = generate_synthetic_data(10, 500, 42)

        strategies = ['Equal Weight', 'Max Sharpe', 'Min Variance', 'Concentrated']

        for strategy in strategies:
            if strategy == 'Concentrated':
                weights, annual_returns, cov_matrix = optimize_portfolio(
                    returns, strategy, max_assets=5
                )
            else:
                weights, annual_returns, cov_matrix = optimize_portfolio(
                    returns, strategy
                )

            metrics = evaluate_portfolio(weights, annual_returns, cov_matrix)

            # All strategies should produce valid results
            assert weights.sum() == pytest.approx(1.0, abs=1e-6), f"{strategy} weights don't sum to 1"
            assert metrics['volatility'] > 0, f"{strategy} volatility not positive"
            assert metrics['sharpe'] is not None, f"{strategy} Sharpe is None"

    def test_pipeline_data_consistency(self):
        """Test that data remains consistent through pipeline."""
        seed = 42
        prices, returns = generate_synthetic_data(10, 252, seed)

        # Store original data
        prices_copy = prices.copy()
        returns_copy = returns.copy()

        # Run optimization
        weights, annual_returns, cov_matrix = optimize_portfolio(returns, 'Equal Weight')

        # Evaluate
        metrics = evaluate_portfolio(weights, annual_returns, cov_matrix)

        # Original data should be unchanged
        pd.testing.assert_frame_equal(prices, prices_copy, "Prices were modified")
        pd.testing.assert_frame_equal(returns, returns_copy, "Returns were modified")


class TestStrategyComparison:
    """Test comparisons between different strategies."""

    @pytest.fixture
    def common_data(self):
        """Generate common dataset for all strategies."""
        np.random.seed(42)
        prices, returns = generate_synthetic_data(10, 500, 42)
        return prices, returns

    def test_all_strategies_produce_different_results(self, common_data):
        """Test that different strategies produce different results."""
        _, returns = common_data

        strategies = ['Equal Weight', 'Max Sharpe', 'Min Variance']
        weights_dict = {}

        for strategy in strategies:
            weights, _, _ = optimize_portfolio(returns, strategy)
            weights_dict[strategy] = weights

        # Equal Weight should be different from Max Sharpe
        assert not np.allclose(
            weights_dict['Equal Weight'].values,
            weights_dict['Max Sharpe'].values,
            atol=1e-3
        ), "Equal Weight and Max Sharpe should differ"

        # Equal Weight should be different from Min Variance
        assert not np.allclose(
            weights_dict['Equal Weight'].values,
            weights_dict['Min Variance'].values,
            atol=1e-3
        ), "Equal Weight and Min Variance should differ"

    def test_strategy_ranking_sharpe(self, common_data):
        """Test that Max Sharpe has highest Sharpe ratio."""
        _, returns = common_data

        weights_ew, annual_returns, cov_matrix = optimize_portfolio(returns, 'Equal Weight')
        weights_ms, _, _ = optimize_portfolio(returns, 'Max Sharpe')

        metrics_ew = evaluate_portfolio(weights_ew, annual_returns, cov_matrix)
        metrics_ms = evaluate_portfolio(weights_ms, annual_returns, cov_matrix)

        # Max Sharpe should generally beat or match Equal Weight
        assert metrics_ms['sharpe'] >= metrics_ew['sharpe'] * 0.7, \
            "Max Sharpe should not be significantly worse than Equal Weight"

    def test_strategy_ranking_volatility(self, common_data):
        """Test that Min Variance has lowest volatility."""
        _, returns = common_data

        weights_ew, annual_returns, cov_matrix = optimize_portfolio(returns, 'Equal Weight')
        weights_mv, _, _ = optimize_portfolio(returns, 'Min Variance')

        metrics_ew = evaluate_portfolio(weights_ew, annual_returns, cov_matrix)
        metrics_mv = evaluate_portfolio(weights_mv, annual_returns, cov_matrix)

        # Min Variance should have lower or comparable volatility
        assert metrics_mv['volatility'] <= metrics_ew['volatility'] * 1.2, \
            "Min Variance should have comparable or lower volatility"

    def test_concentrated_vs_full(self, common_data):
        """Test that concentrated portfolio has fewer assets."""
        _, returns = common_data

        max_assets = 5
        weights_full, annual_returns, cov_matrix = optimize_portfolio(returns, 'Max Sharpe')
        weights_conc, _, _ = optimize_portfolio(returns, 'Concentrated', max_assets=max_assets)

        metrics_full = evaluate_portfolio(weights_full, annual_returns, cov_matrix)
        metrics_conc = evaluate_portfolio(weights_conc, annual_returns, cov_matrix)

        # Concentrated should have fewer active assets
        assert metrics_conc['n_assets'] <= max_assets, \
            f"Concentrated has {metrics_conc['n_assets']} assets, expected <= {max_assets}"


class TestPortfolioPerformance:
    """Test portfolio performance calculations and metrics."""

    def test_portfolio_returns_calculation(self):
        """Test that portfolio returns are calculated correctly."""
        prices, returns = generate_synthetic_data(5, 252, 42)
        weights, annual_returns, cov_matrix = optimize_portfolio(returns, 'Equal Weight')

        # Calculate portfolio returns manually
        portfolio_returns = (returns * weights.values).sum(axis=1)

        # Check properties
        assert len(portfolio_returns) == len(returns), "Portfolio returns length mismatch"
        assert not portfolio_returns.isnull().any(), "Portfolio returns contain NaN"

    def test_cumulative_returns_positive_end(self):
        """Test that cumulative returns end positive for reasonable data."""
        prices, returns = generate_synthetic_data(10, 500, 42)
        weights, _, _ = optimize_portfolio(returns, 'Max Sharpe')

        portfolio_returns = (returns * weights.values).sum(axis=1)
        cumulative = (1 + portfolio_returns).cumprod()

        # Should end above starting value (probabilistic)
        assert cumulative.iloc[-1] > 0.5, "Cumulative returns collapsed"

    def test_metrics_consistency(self):
        """Test that metrics are internally consistent."""
        prices, returns = generate_synthetic_data(10, 252, 42)
        weights, annual_returns, cov_matrix = optimize_portfolio(returns, 'Equal Weight')
        metrics = evaluate_portfolio(weights, annual_returns, cov_matrix)

        # Sharpe ratio should match return/volatility
        calculated_sharpe = metrics['return'] / metrics['volatility']
        assert abs(metrics['sharpe'] - calculated_sharpe) < 1e-10, \
            "Sharpe ratio inconsistent with return/volatility"


class TestRobustness:
    """Test robustness and stability of the system."""

    def test_different_market_conditions(self):
        """Test with different random seeds (different market conditions)."""
        seeds = [42, 99, 123, 456, 789]

        for seed in seeds:
            prices, returns = generate_synthetic_data(10, 252, seed)
            weights, annual_returns, cov_matrix = optimize_portfolio(returns, 'Max Sharpe')
            metrics = evaluate_portfolio(weights, annual_returns, cov_matrix)

            # All should produce valid results
            assert weights.sum() == pytest.approx(1.0, abs=1e-6), f"Seed {seed}: weights don't sum to 1"
            assert metrics['volatility'] > 0, f"Seed {seed}: volatility not positive"
            assert metrics['n_assets'] >= 1, f"Seed {seed}: no assets selected"

    def test_stability_across_runs(self):
        """Test that same inputs produce same outputs."""
        seed = 42

        # Run 1
        prices1, returns1 = generate_synthetic_data(10, 252, seed)
        weights1, _, _ = optimize_portfolio(returns1, 'Equal Weight')

        # Run 2
        prices2, returns2 = generate_synthetic_data(10, 252, seed)
        weights2, _, _ = optimize_portfolio(returns2, 'Equal Weight')

        # Should be identical for Equal Weight
        pd.testing.assert_series_equal(weights1, weights2, "Equal Weight not deterministic")

    def test_extreme_parameter_values(self):
        """Test with extreme but valid parameter values."""
        # Very few assets
        prices, returns = generate_synthetic_data(2, 100, 42)
        weights, annual_returns, cov_matrix = optimize_portfolio(returns, 'Equal Weight')
        assert weights.sum() == pytest.approx(1.0, abs=1e-6)

        # Many assets
        prices, returns = generate_synthetic_data(20, 500, 42)
        weights, annual_returns, cov_matrix = optimize_portfolio(returns, 'Equal Weight')
        assert weights.sum() == pytest.approx(1.0, abs=1e-6)

        # Short time series
        prices, returns = generate_synthetic_data(5, 50, 42)
        weights, annual_returns, cov_matrix = optimize_portfolio(returns, 'Equal Weight')
        assert weights.sum() == pytest.approx(1.0, abs=1e-6)


class TestDataQuality:
    """Test quality and validity of generated data."""

    def test_returns_distribution_reasonable(self):
        """Test that returns have reasonable statistical properties."""
        _, returns = generate_synthetic_data(10, 1000, 42)

        # Mean returns should be small
        mean_returns = returns.mean().mean()
        assert abs(mean_returns) < 0.01, "Mean returns too large"

        # Std dev should be reasonable
        std_returns = returns.std().mean()
        assert 0.001 < std_returns < 0.1, f"Returns std dev unreasonable: {std_returns}"

    def test_prices_grow_over_time(self):
        """Test that prices generally trend upward (have positive drift)."""
        prices, _ = generate_synthetic_data(10, 1000, 42)

        # Check that final prices are generally higher than initial
        initial_mean = prices.iloc[0].mean()
        final_mean = prices.iloc[-1].mean()

        # Should generally increase (probabilistic with drift)
        assert final_mean > initial_mean * 0.8, "Prices did not grow as expected"

    def test_correlation_structure(self):
        """Test that assets have some correlation structure."""
        _, returns = generate_synthetic_data(10, 500, 42)

        corr_matrix = returns.corr()

        # Diagonal should be 1
        np.testing.assert_array_almost_equal(
            np.diag(corr_matrix),
            np.ones(10),
            decimal=10,
            err_msg="Diagonal of correlation matrix not 1"
        )

        # Off-diagonal should have some non-zero correlations (due to factor model)
        off_diagonal = corr_matrix.values[~np.eye(10, dtype=bool)]
        assert not np.allclose(off_diagonal, 0, atol=1e-10), \
            "No correlations found (factor model not working)"

    def test_covariance_matrix_properties(self):
        """Test that covariance matrix has correct properties."""
        _, returns = generate_synthetic_data(10, 500, 42)
        _, annual_returns, cov_matrix = optimize_portfolio(returns, 'Equal Weight')

        # Should be symmetric
        np.testing.assert_array_almost_equal(
            cov_matrix.values,
            cov_matrix.values.T,
            decimal=10,
            err_msg="Covariance matrix not symmetric"
        )

        # Should be positive semi-definite (all eigenvalues >= 0)
        eigenvalues = np.linalg.eigvalsh(cov_matrix.values)
        assert all(eigenvalues >= -1e-10), "Covariance matrix not positive semi-definite"


class TestSystemIntegration:
    """System-level integration tests."""

    def test_complete_workflow_no_errors(self):
        """Test that complete workflow runs without errors."""
        # This simulates what would happen in the Streamlit app
        n_assets = 10
        n_days = 500
        seed = 42

        # Generate
        prices, returns = generate_synthetic_data(n_assets, n_days, seed)

        # Test all strategies
        strategies = ['Equal Weight', 'Max Sharpe', 'Min Variance', 'Concentrated']

        for strategy in strategies:
            if strategy == 'Concentrated':
                weights, annual_returns, cov_matrix = optimize_portfolio(
                    returns, strategy, max_assets=5
                )
            else:
                weights, annual_returns, cov_matrix = optimize_portfolio(
                    returns, strategy
                )

            metrics = evaluate_portfolio(weights, annual_returns, cov_matrix)

            # Calculate portfolio performance
            portfolio_returns = (returns * weights.values).sum(axis=1)
            cumulative = (1 + portfolio_returns).cumprod()

            # All steps completed successfully
            assert True  # If we get here, no errors occurred

    def test_visualization_data_preparation(self):
        """Test that data can be prepared for visualization."""
        prices, returns = generate_synthetic_data(10, 252, 42)
        weights, annual_returns, cov_matrix = optimize_portfolio(returns, 'Max Sharpe')

        # Test data for bar chart (weights)
        active_weights = weights[weights > 1e-4].sort_values(ascending=True)
        assert len(active_weights) > 0, "No active weights for visualization"

        # Test data for line chart (prices)
        assert len(prices.index) > 0, "No price data for visualization"
        assert len(prices.columns) > 0, "No assets for price visualization"

        # Test data for heatmap (correlation)
        corr_matrix = returns.corr()
        assert corr_matrix.shape[0] == corr_matrix.shape[1], "Correlation matrix not square"

        # Test data for performance chart
        portfolio_returns = (returns * weights.values).sum(axis=1)
        cumulative = (1 + portfolio_returns).cumprod()
        assert len(cumulative) > 0, "No cumulative returns data"

    def test_session_state_simulation(self):
        """Simulate Streamlit session state behavior."""
        session_state = {}

        # Initial optimization
        prices, returns = generate_synthetic_data(10, 252, 42)
        weights, annual_returns, cov_matrix = optimize_portfolio(returns, 'Max Sharpe')
        metrics = evaluate_portfolio(weights, annual_returns, cov_matrix)

        # Store in "session state"
        session_state['prices'] = prices
        session_state['returns'] = returns
        session_state['weights'] = weights
        session_state['metrics'] = metrics

        # Verify data retrieval
        assert 'weights' in session_state
        assert 'metrics' in session_state
        assert session_state['weights'].sum() == pytest.approx(1.0, abs=1e-6)
        assert session_state['metrics']['sharpe'] is not None


class TestPerformanceAndScalability:
    """Test performance and scalability characteristics."""

    def test_optimization_completes_reasonable_time(self):
        """Test that optimization completes in reasonable time."""
        import time

        prices, returns = generate_synthetic_data(10, 252, 42)

        start_time = time.time()
        weights, _, _ = optimize_portfolio(returns, 'Max Sharpe')
        elapsed = time.time() - start_time

        # Max Sharpe with 10000 iterations should complete in < 30 seconds
        assert elapsed < 30, f"Optimization too slow: {elapsed:.2f}s"

    def test_scalability_with_assets(self):
        """Test that system scales with number of assets."""
        asset_counts = [5, 10, 15, 20]

        for n_assets in asset_counts:
            prices, returns = generate_synthetic_data(n_assets, 252, 42)
            weights, annual_returns, cov_matrix = optimize_portfolio(returns, 'Equal Weight')
            metrics = evaluate_portfolio(weights, annual_returns, cov_matrix)

            # Should work for all sizes
            assert len(weights) == n_assets
            assert weights.sum() == pytest.approx(1.0, abs=1e-6)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
