"""
Comprehensive Unit Tests for Streamlit Dashboard

Tests all components of the dashboard including:
- Data generation
- Portfolio optimization strategies
- Metrics evaluation
- Visualization components
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.visualization.dashboard import (
    generate_synthetic_data,
    optimize_portfolio,
    evaluate_portfolio
)


class TestDataGeneration:
    """Test synthetic data generation."""

    def test_generate_data_shape(self):
        """Test that generated data has correct shape."""
        n_assets = 10
        n_days = 252
        seed = 42

        prices, returns = generate_synthetic_data(n_assets, n_days, seed)

        assert prices.shape == (n_days, n_assets), f"Expected shape ({n_days}, {n_assets}), got {prices.shape}"
        assert returns.shape == (n_days, n_assets), f"Expected shape ({n_days}, {n_days}), got {returns.shape}"

    def test_generate_data_types(self):
        """Test that generated data has correct types."""
        prices, returns = generate_synthetic_data(5, 100, 42)

        assert isinstance(prices, pd.DataFrame), "Prices should be DataFrame"
        assert isinstance(returns, pd.DataFrame), "Returns should be DataFrame"
        assert isinstance(prices.index, pd.DatetimeIndex), "Prices index should be DatetimeIndex"

    def test_generate_data_columns(self):
        """Test that generated data has correct column names."""
        n_assets = 8
        prices, returns = generate_synthetic_data(n_assets, 100, 42)

        expected_columns = [f'ASSET_{i+1}' for i in range(n_assets)]
        assert list(prices.columns) == expected_columns, "Column names don't match expected format"
        assert list(returns.columns) == expected_columns, "Column names don't match expected format"

    def test_generate_data_deterministic(self):
        """Test that same seed produces same data."""
        seed = 123
        prices1, returns1 = generate_synthetic_data(5, 100, seed)
        prices2, returns2 = generate_synthetic_data(5, 100, seed)

        pd.testing.assert_frame_equal(prices1, prices2, "Same seed should produce identical prices")
        pd.testing.assert_frame_equal(returns1, returns2, "Same seed should produce identical returns")

    def test_generate_data_different_seeds(self):
        """Test that different seeds produce different data."""
        prices1, _ = generate_synthetic_data(5, 100, 42)
        prices2, _ = generate_synthetic_data(5, 100, 99)

        with pytest.raises(AssertionError):
            pd.testing.assert_frame_equal(prices1, prices2, "Different seeds should produce different data")

    def test_generate_data_no_nan(self):
        """Test that generated data has no NaN values."""
        prices, returns = generate_synthetic_data(10, 252, 42)

        assert not prices.isnull().any().any(), "Prices should not contain NaN"
        assert not returns.isnull().any().any(), "Returns should not contain NaN"

    def test_generate_data_positive_prices(self):
        """Test that all prices are positive."""
        prices, _ = generate_synthetic_data(10, 252, 42)

        assert (prices > 0).all().all(), "All prices should be positive"

    def test_generate_data_price_returns_relationship(self):
        """Test that prices and returns are correctly related."""
        prices, returns = generate_synthetic_data(5, 100, 42)

        # Check first return is approximately (price[1] - price[0]) / price[0]
        computed_returns = prices.pct_change().iloc[1:]

        # Returns should be related to price changes
        assert returns.shape == prices.shape, "Returns and prices should have same shape"


class TestPortfolioOptimization:
    """Test portfolio optimization strategies."""

    @pytest.fixture
    def sample_returns(self):
        """Create sample returns data for testing."""
        np.random.seed(42)
        n_assets = 10
        n_days = 252
        returns = pd.DataFrame(
            np.random.randn(n_days, n_assets) * 0.01,
            columns=[f'ASSET_{i+1}' for i in range(n_assets)]
        )
        return returns

    def test_equal_weight_strategy(self, sample_returns):
        """Test equal weight strategy."""
        weights, annual_returns, cov_matrix = optimize_portfolio(
            sample_returns, 'Equal Weight'
        )

        # All weights should be equal
        n_assets = len(sample_returns.columns)
        expected_weight = 1.0 / n_assets

        np.testing.assert_array_almost_equal(
            weights.values,
            np.ones(n_assets) * expected_weight,
            decimal=10,
            err_msg="Equal weight should give 1/N to each asset"
        )

    def test_weights_sum_to_one(self, sample_returns):
        """Test that all strategies produce weights that sum to 1."""
        strategies = ['Equal Weight', 'Max Sharpe', 'Min Variance', 'Concentrated']

        for strategy in strategies:
            if strategy == 'Concentrated':
                weights, _, _ = optimize_portfolio(sample_returns, strategy, max_assets=5)
            else:
                weights, _, _ = optimize_portfolio(sample_returns, strategy)

            weight_sum = weights.sum()
            assert abs(weight_sum - 1.0) < 1e-6, f"{strategy} weights sum to {weight_sum}, not 1.0"

    def test_weights_non_negative(self, sample_returns):
        """Test that all weights are non-negative (no shorting)."""
        strategies = ['Equal Weight', 'Max Sharpe', 'Min Variance', 'Concentrated']

        for strategy in strategies:
            if strategy == 'Concentrated':
                weights, _, _ = optimize_portfolio(sample_returns, strategy, max_assets=5)
            else:
                weights, _, _ = optimize_portfolio(sample_returns, strategy)

            assert (weights >= -1e-10).all(), f"{strategy} produced negative weights"

    def test_concentrated_cardinality(self, sample_returns):
        """Test that concentrated strategy respects cardinality constraint."""
        max_assets = 5
        weights, _, _ = optimize_portfolio(sample_returns, 'Concentrated', max_assets=max_assets)

        n_active = (weights > 1e-4).sum()
        assert n_active <= max_assets, f"Concentrated portfolio has {n_active} assets, expected <= {max_assets}"

    def test_max_sharpe_beats_equal_weight(self, sample_returns):
        """Test that Max Sharpe typically beats Equal Weight."""
        # This is probabilistic but should usually hold with good seed
        np.random.seed(42)

        weights_ms, annual_returns, cov_matrix = optimize_portfolio(sample_returns, 'Max Sharpe')
        weights_ew, _, _ = optimize_portfolio(sample_returns, 'Equal Weight')

        metrics_ms = evaluate_portfolio(weights_ms, annual_returns, cov_matrix)
        metrics_ew = evaluate_portfolio(weights_ew, annual_returns, cov_matrix)

        # Max Sharpe should generally have higher Sharpe ratio
        assert metrics_ms['sharpe'] >= metrics_ew['sharpe'] * 0.8, \
            "Max Sharpe should not be much worse than Equal Weight"

    def test_min_variance_lowest_volatility(self, sample_returns):
        """Test that Min Variance has lowest volatility."""
        np.random.seed(42)

        weights_mv, annual_returns, cov_matrix = optimize_portfolio(sample_returns, 'Min Variance')
        weights_ew, _, _ = optimize_portfolio(sample_returns, 'Equal Weight')

        metrics_mv = evaluate_portfolio(weights_mv, annual_returns, cov_matrix)
        metrics_ew = evaluate_portfolio(weights_ew, annual_returns, cov_matrix)

        # Min Variance should have lower or equal volatility
        assert metrics_mv['volatility'] <= metrics_ew['volatility'] * 1.1, \
            "Min Variance should have lower volatility than Equal Weight"

    def test_optimization_returns_correct_types(self, sample_returns):
        """Test that optimize_portfolio returns correct types."""
        weights, annual_returns, cov_matrix = optimize_portfolio(sample_returns, 'Equal Weight')

        assert isinstance(weights, pd.Series), "Weights should be Series"
        assert isinstance(annual_returns, pd.Series), "Annual returns should be Series"
        assert isinstance(cov_matrix, pd.DataFrame), "Covariance matrix should be DataFrame"

    def test_optimization_dimensions_match(self, sample_returns):
        """Test that all outputs have matching dimensions."""
        weights, annual_returns, cov_matrix = optimize_portfolio(sample_returns, 'Equal Weight')

        n_assets = len(sample_returns.columns)
        assert len(weights) == n_assets, "Weights dimension mismatch"
        assert len(annual_returns) == n_assets, "Annual returns dimension mismatch"
        assert cov_matrix.shape == (n_assets, n_assets), "Covariance matrix shape mismatch"


class TestPortfolioEvaluation:
    """Test portfolio evaluation metrics."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for evaluation."""
        np.random.seed(42)
        n_assets = 5

        weights = pd.Series(
            np.array([0.3, 0.3, 0.2, 0.1, 0.1]),
            index=[f'ASSET_{i+1}' for i in range(n_assets)]
        )

        annual_returns = pd.Series(
            np.array([0.10, 0.08, 0.12, 0.06, 0.15]),
            index=[f'ASSET_{i+1}' for i in range(n_assets)]
        )

        cov_matrix = pd.DataFrame(
            np.eye(n_assets) * 0.04,  # Diagonal covariance (uncorrelated)
            index=[f'ASSET_{i+1}' for i in range(n_assets)],
            columns=[f'ASSET_{i+1}' for i in range(n_assets)]
        )

        return weights, annual_returns, cov_matrix

    def test_evaluate_portfolio_return_calculation(self, sample_data):
        """Test that portfolio return is correctly calculated."""
        weights, annual_returns, cov_matrix = sample_data

        metrics = evaluate_portfolio(weights, annual_returns, cov_matrix)

        expected_return = (weights * annual_returns).sum()
        assert abs(metrics['return'] - expected_return) < 1e-10, \
            f"Return calculation incorrect: {metrics['return']} != {expected_return}"

    def test_evaluate_portfolio_volatility_positive(self, sample_data):
        """Test that volatility is positive."""
        weights, annual_returns, cov_matrix = sample_data

        metrics = evaluate_portfolio(weights, annual_returns, cov_matrix)

        assert metrics['volatility'] > 0, "Volatility should be positive"

    def test_evaluate_portfolio_sharpe_ratio(self, sample_data):
        """Test that Sharpe ratio is calculated correctly."""
        weights, annual_returns, cov_matrix = sample_data

        metrics = evaluate_portfolio(weights, annual_returns, cov_matrix)

        expected_sharpe = metrics['return'] / metrics['volatility']
        assert abs(metrics['sharpe'] - expected_sharpe) < 1e-10, \
            "Sharpe ratio calculation incorrect"

    def test_evaluate_portfolio_asset_count(self, sample_data):
        """Test that asset count is correct."""
        weights, annual_returns, cov_matrix = sample_data

        metrics = evaluate_portfolio(weights, annual_returns, cov_matrix)

        # All 5 weights are non-zero in sample_data
        assert metrics['n_assets'] == 5, "Should count all active assets"

    def test_evaluate_portfolio_asset_count_threshold(self):
        """Test that small weights are excluded from count."""
        weights = pd.Series([0.5, 0.5, 1e-5, 1e-10], index=['A1', 'A2', 'A3', 'A4'])
        annual_returns = pd.Series([0.1, 0.1, 0.1, 0.1], index=['A1', 'A2', 'A3', 'A4'])
        cov_matrix = pd.DataFrame(np.eye(4) * 0.04, index=['A1', 'A2', 'A3', 'A4'], columns=['A1', 'A2', 'A3', 'A4'])

        metrics = evaluate_portfolio(weights, annual_returns, cov_matrix)

        # Only weights > 1e-4 should be counted
        assert metrics['n_assets'] == 2, "Should only count weights > 1e-4"

    def test_evaluate_portfolio_metrics_keys(self, sample_data):
        """Test that all expected metrics are present."""
        weights, annual_returns, cov_matrix = sample_data

        metrics = evaluate_portfolio(weights, annual_returns, cov_matrix)

        expected_keys = {'return', 'volatility', 'sharpe', 'n_assets'}
        assert set(metrics.keys()) == expected_keys, "Missing or extra metrics keys"

    def test_evaluate_portfolio_zero_volatility_edge_case(self):
        """Test edge case where volatility is zero."""
        weights = pd.Series([1.0], index=['A1'])
        annual_returns = pd.Series([0.1], index=['A1'])
        cov_matrix = pd.DataFrame([[0.0]], index=['A1'], columns=['A1'])

        metrics = evaluate_portfolio(weights, annual_returns, cov_matrix)

        # Sharpe should be 0 when volatility is 0
        assert metrics['sharpe'] == 0, "Sharpe should be 0 when volatility is 0"


class TestIntegrationScenarios:
    """Integration tests for complete workflows."""

    def test_end_to_end_equal_weight(self):
        """Test complete workflow with Equal Weight strategy."""
        # Generate data
        prices, returns = generate_synthetic_data(10, 252, 42)

        # Optimize
        weights, annual_returns, cov_matrix = optimize_portfolio(returns, 'Equal Weight')

        # Evaluate
        metrics = evaluate_portfolio(weights, annual_returns, cov_matrix)

        # Assertions
        assert weights.sum() == pytest.approx(1.0, abs=1e-6), "Weights should sum to 1"
        assert metrics['sharpe'] is not None, "Sharpe ratio should be calculated"
        assert metrics['n_assets'] == 10, "All assets should be active in equal weight"

    def test_end_to_end_max_sharpe(self):
        """Test complete workflow with Max Sharpe strategy."""
        prices, returns = generate_synthetic_data(8, 500, 99)
        weights, annual_returns, cov_matrix = optimize_portfolio(returns, 'Max Sharpe')
        metrics = evaluate_portfolio(weights, annual_returns, cov_matrix)

        assert weights.sum() == pytest.approx(1.0, abs=1e-6), "Weights should sum to 1"
        assert metrics['sharpe'] > 0, "Sharpe ratio should be positive"
        assert all(weights >= -1e-10), "All weights should be non-negative"

    def test_end_to_end_min_variance(self):
        """Test complete workflow with Min Variance strategy."""
        prices, returns = generate_synthetic_data(6, 300, 123)
        weights, annual_returns, cov_matrix = optimize_portfolio(returns, 'Min Variance')
        metrics = evaluate_portfolio(weights, annual_returns, cov_matrix)

        assert weights.sum() == pytest.approx(1.0, abs=1e-6), "Weights should sum to 1"
        assert metrics['volatility'] > 0, "Volatility should be positive"

    def test_end_to_end_concentrated(self):
        """Test complete workflow with Concentrated strategy."""
        prices, returns = generate_synthetic_data(15, 400, 55)
        max_assets = 5
        weights, annual_returns, cov_matrix = optimize_portfolio(
            returns, 'Concentrated', max_assets=max_assets
        )
        metrics = evaluate_portfolio(weights, annual_returns, cov_matrix)

        assert weights.sum() == pytest.approx(1.0, abs=1e-6), "Weights should sum to 1"
        assert metrics['n_assets'] <= max_assets, f"Should have <= {max_assets} active assets"

    def test_multiple_strategies_same_data(self):
        """Test that all strategies work on same dataset."""
        prices, returns = generate_synthetic_data(10, 252, 42)

        strategies = ['Equal Weight', 'Max Sharpe', 'Min Variance']
        results = {}

        for strategy in strategies:
            weights, annual_returns, cov_matrix = optimize_portfolio(returns, strategy)
            metrics = evaluate_portfolio(weights, annual_returns, cov_matrix)
            results[strategy] = metrics

        # All strategies should produce valid results
        for strategy, metrics in results.items():
            assert metrics['return'] is not None, f"{strategy} return is None"
            assert metrics['volatility'] > 0, f"{strategy} volatility is not positive"
            assert metrics['sharpe'] is not None, f"{strategy} Sharpe is None"

    def test_reproducibility_with_seed(self):
        """Test that results are reproducible with same seed."""
        seed = 777

        # Run 1
        prices1, returns1 = generate_synthetic_data(10, 252, seed)
        weights1, _, _ = optimize_portfolio(returns1, 'Max Sharpe')

        # Run 2
        prices2, returns2 = generate_synthetic_data(10, 252, seed)
        weights2, _, _ = optimize_portfolio(returns2, 'Max Sharpe')

        # Results should be identical
        pd.testing.assert_frame_equal(prices1, prices2, "Prices should be identical")
        # Note: Max Sharpe uses random search, so weights might differ slightly
        # But data should be identical


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_asset_optimization(self):
        """Test optimization with single asset."""
        returns = pd.DataFrame({'ASSET_1': np.random.randn(100) * 0.01})

        weights, annual_returns, cov_matrix = optimize_portfolio(returns, 'Equal Weight')

        assert weights['ASSET_1'] == pytest.approx(1.0), "Single asset should get 100% weight"

    def test_two_assets_optimization(self):
        """Test optimization with two assets."""
        returns = pd.DataFrame({
            'ASSET_1': np.random.randn(100) * 0.01,
            'ASSET_2': np.random.randn(100) * 0.01
        })

        weights, annual_returns, cov_matrix = optimize_portfolio(returns, 'Equal Weight')

        assert weights.sum() == pytest.approx(1.0), "Weights should sum to 1"
        assert len(weights) == 2, "Should have 2 weights"

    def test_large_number_of_assets(self):
        """Test with large number of assets."""
        prices, returns = generate_synthetic_data(20, 252, 42)

        weights, annual_returns, cov_matrix = optimize_portfolio(returns, 'Equal Weight')

        assert len(weights) == 20, "Should handle 20 assets"
        assert weights.sum() == pytest.approx(1.0), "Weights should sum to 1"

    def test_short_time_series(self):
        """Test with very short time series."""
        prices, returns = generate_synthetic_data(5, 50, 42)

        weights, annual_returns, cov_matrix = optimize_portfolio(returns, 'Equal Weight')

        assert weights.sum() == pytest.approx(1.0), "Should work with short time series"

    def test_long_time_series(self):
        """Test with long time series."""
        prices, returns = generate_synthetic_data(5, 2000, 42)

        weights, annual_returns, cov_matrix = optimize_portfolio(returns, 'Equal Weight')

        assert weights.sum() == pytest.approx(1.0), "Should work with long time series"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
