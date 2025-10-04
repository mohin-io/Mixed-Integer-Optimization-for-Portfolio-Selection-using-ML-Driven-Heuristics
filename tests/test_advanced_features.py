"""
Tests for advanced optimization features.

Tests:
- Fama-French factor models
- CVaR optimization
- Black-Litterman model
- Multi-period optimization
- Short-selling constraints
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Import modules
from src.forecasting.factor_models import FamaFrenchFactors, BarraFactorModel
from src.optimization.cvar_optimizer import CVaROptimizer, RobustCVaROptimizer
from src.forecasting.black_litterman import (
    BlackLittermanModel,
    InvestorView,
    create_absolute_view,
    create_relative_view
)
from src.optimization.multiperiod_optimizer import (
    MultiPeriodOptimizer,
    MultiPeriodConfig,
    ThresholdRebalancingPolicy
)
from src.optimization.mio_optimizer import MIOOptimizer, OptimizationConfig


class TestFamaFrenchFactors:
    """Test Fama-French factor model."""

    def setup_method(self):
        """Setup test data."""
        self.ff_model = FamaFrenchFactors(use_local_data=True)
        self.n_periods = 252
        self.n_assets = 5

        # Generate synthetic factor data
        self.factors = self.ff_model._generate_synthetic_factors(
            '2022-01-01',
            '2022-12-31',
            frequency='daily'
        )

        # Generate asset returns
        np.random.seed(42)
        self.returns = pd.DataFrame(
            np.random.randn(self.n_periods, self.n_assets) * 0.01,
            index=self.factors.index[:self.n_periods],
            columns=[f'Asset_{i}' for i in range(self.n_assets)]
        )

    def test_factor_data_generation(self):
        """Test synthetic factor data generation."""
        assert self.factors is not None
        assert len(self.factors) > 0
        assert all(col in self.factors.columns for col in ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA'])

    def test_factor_loading_estimation(self):
        """Test factor loading estimation."""
        result = self.ff_model.estimate_factor_loadings(self.returns, self.factors)

        assert result.factor_loadings.shape == (self.n_assets, 5)
        assert len(result.r_squared) == self.n_assets
        assert all(0 <= r2 <= 1 for r2 in result.r_squared)

    def test_covariance_computation(self):
        """Test total covariance matrix computation."""
        result = self.ff_model.estimate_factor_loadings(self.returns, self.factors)
        total_cov = self.ff_model.compute_total_covariance(result)

        assert total_cov.shape == (self.n_assets, self.n_assets)
        assert np.allclose(total_cov, total_cov.T)  # Symmetric
        assert np.all(np.linalg.eigvals(total_cov) >= -1e-10)  # Positive semi-definite

    def test_return_forecasting(self):
        """Test return forecasting with factor model."""
        result = self.ff_model.estimate_factor_loadings(self.returns, self.factors)
        forecasted = self.ff_model.forecast_returns(result, use_historical_mean=True)

        assert len(forecasted) == self.n_assets
        assert isinstance(forecasted, pd.Series)


class TestCVaROptimization:
    """Test CVaR optimization."""

    def setup_method(self):
        """Setup test data."""
        np.random.seed(42)
        self.n_assets = 5

        self.expected_returns = np.array([0.10, 0.12, 0.08, 0.15, 0.09])
        volatilities = np.array([0.15, 0.20, 0.12, 0.25, 0.18])

        correlation = np.eye(self.n_assets)
        for i in range(self.n_assets):
            for j in range(i+1, self.n_assets):
                correlation[i, j] = correlation[j, i] = 0.3

        self.covariance = np.outer(volatilities, volatilities) * correlation

    def test_cvar_optimizer_initialization(self):
        """Test CVaR optimizer initialization."""
        optimizer = CVaROptimizer(confidence_level=0.95, n_scenarios=1000)

        assert optimizer.confidence_level == 0.95
        assert optimizer.alpha == 0.05
        assert optimizer.n_scenarios == 1000

    def test_scenario_generation(self):
        """Test scenario generation."""
        optimizer = CVaROptimizer(n_scenarios=100)
        scenarios = optimizer.generate_scenarios(
            self.expected_returns,
            self.covariance,
            n_scenarios=100
        )

        assert scenarios.shape == (100, self.n_assets)

    def test_cvar_optimization(self):
        """Test CVaR portfolio optimization."""
        optimizer = CVaROptimizer(confidence_level=0.95, n_scenarios=500)

        result = optimizer.optimize(
            self.expected_returns,
            self.covariance,
            min_return=0.10
        )

        assert 'weights' in result
        assert len(result['weights']) == self.n_assets
        assert np.abs(np.sum(result['weights']) - 1.0) < 1e-4
        assert result['status'] in ['optimal', 'optimal_inaccurate']

    def test_robust_cvar_optimization(self):
        """Test robust CVaR optimization."""
        optimizer = RobustCVaROptimizer(
            confidence_level=0.95,
            robustness_param=0.1
        )

        result = optimizer.optimize_robust_cvar(
            self.expected_returns,
            self.covariance,
            min_return=0.10
        )

        assert 'weights' in result
        assert 'robustness' in result
        assert result['robustness'] == 0.1


class TestBlackLitterman:
    """Test Black-Litterman model."""

    def setup_method(self):
        """Setup test data."""
        self.assets = ['AAPL', 'MSFT', 'GOOGL']
        n = len(self.assets)

        np.random.seed(42)
        volatilities = np.array([0.20, 0.18, 0.22])
        correlation = np.eye(n)
        correlation[0, 1] = correlation[1, 0] = 0.5
        correlation[0, 2] = correlation[2, 0] = 0.4
        correlation[1, 2] = correlation[2, 1] = 0.45

        self.covariance = pd.DataFrame(
            np.outer(volatilities, volatilities) * correlation,
            index=self.assets,
            columns=self.assets
        )

        self.market_weights = pd.Series([0.4, 0.4, 0.2], index=self.assets)

    def test_equilibrium_returns(self):
        """Test equilibrium return computation."""
        bl_model = BlackLittermanModel(risk_aversion=2.5)
        eq_returns = bl_model.compute_equilibrium_returns(
            self.covariance,
            self.market_weights
        )

        assert len(eq_returns) == len(self.assets)
        assert isinstance(eq_returns, pd.Series)

    def test_view_creation(self):
        """Test investor view creation."""
        absolute_view = create_absolute_view('AAPL', 0.15, confidence=0.7)
        assert absolute_view.assets == ['AAPL']
        assert absolute_view.expected_return == 0.15

        relative_view = create_relative_view('MSFT', 'GOOGL', 0.03, confidence=0.6)
        assert relative_view.assets == ['MSFT', 'GOOGL']
        assert relative_view.weights == [1.0, -1.0]

    def test_posterior_returns(self):
        """Test posterior return computation."""
        bl_model = BlackLittermanModel(risk_aversion=2.5, tau=0.05)

        eq_returns = bl_model.compute_equilibrium_returns(
            self.covariance,
            self.market_weights
        )

        views = [
            create_absolute_view('AAPL', 0.15, confidence=0.7),
            create_relative_view('MSFT', 'GOOGL', 0.03, confidence=0.6)
        ]

        posterior_returns, posterior_cov = bl_model.compute_posterior_returns(
            eq_returns,
            self.covariance,
            views
        )

        assert len(posterior_returns) == len(self.assets)
        assert posterior_cov.shape == (len(self.assets), len(self.assets))

    def test_bl_run(self):
        """Test complete Black-Litterman workflow."""
        bl_model = BlackLittermanModel(risk_aversion=2.5)

        views = [create_absolute_view('AAPL', 0.15, confidence=0.7)]

        result = bl_model.run(self.covariance, views, self.market_weights)

        assert 'equilibrium_returns' in result
        assert 'posterior_returns' in result
        assert 'posterior_covariance' in result


class TestMultiPeriodOptimization:
    """Test multi-period optimization."""

    def setup_method(self):
        """Setup test data."""
        self.n_assets = 3
        self.n_periods = 6

        self.expected_returns = np.array([0.08, 0.10, 0.06]) / 12
        volatilities = np.array([0.15, 0.20, 0.12]) / np.sqrt(12)

        correlation = np.array([
            [1.0, 0.5, 0.3],
            [0.5, 1.0, 0.4],
            [0.3, 0.4, 1.0]
        ])

        self.covariance = np.outer(volatilities, volatilities) * correlation

    def test_scenario_generation(self):
        """Test scenario tree generation."""
        config = MultiPeriodConfig(n_periods=self.n_periods)
        optimizer = MultiPeriodOptimizer(config)

        scenario_tree = optimizer.generate_scenarios(
            self.expected_returns,
            self.covariance,
            n_scenarios=10
        )

        assert scenario_tree.n_scenarios == 10
        assert scenario_tree.n_periods == self.n_periods
        assert scenario_tree.n_assets == self.n_assets

    def test_deterministic_optimization(self):
        """Test deterministic multi-period optimization."""
        config = MultiPeriodConfig(
            n_periods=self.n_periods,
            risk_aversion=2.5,
            transaction_cost=0.001
        )

        optimizer = MultiPeriodOptimizer(config)

        returns_path = np.tile(self.expected_returns, (self.n_periods, 1))
        cov_path = np.tile(self.covariance, (self.n_periods, 1, 1))

        result = optimizer.deterministic_multi_period(
            returns_path,
            cov_path,
            initial_wealth=100.0
        )

        assert 'weights_trajectory' in result
        assert 'wealth_trajectory' in result
        assert len(result['wealth_trajectory']) == self.n_periods + 1

    def test_threshold_rebalancing(self):
        """Test threshold-based rebalancing."""
        target = np.array([0.4, 0.4, 0.2])
        rebalancer = ThresholdRebalancingPolicy(
            target_weights=target,
            threshold=0.05
        )

        # Small drift - should not rebalance
        current_small_drift = np.array([0.42, 0.39, 0.19])
        assert not rebalancer.should_rebalance(current_small_drift)

        # Large drift - should rebalance
        current_large_drift = np.array([0.50, 0.35, 0.15])
        assert rebalancer.should_rebalance(current_large_drift)


class TestShortSellingConstraints:
    """Test short-selling and leverage constraints."""

    def setup_method(self):
        """Setup test data."""
        np.random.seed(42)
        self.assets = ['A', 'B', 'C', 'D', 'E']

        self.expected_returns = pd.Series(
            [0.10, 0.12, 0.08, 0.15, 0.09],
            index=self.assets
        )

        volatilities = np.array([0.15, 0.20, 0.12, 0.25, 0.18])
        correlation = np.eye(5)
        for i in range(5):
            for j in range(i+1, 5):
                correlation[i, j] = correlation[j, i] = 0.3

        self.covariance = pd.DataFrame(
            np.outer(volatilities, volatilities) * correlation,
            index=self.assets,
            columns=self.assets
        )

    def test_long_only_constraint(self):
        """Test traditional long-only optimization."""
        config = OptimizationConfig(
            risk_aversion=2.5,
            allow_short_selling=False,
            max_assets=5
        )

        optimizer = MIOOptimizer(config)
        weights = optimizer.optimize(self.expected_returns, self.covariance)

        assert all(weights >= -1e-6)  # All weights non-negative
        assert np.abs(weights.sum() - 1.0) < 1e-4

    def test_short_selling_enabled(self):
        """Test optimization with short-selling."""
        config = OptimizationConfig(
            risk_aversion=2.5,
            allow_short_selling=True,
            max_short_weight=0.20,
            max_leverage=1.5,
            net_exposure=1.0
        )

        optimizer = MIOOptimizer(config)
        weights = optimizer.optimize(self.expected_returns, self.covariance)

        # Check solution exists
        assert optimizer.solution['status'] == 'optimal'

        # Check constraints
        long_weights = optimizer.solution['long_weights']
        short_weights = optimizer.solution['short_weights']

        gross_exposure = long_weights.sum() + short_weights.sum()
        net_exposure = long_weights.sum() - short_weights.sum()

        assert gross_exposure <= config.max_leverage + 1e-4
        assert np.abs(net_exposure - config.net_exposure) < 1e-4


def run_all_tests():
    """Run all tests."""
    print("Running Advanced Features Tests...")
    print("=" * 60)

    # Run tests
    pytest.main([__file__, '-v', '--tb=short'])


if __name__ == "__main__":
    run_all_tests()
