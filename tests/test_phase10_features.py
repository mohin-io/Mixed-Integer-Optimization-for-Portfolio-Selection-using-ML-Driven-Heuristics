"""
Tests for Phase 10 advanced features.

Tests:
- Reinforcement Learning for rebalancing
- ESG scoring and constraints
- Transformer models
- Broker API integration
- WebSocket streaming
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Import modules
from src.data.esg_scorer import (
    ESGDataProvider,
    ESGConstrainedOptimizer,
    ESGFactor,
    SustainableInvestingMetrics
)
from src.api.broker_integration import (
    AlpacaBroker,
    BrokerConfig,
    PortfolioRebalancer,
    Order,
    Position
)

# Conditional imports for optional dependencies
try:
    from src.optimization.rl_rebalancer import (
        PortfolioEnv,
        RLRebalancer,
        RLConfig
    )
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False

try:
    from src.forecasting.transformer_forecast import (
        TransformerForecasterWrapper,
        TimeSeriesDataset
    )
    TRANSFORMER_AVAILABLE = True
except ImportError:
    TRANSFORMER_AVAILABLE = False


class TestESGScoring:
    """Test ESG scoring and integration."""

    def setup_method(self):
        """Setup test data."""
        self.tickers = ['AAPL', 'MSFT', 'GOOGL', 'XOM', 'TSLA']
        self.provider = ESGDataProvider(data_source='synthetic')

    def test_esg_data_generation(self):
        """Test ESG score generation."""
        esg_scores = self.provider.fetch_esg_scores(self.tickers)

        assert len(esg_scores) == len(self.tickers)

        for ticker in self.tickers:
            assert ticker in esg_scores
            score = esg_scores[ticker]

            assert 0 <= score.environmental_score <= 100
            assert 0 <= score.social_score <= 100
            assert 0 <= score.governance_score <= 100
            assert 0 <= score.total_score <= 100

    def test_esg_filtering(self):
        """Test ESG-based filtering."""
        esg_scores = self.provider.fetch_esg_scores(self.tickers)
        optimizer = ESGConstrainedOptimizer(min_esg_score=60.0)

        filtered = optimizer.filter_by_esg(self.tickers, esg_scores, min_score=65)

        assert isinstance(filtered, list)
        assert all(esg_scores[t].total_score >= 65 for t in filtered)

    def test_portfolio_esg_calculation(self):
        """Test portfolio-level ESG score calculation."""
        esg_scores = self.provider.fetch_esg_scores(self.tickers)
        optimizer = ESGConstrainedOptimizer()

        weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

        portfolio_esg = optimizer.compute_portfolio_esg(weights, esg_scores, self.tickers)

        assert 'environmental' in portfolio_esg
        assert 'social' in portfolio_esg
        assert 'governance' in portfolio_esg
        assert 'total' in portfolio_esg
        assert 0 <= portfolio_esg['total'] <= 100

    def test_esg_constrained_optimization(self):
        """Test portfolio optimization with ESG constraints."""
        esg_scores = self.provider.fetch_esg_scores(self.tickers)

        np.random.seed(42)
        expected_returns = pd.Series(
            np.random.uniform(0.08, 0.15, len(self.tickers)),
            index=self.tickers
        )

        volatilities = np.random.uniform(0.15, 0.30, len(self.tickers))
        corr = np.eye(len(self.tickers))
        covariance = pd.DataFrame(
            np.outer(volatilities, volatilities) * corr,
            index=self.tickers,
            columns=self.tickers
        )

        optimizer = ESGConstrainedOptimizer(min_esg_score=50.0)
        result = optimizer.optimize_with_esg_constraint(
            expected_returns,
            covariance,
            esg_scores
        )

        assert 'weights' in result
        assert 'esg_scores' in result
        assert len(result['weights']) == len(self.tickers)
        assert np.abs(result['weights'].sum() - 1.0) < 1e-4

    def test_esg_factor_creation(self):
        """Test ESG factor creation."""
        esg_scores = self.provider.fetch_esg_scores(self.tickers)

        np.random.seed(42)
        returns = pd.DataFrame(
            np.random.randn(252, len(self.tickers)) * 0.01,
            columns=self.tickers
        )

        esg_factor_model = ESGFactor()
        esg_factor = esg_factor_model.create_esg_factor(
            returns,
            esg_scores,
            method='long_short'
        )

        assert isinstance(esg_factor, pd.Series)
        assert len(esg_factor) == len(returns)

    def test_sustainable_metrics(self):
        """Test sustainable investing metrics."""
        weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        carbon_intensity = {t: np.random.uniform(50, 200) for t in self.tickers}

        footprint = SustainableInvestingMetrics.carbon_footprint(
            weights,
            carbon_intensity,
            self.tickers
        )

        assert isinstance(footprint, float)
        assert footprint > 0


class TestBrokerIntegration:
    """Test broker API integration."""

    def setup_method(self):
        """Setup broker connection."""
        config = BrokerConfig(
            api_key="test_key",
            api_secret="test_secret"
        )
        self.broker = AlpacaBroker(config)

    def test_account_info(self):
        """Test getting account information."""
        account = self.broker.get_account()

        assert 'equity' in account
        assert 'cash' in account
        assert 'buying_power' in account
        assert account['equity'] > 0

    def test_positions(self):
        """Test getting positions."""
        positions = self.broker.get_positions()

        assert isinstance(positions, list)
        if positions:
            p = positions[0]
            assert isinstance(p, Position)
            assert hasattr(p, 'symbol')
            assert hasattr(p, 'qty')
            assert hasattr(p, 'current_price')

    def test_position_weights(self):
        """Test position weight calculation."""
        weights = self.broker.get_position_weights()

        assert isinstance(weights, pd.Series)
        if len(weights) > 0:
            assert np.abs(weights.sum() - 1.0) < 1e-4

    def test_order_submission(self):
        """Test order submission."""
        order = Order(
            symbol='AAPL',
            qty=10,
            side='buy',
            order_type='market'
        )

        result = self.broker.submit_order(order)

        assert 'id' in result
        assert 'symbol' in result
        assert result['symbol'] == 'AAPL'

    def test_market_data(self):
        """Test market data fetching."""
        data = self.broker.get_market_data(['AAPL', 'MSFT'], limit=5)

        assert isinstance(data, pd.DataFrame)
        assert not data.empty

    def test_rebalancing(self):
        """Test portfolio rebalancing."""
        rebalancer = PortfolioRebalancer(self.broker)

        target = pd.Series({
            'AAPL': 0.50,
            'MSFT': 0.50
        })

        result = rebalancer.rebalance_to_target(target, dry_run=True)

        assert 'trades' in result
        assert 'total_turnover' in result
        assert 'estimated_cost' in result
        assert isinstance(result['trades'], list)


@pytest.mark.skipif(not RL_AVAILABLE, reason="PyTorch not available")
class TestRLRebalancing:
    """Test Reinforcement Learning rebalancing."""

    def setup_method(self):
        """Setup RL environment."""
        np.random.seed(42)
        n_periods = 100
        n_assets = 3

        returns_data = np.random.randn(n_periods, n_assets) * 0.01
        self.returns = pd.DataFrame(
            returns_data,
            columns=['Asset_A', 'Asset_B', 'Asset_C']
        )

        initial_weights = np.array([1/3, 1/3, 1/3])
        self.env = PortfolioEnv(
            returns=self.returns,
            initial_weights=initial_weights
        )

    def test_environment_initialization(self):
        """Test RL environment initialization."""
        assert self.env.n_periods == len(self.returns)
        assert self.env.n_assets == 3
        assert self.env.wealth == 1.0

    def test_environment_reset(self):
        """Test environment reset."""
        state = self.env.reset()

        assert isinstance(state, np.ndarray)
        assert len(state) > 0
        assert self.env.current_step == self.env.lookback

    def test_environment_step(self):
        """Test environment step."""
        state = self.env.reset()
        next_state, reward, done, info = self.env.step(0)  # Hold action

        assert isinstance(next_state, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert 'wealth' in info
        assert 'portfolio_return' in info

    def test_rl_agent_initialization(self):
        """Test RL agent initialization."""
        config = RLConfig(
            state_dim=50,
            action_dim=5,
            learning_rate=1e-3
        )

        agent = RLRebalancer(config)

        assert agent.config.learning_rate == 1e-3
        assert agent.epsilon == config.epsilon_start


@pytest.mark.skipif(not TRANSFORMER_AVAILABLE, reason="PyTorch not available")
class TestTransformerForecasting:
    """Test Transformer models."""

    def setup_method(self):
        """Setup test data."""
        np.random.seed(42)
        n_periods = 200
        n_assets = 3

        self.returns = pd.DataFrame(
            np.random.randn(n_periods, n_assets) * 0.01,
            columns=['Asset_A', 'Asset_B', 'Asset_C']
        )

    def test_dataset_creation(self):
        """Test time series dataset creation."""
        dataset = TimeSeriesDataset(
            self.returns.values,
            lookback=30,
            horizon=1
        )

        assert len(dataset) > 0

        X, y = dataset[0]
        assert X.shape[0] == 30
        assert y.shape[0] == 1

    def test_transformer_initialization(self):
        """Test Transformer model initialization."""
        forecaster = TransformerForecasterWrapper(
            model_type='transformer',
            lookback_window=30,
            d_model=32,
            nhead=2,
            epochs=2
        )

        assert forecaster.lookback_window == 30
        assert forecaster.d_model == 32

    def test_tft_initialization(self):
        """Test TFT model initialization."""
        forecaster = TransformerForecasterWrapper(
            model_type='tft',
            lookback_window=30,
            d_model=32,
            epochs=2
        )

        assert forecaster.model_type == 'tft'


def run_all_tests():
    """Run all tests."""
    print("Running Phase 10 Features Tests...")
    print("=" * 60)

    pytest.main([__file__, '-v', '--tb=short'])


if __name__ == "__main__":
    run_all_tests()
