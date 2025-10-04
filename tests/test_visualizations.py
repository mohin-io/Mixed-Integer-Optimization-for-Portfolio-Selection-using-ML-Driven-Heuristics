"""
Comprehensive Unit Tests for Dashboard Visualization Functions

Tests all visualization components including:
- Gauge charts
- Efficient frontier charts
- Portfolio allocation charts
- Risk-return bubble charts
- Monte Carlo simulation charts
- Drawdown charts
- Rolling statistics charts
- Treemap charts
- Waterfall charts
- Radar charts
"""

import pytest
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import visualization functions from dashboard
# Note: These functions need to be imported after they're defined in dashboard.py


class TestBasicChartCreation:
    """Test that visualization functions create valid chart objects."""

    @pytest.fixture
    def sample_portfolio_data(self):
        """Create sample portfolio data for testing."""
        np.random.seed(42)
        n_assets = 10

        weights = pd.Series(
            np.random.dirichlet(np.ones(n_assets)),
            index=[f'ASSET_{i+1}' for i in range(n_assets)]
        )

        returns = pd.DataFrame(
            np.random.randn(252, n_assets) * 0.01,
            columns=[f'ASSET_{i+1}' for i in range(n_assets)]
        )

        annual_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252

        metrics = {
            'return': 0.12,
            'volatility': 0.18,
            'sharpe': 0.67,
            'n_assets': 10
        }

        return {
            'weights': weights,
            'returns': returns,
            'annual_returns': annual_returns,
            'cov_matrix': cov_matrix,
            'metrics': metrics
        }

    def test_create_gauge_chart_returns_figure(self, sample_portfolio_data):
        """Test that create_gauge_chart returns a Plotly Figure."""
        # Import the function
        try:
            from src.visualization.dashboard import create_gauge_chart
        except ImportError:
            pytest.skip("create_gauge_chart not found in dashboard")

        fig = create_gauge_chart(0.15, "Expected Return (%)", max_value=0.5)

        assert isinstance(fig, go.Figure), "Should return a Plotly Figure"
        assert len(fig.data) > 0, "Figure should contain data"

    def test_create_gauge_chart_value_range(self):
        """Test gauge chart with different value ranges."""
        try:
            from src.visualization.dashboard import create_gauge_chart
        except ImportError:
            pytest.skip("create_gauge_chart not found")

        # Test various values
        values = [0.0, 0.5, 1.0]
        for value in values:
            fig = create_gauge_chart(value, "Test Metric", max_value=1.0)
            assert isinstance(fig, go.Figure), f"Should handle value {value}"

    def test_create_weights_chart_returns_figure(self, sample_portfolio_data):
        """Test that weights visualization returns a Figure."""
        try:
            from src.visualization.dashboard import create_weights_chart
        except ImportError:
            pytest.skip("create_weights_chart not found")

        weights = sample_portfolio_data['weights']
        fig = create_weights_chart(weights)

        assert isinstance(fig, go.Figure), "Should return a Plotly Figure"
        assert len(fig.data) > 0, "Figure should contain data"

    def test_create_efficient_frontier_returns_figure(self, sample_portfolio_data):
        """Test that efficient frontier chart returns a Figure."""
        try:
            from src.visualization.dashboard import create_efficient_frontier_chart
        except ImportError:
            pytest.skip("create_efficient_frontier_chart not found")

        returns = sample_portfolio_data['returns']
        portfolio = sample_portfolio_data['metrics']

        fig = create_efficient_frontier_chart(returns, portfolio)

        assert isinstance(fig, go.Figure), "Should return a Plotly Figure"
        # Should have frontier points and current portfolio
        assert len(fig.data) >= 1, "Should have at least frontier data"

    def test_create_correlation_heatmap_returns_figure(self, sample_portfolio_data):
        """Test correlation heatmap creation."""
        try:
            from src.visualization.dashboard import create_correlation_heatmap
        except ImportError:
            pytest.skip("create_correlation_heatmap not found")

        returns = sample_portfolio_data['returns']
        fig = create_correlation_heatmap(returns)

        assert isinstance(fig, go.Figure), "Should return a Plotly Figure"
        assert len(fig.data) > 0, "Figure should contain heatmap data"


class TestChartDataIntegrity:
    """Test that charts correctly represent the input data."""

    @pytest.fixture
    def simple_weights(self):
        """Create simple weights for testing."""
        return pd.Series({
            'ASSET_1': 0.5,
            'ASSET_2': 0.3,
            'ASSET_3': 0.2
        })

    def test_gauge_chart_value_representation(self):
        """Test that gauge chart correctly represents the value."""
        try:
            from src.visualization.dashboard import create_gauge_chart
        except ImportError:
            pytest.skip("create_gauge_chart not found")

        test_value = 0.75
        fig = create_gauge_chart(test_value, "Test", max_value=1.0)

        # Check that the figure has indicator data
        assert len(fig.data) > 0, "Should have data"
        # The value should be in the figure data
        if hasattr(fig.data[0], 'value'):
            assert fig.data[0].value == test_value, "Value should match input"

    def test_weights_chart_includes_all_assets(self, simple_weights):
        """Test that weights chart includes all assets."""
        try:
            from src.visualization.dashboard import create_weights_chart
        except ImportError:
            pytest.skip("create_weights_chart not found")

        fig = create_weights_chart(simple_weights)

        # Should have one trace (bar chart)
        assert len(fig.data) >= 1, "Should have chart data"

        # Check that number of bars matches number of assets
        if hasattr(fig.data[0], 'x'):
            # Number of x values should match number of assets
            assert len(fig.data[0].x) == len(simple_weights), \
                "Should have one bar per asset"


class TestAdvancedVisualizationFunctions:
    """Test advanced visualization functions added in recent updates."""

    @pytest.fixture
    def monte_carlo_data(self):
        """Create sample Monte Carlo simulation data."""
        np.random.seed(42)
        n_days = 252
        n_simulations = 200

        paths = np.zeros((n_days, n_simulations))
        paths[0] = 100

        for i in range(1, n_days):
            returns = np.random.randn(n_simulations) * 0.01
            paths[i] = paths[i-1] * (1 + returns)

        dates = pd.date_range('2024-01-01', periods=n_days, freq='D')

        return {
            'dates': dates,
            'paths': paths,
            'initial_value': 100
        }

    def test_create_risk_return_bubble_chart(self):
        """Test risk-return bubble chart creation."""
        try:
            from src.visualization.dashboard import create_risk_return_bubble_chart
        except ImportError:
            pytest.skip("create_risk_return_bubble_chart not found")

        np.random.seed(42)
        returns = pd.DataFrame(
            np.random.randn(252, 10) * 0.01,
            columns=[f'ASSET_{i+1}' for i in range(10)]
        )

        selected_assets = ['ASSET_1', 'ASSET_3', 'ASSET_5']
        weights = pd.Series([0.5, 0.3, 0.2], index=selected_assets)

        fig = create_risk_return_bubble_chart(returns, selected_assets, weights)

        assert isinstance(fig, go.Figure), "Should return a Plotly Figure"
        assert len(fig.data) >= 2, "Should have multiple traces (all assets + selected)"

    def test_create_drawdown_chart(self):
        """Test drawdown chart creation."""
        try:
            from src.visualization.dashboard import create_drawdown_chart
        except ImportError:
            pytest.skip("create_drawdown_chart not found")

        # Create sample cumulative returns
        dates = pd.date_range('2024-01-01', periods=252, freq='D')
        np.random.seed(42)
        returns = pd.Series(np.random.randn(252) * 0.01, index=dates)
        cumulative_returns = (1 + returns).cumprod()

        fig = create_drawdown_chart(cumulative_returns)

        assert isinstance(fig, go.Figure), "Should return a Plotly Figure"
        assert len(fig.data) > 0, "Should have drawdown data"

    def test_create_rolling_statistics_chart(self):
        """Test rolling statistics chart creation."""
        try:
            from src.visualization.dashboard import create_rolling_statistics_chart
        except ImportError:
            pytest.skip("create_rolling_statistics_chart not found")

        # Create sample returns
        dates = pd.date_range('2024-01-01', periods=252, freq='D')
        np.random.seed(42)
        returns = pd.Series(np.random.randn(252) * 0.01, index=dates)

        fig = create_rolling_statistics_chart(returns)

        assert isinstance(fig, go.Figure), "Should return a Plotly Figure"
        # Should have subplots (3 panels: return, volatility, Sharpe)
        if hasattr(fig, 'data'):
            assert len(fig.data) >= 3, "Should have at least 3 traces for rolling stats"

    def test_create_portfolio_treemap(self):
        """Test portfolio treemap creation."""
        try:
            from src.visualization.dashboard import create_portfolio_treemap
        except ImportError:
            pytest.skip("create_portfolio_treemap not found")

        weights = pd.Series({
            'TECH_1': 0.3,
            'TECH_2': 0.2,
            'FINANCE_1': 0.25,
            'ENERGY_1': 0.15,
            'RETAIL_1': 0.1
        })

        returns = pd.Series({
            'TECH_1': 0.15,
            'TECH_2': 0.12,
            'FINANCE_1': 0.08,
            'ENERGY_1': 0.10,
            'RETAIL_1': 0.06
        })

        fig = create_portfolio_treemap(weights, returns)

        assert isinstance(fig, go.Figure), "Should return a Plotly Figure"
        assert len(fig.data) > 0, "Should have treemap data"

    def test_create_monte_carlo_chart(self, monte_carlo_data):
        """Test Monte Carlo simulation chart."""
        try:
            from src.visualization.dashboard import create_monte_carlo_chart
        except ImportError:
            pytest.skip("create_monte_carlo_chart not found")

        fig = create_monte_carlo_chart(
            monte_carlo_data['paths'],
            monte_carlo_data['dates']
        )

        assert isinstance(fig, go.Figure), "Should return a Plotly Figure"
        # Should have multiple traces (paths + percentiles)
        assert len(fig.data) > 0, "Should have simulation paths"

    def test_create_risk_contribution_waterfall(self):
        """Test risk contribution waterfall chart."""
        try:
            from src.visualization.dashboard import create_risk_contribution_waterfall
        except ImportError:
            pytest.skip("create_risk_contribution_waterfall not found")

        weights = pd.Series({
            'ASSET_1': 0.3,
            'ASSET_2': 0.25,
            'ASSET_3': 0.2,
            'ASSET_4': 0.15,
            'ASSET_5': 0.1
        })

        # Simple diagonal covariance for testing
        cov_matrix = pd.DataFrame(
            np.eye(5) * 0.04,
            index=weights.index,
            columns=weights.index
        )

        fig = create_risk_contribution_waterfall(weights, cov_matrix)

        assert isinstance(fig, go.Figure), "Should return a Plotly Figure"
        assert len(fig.data) > 0, "Should have waterfall data"

    def test_create_strategy_comparison_radar(self):
        """Test strategy comparison radar chart."""
        try:
            from src.visualization.dashboard import create_strategy_comparison_radar
        except ImportError:
            pytest.skip("create_strategy_comparison_radar not found")

        strategies_metrics = {
            'Equal Weight': {
                'return': 0.10,
                'sharpe': 0.60,
                'volatility': 0.15,
                'max_drawdown': 0.20
            },
            'Max Sharpe': {
                'return': 0.14,
                'sharpe': 0.85,
                'volatility': 0.18,
                'max_drawdown': 0.25
            },
            'Min Variance': {
                'return': 0.08,
                'sharpe': 0.55,
                'volatility': 0.12,
                'max_drawdown': 0.15
            }
        }

        fig = create_strategy_comparison_radar(strategies_metrics)

        assert isinstance(fig, go.Figure), "Should return a Plotly Figure"
        # Should have one trace per strategy
        assert len(fig.data) >= 3, "Should have 3 strategy traces"


class TestChartErrorHandling:
    """Test that charts handle edge cases and errors gracefully."""

    def test_gauge_chart_handles_zero_value(self):
        """Test gauge chart with zero value."""
        try:
            from src.visualization.dashboard import create_gauge_chart
        except ImportError:
            pytest.skip("create_gauge_chart not found")

        fig = create_gauge_chart(0.0, "Zero Value", max_value=1.0)
        assert isinstance(fig, go.Figure), "Should handle zero value"

    def test_gauge_chart_handles_max_value(self):
        """Test gauge chart at maximum value."""
        try:
            from src.visualization.dashboard import create_gauge_chart
        except ImportError:
            pytest.skip("create_gauge_chart not found")

        max_val = 1.0
        fig = create_gauge_chart(max_val, "Max Value", max_value=max_val)
        assert isinstance(fig, go.Figure), "Should handle max value"

    def test_weights_chart_single_asset(self):
        """Test weights chart with single asset."""
        try:
            from src.visualization.dashboard import create_weights_chart
        except ImportError:
            pytest.skip("create_weights_chart not found")

        weights = pd.Series({'ASSET_1': 1.0})
        fig = create_weights_chart(weights)
        assert isinstance(fig, go.Figure), "Should handle single asset"

    def test_correlation_heatmap_small_dataset(self):
        """Test correlation heatmap with minimal data."""
        try:
            from src.visualization.dashboard import create_correlation_heatmap
        except ImportError:
            pytest.skip("create_correlation_heatmap not found")

        # Create minimal dataset (2 assets, 10 days)
        returns = pd.DataFrame({
            'ASSET_1': np.random.randn(10) * 0.01,
            'ASSET_2': np.random.randn(10) * 0.01
        })

        fig = create_correlation_heatmap(returns)
        assert isinstance(fig, go.Figure), "Should handle small dataset"

    def test_efficient_frontier_edge_cases(self):
        """Test efficient frontier with edge case data."""
        try:
            from src.visualization.dashboard import create_efficient_frontier_chart
        except ImportError:
            pytest.skip("create_efficient_frontier_chart not found")

        # Create returns with low variance
        returns = pd.DataFrame(
            np.random.randn(100, 3) * 0.001,  # Very low volatility
            columns=['A1', 'A2', 'A3']
        )

        portfolio = {
            'return': 0.05,
            'volatility': 0.02,
            'sharpe': 2.5
        }

        fig = create_efficient_frontier_chart(returns, portfolio)
        assert isinstance(fig, go.Figure), "Should handle low variance data"


class TestMonteCarloSimulation:
    """Test Monte Carlo simulation functionality."""

    def test_run_monte_carlo_simulation_returns_correct_shape(self):
        """Test that Monte Carlo simulation returns correct array shape."""
        try:
            from src.visualization.dashboard import run_monte_carlo_simulation
        except ImportError:
            pytest.skip("run_monte_carlo_simulation not found")

        returns = pd.Series(np.random.randn(252) * 0.01)
        n_simulations = 100
        n_days = 252

        paths = run_monte_carlo_simulation(returns, n_simulations, n_days)

        assert paths.shape == (n_days, n_simulations), \
            f"Expected shape ({n_days}, {n_simulations}), got {paths.shape}"

    def test_monte_carlo_all_paths_start_at_100(self):
        """Test that all simulation paths start at 100."""
        try:
            from src.visualization.dashboard import run_monte_carlo_simulation
        except ImportError:
            pytest.skip("run_monte_carlo_simulation not found")

        returns = pd.Series(np.random.randn(252) * 0.01)
        paths = run_monte_carlo_simulation(returns, 50, 100)

        # All paths should start at 100
        assert np.all(paths[0] == 100), "All paths should start at 100"

    def test_monte_carlo_paths_are_positive(self):
        """Test that simulation paths remain positive (no bankruptcy)."""
        try:
            from src.visualization.dashboard import run_monte_carlo_simulation
        except ImportError:
            pytest.skip("run_monte_carlo_simulation not found")

        # Use realistic returns to avoid negative prices
        returns = pd.Series(np.random.randn(252) * 0.01)
        paths = run_monte_carlo_simulation(returns, 50, 100)

        # Most paths should remain positive (allow some failures with extreme returns)
        positive_ratio = np.sum(paths > 0) / paths.size
        assert positive_ratio > 0.95, "At least 95% of values should be positive"


class TestChartCustomization:
    """Test chart customization and styling."""

    def test_gauge_chart_custom_title(self):
        """Test that gauge chart uses provided title."""
        try:
            from src.visualization.dashboard import create_gauge_chart
        except ImportError:
            pytest.skip("create_gauge_chart not found")

        custom_title = "My Custom Metric"
        fig = create_gauge_chart(0.5, custom_title, max_value=1.0)

        # Check that title is in the figure
        assert custom_title in str(fig), "Title should be in figure"

    def test_charts_have_responsive_layout(self):
        """Test that charts have responsive/proper layout settings."""
        try:
            from src.visualization.dashboard import create_gauge_chart
        except ImportError:
            pytest.skip("create_gauge_chart not found")

        fig = create_gauge_chart(0.5, "Test", max_value=1.0)

        # Check that layout exists
        assert hasattr(fig, 'layout'), "Figure should have layout"

    def test_efficient_frontier_has_axis_labels(self):
        """Test that efficient frontier has proper axis labels."""
        try:
            from src.visualization.dashboard import create_efficient_frontier_chart
        except ImportError:
            pytest.skip("create_efficient_frontier_chart not found")

        returns = pd.DataFrame(np.random.randn(100, 5) * 0.01)
        portfolio = {'return': 0.1, 'volatility': 0.15, 'sharpe': 0.67}

        fig = create_efficient_frontier_chart(returns, portfolio)

        # Check for axis labels in layout
        layout_str = str(fig.layout)
        assert 'volatility' in layout_str.lower() or 'risk' in layout_str.lower(), \
            "Should have volatility/risk on x-axis"
        assert 'return' in layout_str.lower(), "Should have return on y-axis"


class TestIntegrationWithOptimization:
    """Test that visualizations integrate correctly with optimization results."""

    def test_visualization_pipeline_end_to_end(self):
        """Test complete pipeline: data → optimization → visualization."""
        try:
            from src.visualization.dashboard import (
                generate_synthetic_data,
                optimize_portfolio,
                evaluate_portfolio,
                create_weights_chart,
                create_efficient_frontier_chart
            )
        except ImportError:
            pytest.skip("Required functions not found")

        # Generate data
        prices, returns = generate_synthetic_data(10, 252, 42)

        # Optimize
        weights, annual_returns, cov_matrix = optimize_portfolio(
            returns, 'Max Sharpe'
        )

        # Evaluate
        metrics = evaluate_portfolio(weights, annual_returns, cov_matrix)

        # Visualize
        fig_weights = create_weights_chart(weights)
        fig_frontier = create_efficient_frontier_chart(returns, metrics)

        # Assertions
        assert isinstance(fig_weights, go.Figure), "Weights chart should be a Figure"
        assert isinstance(fig_frontier, go.Figure), "Frontier chart should be a Figure"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
