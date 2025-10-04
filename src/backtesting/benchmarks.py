"""
Benchmark Strategies for Portfolio Optimization Comparison

Provides reference implementations for:
1. Equal Weight (1/N)
2. Market Cap Weight
3. Risk Parity
4. Maximum Sharpe Ratio
5. Minimum Variance
6. Maximum Diversification
7. Buy and Hold
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Callable
from scipy.optimize import minimize
import warnings


class BenchmarkStrategies:
    """Collection of benchmark portfolio strategies."""

    @staticmethod
    def equal_weight(
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        **kwargs
    ) -> pd.Series:
        """
        Equal weight (1/N) portfolio.

        Args:
            expected_returns: Expected returns (unused)
            cov_matrix: Covariance matrix (unused)

        Returns:
            Equal weights for all assets
        """
        n = len(expected_returns)
        return pd.Series(1.0 / n, index=expected_returns.index)

    @staticmethod
    def market_cap_weight(
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        market_caps: Optional[pd.Series] = None,
        **kwargs
    ) -> pd.Series:
        """
        Market capitalization weighted portfolio.

        Args:
            expected_returns: Expected returns (unused)
            cov_matrix: Covariance matrix (unused)
            market_caps: Market capitalizations

        Returns:
            Market cap weighted portfolio
        """
        if market_caps is None:
            # Default to equal weight if no market caps provided
            return BenchmarkStrategies.equal_weight(expected_returns, cov_matrix)

        # Normalize market caps to weights
        weights = market_caps / market_caps.sum()
        return weights

    @staticmethod
    def max_sharpe(
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        risk_free_rate: float = 0.02,
        **kwargs
    ) -> pd.Series:
        """
        Maximum Sharpe ratio portfolio.

        Args:
            expected_returns: Expected returns
            cov_matrix: Covariance matrix
            risk_free_rate: Risk-free rate

        Returns:
            Portfolio weights maximizing Sharpe ratio
        """
        n = len(expected_returns)

        def neg_sharpe(weights):
            port_return = np.dot(weights, expected_returns.values)
            port_vol = np.sqrt(weights @ cov_matrix.values @ weights)

            if port_vol > 1e-8:
                sharpe = (port_return - risk_free_rate) / port_vol
                return -sharpe
            return 1e10

        # Constraints: weights sum to 1
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

        # Bounds: long-only (0 to 1)
        bounds = tuple((0, 1) for _ in range(n))

        # Initial guess: equal weight
        x0 = np.ones(n) / n

        # Optimize
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(
                neg_sharpe,
                x0=x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )

        if result.success:
            return pd.Series(result.x, index=expected_returns.index)
        else:
            # Fallback to equal weight
            return BenchmarkStrategies.equal_weight(expected_returns, cov_matrix)

    @staticmethod
    def min_variance(
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        **kwargs
    ) -> pd.Series:
        """
        Minimum variance portfolio.

        Args:
            expected_returns: Expected returns (unused)
            cov_matrix: Covariance matrix

        Returns:
            Portfolio weights minimizing variance
        """
        n = len(expected_returns)

        def portfolio_variance(weights):
            return weights @ cov_matrix.values @ weights

        # Constraints: weights sum to 1
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

        # Bounds: long-only
        bounds = tuple((0, 1) for _ in range(n))

        # Initial guess
        x0 = np.ones(n) / n

        # Optimize
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(
                portfolio_variance,
                x0=x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )

        if result.success:
            return pd.Series(result.x, index=expected_returns.index)
        else:
            return BenchmarkStrategies.equal_weight(expected_returns, cov_matrix)

    @staticmethod
    def risk_parity(
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        **kwargs
    ) -> pd.Series:
        """
        Risk parity portfolio (equal risk contribution).

        Args:
            expected_returns: Expected returns (unused)
            cov_matrix: Covariance matrix

        Returns:
            Risk parity portfolio weights
        """
        n = len(expected_returns)

        def risk_parity_objective(weights):
            """Minimize difference in risk contributions."""
            # Portfolio volatility
            port_vol = np.sqrt(weights @ cov_matrix.values @ weights)

            # Marginal risk contributions
            marginal_contrib = cov_matrix.values @ weights

            # Risk contributions
            risk_contrib = weights * marginal_contrib / port_vol if port_vol > 0 else weights

            # Target: equal risk contribution
            target_risk = port_vol / n if port_vol > 0 else 0

            # Sum of squared differences from target
            return np.sum((risk_contrib - target_risk) ** 2)

        # Constraints
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

        # Bounds
        bounds = tuple((1e-6, 1) for _ in range(n))

        # Initial guess
        x0 = np.ones(n) / n

        # Optimize
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(
                risk_parity_objective,
                x0=x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )

        if result.success:
            weights = result.x
            # Normalize
            return pd.Series(weights / weights.sum(), index=expected_returns.index)
        else:
            # Fallback: inverse volatility weighting
            inv_vol = 1.0 / np.sqrt(np.diag(cov_matrix.values))
            weights = inv_vol / inv_vol.sum()
            return pd.Series(weights, index=expected_returns.index)

    @staticmethod
    def max_diversification(
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        **kwargs
    ) -> pd.Series:
        """
        Maximum diversification portfolio.

        Maximizes diversification ratio:
            DR = (w^T * σ) / sqrt(w^T * Σ * w)

        Args:
            expected_returns: Expected returns (unused)
            cov_matrix: Covariance matrix

        Returns:
            Maximum diversification portfolio weights
        """
        n = len(expected_returns)

        # Individual volatilities
        individual_vols = np.sqrt(np.diag(cov_matrix.values))

        def neg_diversification_ratio(weights):
            """Negative diversification ratio to minimize."""
            weighted_vol = np.dot(weights, individual_vols)
            port_vol = np.sqrt(weights @ cov_matrix.values @ weights)

            if port_vol > 1e-8:
                dr = weighted_vol / port_vol
                return -dr
            return 1e10

        # Constraints
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

        # Bounds
        bounds = tuple((0, 1) for _ in range(n))

        # Initial guess
        x0 = np.ones(n) / n

        # Optimize
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(
                neg_diversification_ratio,
                x0=x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )

        if result.success:
            return pd.Series(result.x, index=expected_returns.index)
        else:
            return BenchmarkStrategies.equal_weight(expected_returns, cov_matrix)

    @staticmethod
    def concentrated(
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        max_assets: int = 5,
        **kwargs
    ) -> pd.Series:
        """
        Concentrated portfolio with cardinality constraint.

        Select top N assets by Sharpe ratio, then optimize.

        Args:
            expected_returns: Expected returns
            cov_matrix: Covariance matrix
            max_assets: Maximum number of assets

        Returns:
            Concentrated portfolio weights
        """
        # Calculate individual Sharpe ratios
        individual_vols = np.sqrt(np.diag(cov_matrix.values))
        sharpe_ratios = expected_returns.values / individual_vols

        # Select top assets
        top_indices = np.argsort(sharpe_ratios)[-max_assets:]
        selected_assets = expected_returns.index[top_indices]

        # Subset data
        subset_returns = expected_returns[selected_assets]
        subset_cov = cov_matrix.loc[selected_assets, selected_assets]

        # Optimize on subset
        subset_weights = BenchmarkStrategies.max_sharpe(subset_returns, subset_cov)

        # Create full weight vector
        full_weights = pd.Series(0.0, index=expected_returns.index)
        full_weights[selected_assets] = subset_weights

        return full_weights

    @staticmethod
    def mean_variance(
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        risk_aversion: float = 2.5,
        **kwargs
    ) -> pd.Series:
        """
        Mean-variance optimization with risk aversion parameter.

        Maximize: μ^T * w - λ/2 * w^T * Σ * w

        Args:
            expected_returns: Expected returns
            cov_matrix: Covariance matrix
            risk_aversion: Risk aversion parameter (λ)

        Returns:
            Mean-variance optimal portfolio
        """
        n = len(expected_returns)

        def neg_utility(weights):
            """Negative utility to minimize."""
            port_return = np.dot(weights, expected_returns.values)
            port_variance = weights @ cov_matrix.values @ weights
            utility = port_return - (risk_aversion / 2) * port_variance
            return -utility

        # Constraints
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

        # Bounds
        bounds = tuple((0, 1) for _ in range(n))

        # Initial guess
        x0 = np.ones(n) / n

        # Optimize
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(
                neg_utility,
                x0=x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )

        if result.success:
            return pd.Series(result.x, index=expected_returns.index)
        else:
            return BenchmarkStrategies.equal_weight(expected_returns, cov_matrix)


def get_all_benchmark_strategies() -> Dict[str, Callable]:
    """
    Get dictionary of all benchmark strategies.

    Returns:
        Dict mapping strategy names to optimizer functions
    """
    return {
        'Equal Weight': BenchmarkStrategies.equal_weight,
        'Max Sharpe': BenchmarkStrategies.max_sharpe,
        'Min Variance': BenchmarkStrategies.min_variance,
        'Risk Parity': BenchmarkStrategies.risk_parity,
        'Max Diversification': BenchmarkStrategies.max_diversification,
        'Concentrated (5 assets)': lambda er, cov, **kw: BenchmarkStrategies.concentrated(
            er, cov, max_assets=5, **kw
        ),
        'Mean-Variance (λ=2.5)': lambda er, cov, **kw: BenchmarkStrategies.mean_variance(
            er, cov, risk_aversion=2.5, **kw
        )
    }


def get_core_strategies() -> Dict[str, Callable]:
    """
    Get core benchmark strategies for quick comparison.

    Returns:
        Dict with 4 essential strategies
    """
    return {
        'Equal Weight': BenchmarkStrategies.equal_weight,
        'Max Sharpe': BenchmarkStrategies.max_sharpe,
        'Min Variance': BenchmarkStrategies.min_variance,
        'Risk Parity': BenchmarkStrategies.risk_parity
    }


if __name__ == "__main__":
    # Example usage
    from ..data.loader import AssetDataLoader
    from .engine import Backtester, BacktestConfig

    print("Benchmark Strategies Demo\n")

    # Load data
    loader = AssetDataLoader()
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'JPM', 'V', 'WMT']

    try:
        prices = loader.fetch_prices(tickers, '2020-01-01', '2023-12-31')

        # Calculate sample returns and covariance for demonstration
        returns = prices.pct_change().dropna()
        expected_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252

        print("Testing all benchmark strategies:\n")

        strategies = get_all_benchmark_strategies()

        for name, strategy_func in strategies.items():
            weights = strategy_func(expected_returns, cov_matrix)

            # Calculate metrics
            port_return = np.dot(weights, expected_returns)
            port_vol = np.sqrt(weights @ cov_matrix @ weights)
            sharpe = port_return / port_vol if port_vol > 0 else 0
            n_assets = (weights > 0.01).sum()

            print(f"{name}:")
            print(f"  Return: {port_return*100:.2f}%")
            print(f"  Volatility: {port_vol*100:.2f}%")
            print(f"  Sharpe: {sharpe:.3f}")
            print(f"  Assets: {n_assets}")
            print()

        # Run full backtest comparison
        print("\n" + "="*60)
        print("BACKTESTING COMPARISON")
        print("="*60 + "\n")

        config = BacktestConfig(
            train_window=252,
            rebalance_freq=21,
            transaction_cost=0.001,
            verbose=False
        )

        backtester = Backtester(config=config)

        comparison = backtester.compare_strategies(
            prices=prices,
            strategies=get_core_strategies(),
            start_date='2021-01-01'
        )

        print("\nBacktest completed successfully!")

    except Exception as e:
        print(f"Error in demo: {str(e)}")
        import traceback
        traceback.print_exc()
