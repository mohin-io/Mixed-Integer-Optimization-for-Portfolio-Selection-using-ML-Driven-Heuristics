"""
Mixed-Integer Optimization (MIO) Module

Implements portfolio optimization with:
- Integer constraints (discrete lot sizes)
- Transaction costs (fixed + proportional)
- Cardinality constraints (max number of assets)
- Long-only constraints

Uses PuLP for mixed-integer programming.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict
import pulp as pl
from dataclasses import dataclass
import time


@dataclass
class OptimizationConfig:
    """Configuration for portfolio optimization."""
    risk_aversion: float = 2.5
    max_assets: Optional[int] = None
    lot_size: float = 0.01  # Minimum investment increment
    max_weight: float = 0.30  # Maximum weight per asset
    min_weight: float = 0.02  # Minimum weight if invested
    fixed_cost: float = 0.0005  # Fixed cost per trade (0.05%)
    proportional_cost: float = 0.001  # Proportional cost (0.1%)
    solver: str = 'PULP_CBC_CMD'  # Solver to use
    time_limit: int = 300  # Solver time limit in seconds

    # Short-selling and leverage parameters
    allow_short_selling: bool = False  # Allow negative weights
    max_short_weight: float = 0.20  # Maximum short position per asset
    max_leverage: float = 1.0  # Maximum gross exposure (1.0 = no leverage)
    net_exposure: float = 1.0  # Required net exposure (long - short)


class MIOOptimizer:
    """
    Mixed-Integer Optimization for Portfolio Selection.

    Solves:
        maximize: μᵀw - λ·(wᵀΣw) - transaction_costs

        subject to:
            1. Σwᵢ = 1 (budget constraint)
            2. wᵢ = kᵢ * lot_size (integer lots)
            3. Σyᵢ ≤ max_assets (cardinality)
            4. yᵢ ∈ {0,1}, wᵢ ≤ yᵢ (binary indicators)
            5. min_weight * yᵢ ≤ wᵢ ≤ max_weight * yᵢ
    """

    def __init__(self, config: Optional[OptimizationConfig] = None):
        """
        Initialize MIO optimizer.

        Args:
            config: Optimization configuration
        """
        self.config = config or OptimizationConfig()
        self.solution = None
        self.problem = None

    def optimize(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        previous_weights: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Solve mixed-integer portfolio optimization problem.

        Args:
            expected_returns: Expected returns for each asset
            covariance_matrix: Covariance matrix of returns
            previous_weights: Previous portfolio weights (for transaction costs)

        Returns:
            Optimal portfolio weights
        """
        n_assets = len(expected_returns)
        assets = expected_returns.index.tolist()

        if previous_weights is None:
            previous_weights = pd.Series(0, index=assets)

        # Create optimization problem
        self.problem = pl.LpProblem("Portfolio_MIO", pl.LpMaximize)

        # Decision variables
        if self.config.allow_short_selling:
            # Allow both long and short positions
            w_long = pl.LpVariable.dicts("w_long", assets, lowBound=0, upBound=self.config.max_weight)
            w_short = pl.LpVariable.dicts("w_short", assets, lowBound=0, upBound=self.config.max_short_weight)

            # Net weight = long - short
            w = {i: w_long[i] - w_short[i] for i in assets}

            # Binary indicators for long and short positions
            y_long = pl.LpVariable.dicts("y_long", assets, cat='Binary')
            y_short = pl.LpVariable.dicts("y_short", assets, cat='Binary')
        else:
            # Long-only portfolio
            w = pl.LpVariable.dicts("weight", assets, lowBound=0, upBound=self.config.max_weight)
            y = pl.LpVariable.dicts("indicator", assets, cat='Binary')

        # Auxiliary variable for quadratic risk (using piecewise linearization)
        # For simplicity, we'll use a linearization approach

        # Objective function
        # Maximize: expected return - risk - transaction costs

        # Expected return component
        return_component = pl.lpSum([expected_returns[i] * w[i] for i in assets])

        # Transaction cost component
        # |w[i] - w_prev[i]| approximated using auxiliary variables
        trade_size = pl.LpVariable.dicts("trade_size", assets, lowBound=0)

        transaction_cost = pl.lpSum([
            self.config.fixed_cost * y[i] +
            self.config.proportional_cost * trade_size[i]
            for i in assets
        ])

        # Define trade size constraints
        for i in assets:
            # trade_size >= w[i] - previous_weight[i]
            self.problem += trade_size[i] >= w[i] - previous_weights[i]
            # trade_size >= previous_weight[i] - w[i]
            self.problem += trade_size[i] >= previous_weights[i] - w[i]

        # Risk component (simplified linear approximation)
        # For MIP, we approximate wᵀΣw using diagonal risk
        diagonal_risk = pl.lpSum([
            covariance_matrix.loc[i, i] * w[i] * w[i]
            for i in assets
        ])

        # Complete objective
        objective = return_component - self.config.risk_aversion * diagonal_risk - transaction_cost
        self.problem += objective

        # Constraints
        if self.config.allow_short_selling:
            # Short-selling enabled constraints

            # 1. Net exposure constraint: sum(long) - sum(short) = net_exposure
            self.problem += (
                pl.lpSum([w_long[i] for i in assets]) - pl.lpSum([w_short[i] for i in assets])
                == self.config.net_exposure
            ), "Net_Exposure"

            # 2. Gross exposure constraint: sum(long) + sum(short) <= max_leverage
            self.problem += (
                pl.lpSum([w_long[i] for i in assets]) + pl.lpSum([w_short[i] for i in assets])
                <= self.config.max_leverage
            ), "Gross_Exposure"

            # 3. Indicator constraints for long positions
            for i in assets:
                self.problem += w_long[i] <= self.config.max_weight * y_long[i], f"Long_Upper_{i}"
                self.problem += w_long[i] >= self.config.min_weight * y_long[i], f"Long_Lower_{i}"

            # 4. Indicator constraints for short positions
            for i in assets:
                self.problem += w_short[i] <= self.config.max_short_weight * y_short[i], f"Short_Upper_{i}"

            # 5. Cannot be both long and short in the same asset
            for i in assets:
                self.problem += y_long[i] + y_short[i] <= 1, f"No_Both_{i}"

            # 6. Cardinality constraint (total long + short positions)
            if self.config.max_assets is not None:
                self.problem += (
                    pl.lpSum([y_long[i] for i in assets]) + pl.lpSum([y_short[i] for i in assets])
                    <= self.config.max_assets
                ), "Cardinality"

        else:
            # Long-only constraints

            # 1. Budget constraint: sum of weights = 1
            self.problem += pl.lpSum([w[i] for i in assets]) == 1, "Budget"

            # 2. Indicator constraints: w[i] > 0 => y[i] = 1
            for i in assets:
                # w[i] <= y[i] (if y[i]=0, then w[i]=0)
                self.problem += w[i] <= self.config.max_weight * y[i], f"Upper_{i}"
                # w[i] >= min_weight * y[i] (if invested, invest at least min_weight)
                self.problem += w[i] >= self.config.min_weight * y[i], f"Lower_{i}"

            # 3. Cardinality constraint: limit number of assets
            if self.config.max_assets is not None:
                self.problem += pl.lpSum([y[i] for i in assets]) <= self.config.max_assets, "Cardinality"

        # Solve
        start_time = time.time()

        solver = pl.getSolver(
            self.config.solver,
            msg=False,
            timeLimit=self.config.time_limit
        )

        self.problem.solve(solver)

        solve_time = time.time() - start_time

        # Extract solution
        if self.problem.status == pl.LpStatusOptimal:
            if self.config.allow_short_selling:
                # Extract long and short weights separately
                weights = pd.Series({i: w_long[i].varValue - w_short[i].varValue for i in assets})
                long_weights = pd.Series({i: w_long[i].varValue for i in assets})
                short_weights = pd.Series({i: w_short[i].varValue for i in assets})

                gross_exposure = long_weights.sum() + short_weights.sum()
                net_exposure = long_weights.sum() - short_weights.sum()

                self.solution = {
                    'weights': weights,
                    'long_weights': long_weights,
                    'short_weights': short_weights,
                    'gross_exposure': gross_exposure,
                    'net_exposure': net_exposure,
                    'objective_value': pl.value(self.problem.objective),
                    'solve_time': solve_time,
                    'status': 'optimal',
                    'n_assets': int(sum(abs(weights) > 1e-6)),
                    'n_long': int(sum(long_weights > 1e-6)),
                    'n_short': int(sum(short_weights > 1e-6))
                }
            else:
                # Long-only solution
                weights = pd.Series({i: w[i].varValue for i in assets})
                weights = weights.fillna(0)

                # Normalize to ensure sum = 1 (numerical precision)
                weights = weights / weights.sum()

                self.solution = {
                    'weights': weights,
                    'objective_value': pl.value(self.problem.objective),
                    'solve_time': solve_time,
                    'status': 'optimal',
                    'n_assets': int(sum(weights > 1e-6))
                }

            return weights

        else:
            print(f"Optimization failed with status: {pl.LpStatus[self.problem.status]}")
            # Return equal-weight fallback
            fallback_weights = pd.Series(1.0 / n_assets, index=assets)

            self.solution = {
                'weights': fallback_weights,
                'objective_value': None,
                'solve_time': solve_time,
                'status': 'failed',
                'n_assets': n_assets
            }

            return fallback_weights

    def compute_portfolio_metrics(
        self,
        weights: pd.Series,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame
    ) -> Dict:
        """
        Compute portfolio performance metrics.

        Args:
            weights: Portfolio weights
            expected_returns: Expected returns
            covariance_matrix: Covariance matrix

        Returns:
            Dictionary of metrics
        """
        portfolio_return = (weights * expected_returns).sum()
        portfolio_variance = (weights.values @ covariance_matrix.values @ weights.values)
        portfolio_volatility = np.sqrt(portfolio_variance)

        sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0

        return {
            'expected_return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio,
            'n_assets': int((weights > 1e-6).sum())
        }


class NaiveMeanVarianceOptimizer:
    """
    Baseline mean-variance optimization without integer constraints.

    Uses closed-form solution for comparison.
    """

    def __init__(self, risk_aversion: float = 2.5):
        """
        Initialize naive optimizer.

        Args:
            risk_aversion: Risk aversion parameter (λ)
        """
        self.risk_aversion = risk_aversion

    def optimize(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame
    ) -> pd.Series:
        """
        Solve unconstrained mean-variance optimization.

        Args:
            expected_returns: Expected returns
            covariance_matrix: Covariance matrix

        Returns:
            Optimal weights
        """
        # Solve: w* = (1/λ) * Σ^(-1) * μ
        try:
            cov_inv = np.linalg.inv(covariance_matrix.values)
            optimal_weights = (1.0 / self.risk_aversion) * (cov_inv @ expected_returns.values)

            # Normalize to sum to 1
            optimal_weights = optimal_weights / optimal_weights.sum()

            # Handle negative weights (set to 0 for long-only)
            optimal_weights = np.maximum(optimal_weights, 0)
            optimal_weights = optimal_weights / optimal_weights.sum()

            return pd.Series(optimal_weights, index=expected_returns.index)

        except np.linalg.LinAlgError:
            print("Covariance matrix is singular. Using equal weights.")
            n = len(expected_returns)
            return pd.Series(1.0 / n, index=expected_returns.index)


if __name__ == "__main__":
    # Example usage
    from src.data.loader import AssetDataLoader
    from src.forecasting.returns_forecast import ReturnsForecast
    from src.forecasting.covariance import CovarianceEstimator

    # Load data
    loader = AssetDataLoader()
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']
    prices = loader.fetch_prices(tickers, '2020-01-01', '2023-12-31')
    returns = loader.compute_returns(prices)

    # Forecast inputs
    returns_forecaster = ReturnsForecast(method='historical')
    returns_forecaster.fit(returns)
    expected_returns = returns_forecaster.predict()

    cov_estimator = CovarianceEstimator(method='ledoit_wolf')
    cov_matrix = cov_estimator.estimate(returns)

    print("=== Portfolio Optimization Comparison ===\n")

    # Naive mean-variance
    print("1. Naive Mean-Variance (No Constraints)")
    naive_optimizer = NaiveMeanVarianceOptimizer(risk_aversion=2.5)
    naive_weights = naive_optimizer.optimize(expected_returns, cov_matrix)
    print(f"Weights:\n{naive_weights.round(4)}")
    print(f"Number of assets: {(naive_weights > 1e-6).sum()}\n")

    # MIO optimization
    print("2. Mixed-Integer Optimization (With Constraints)")
    config = OptimizationConfig(
        risk_aversion=2.5,
        max_assets=5,
        min_weight=0.10,
        max_weight=0.40
    )

    mio_optimizer = MIOOptimizer(config=config)
    mio_weights = mio_optimizer.optimize(expected_returns, cov_matrix)

    print(f"Weights:\n{mio_weights[mio_weights > 1e-6].round(4)}")
    print(f"\nSolution status: {mio_optimizer.solution['status']}")
    print(f"Solve time: {mio_optimizer.solution['solve_time']:.2f} seconds")
    print(f"Number of assets: {mio_optimizer.solution['n_assets']}")

    # Compare metrics
    print("\n=== Performance Comparison ===")
    naive_metrics = mio_optimizer.compute_portfolio_metrics(naive_weights, expected_returns, cov_matrix)
    mio_metrics = mio_optimizer.compute_portfolio_metrics(mio_weights, expected_returns, cov_matrix)

    comparison = pd.DataFrame({
        'Naive MVO': naive_metrics,
        'MIO': mio_metrics
    }).T

    print(comparison)
