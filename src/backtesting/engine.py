"""
Backtesting Engine for Portfolio Optimization Strategies

Implements rolling window backtesting with:
- Multiple rebalancing frequencies
- Transaction cost accounting
- Slippage simulation
- Performance metric calculation
- Benchmark comparison
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings

from ..data.loader import AssetDataLoader
from ..forecasting.returns_forecast import ReturnsForecast
from ..forecasting.covariance import CovarianceEstimator


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    train_window: int = 252  # Trading days for training
    rebalance_freq: int = 21  # Rebalancing frequency (days)
    transaction_cost: float = 0.001  # Transaction cost (0.1%)
    slippage: float = 0.0005  # Slippage (0.05%)
    initial_capital: float = 100000.0  # Starting capital
    min_history: int = 60  # Minimum history required
    risk_free_rate: float = 0.02  # Risk-free rate (annual)
    verbose: bool = True


@dataclass
class BacktestResults:
    """Results from backtesting."""
    strategy_name: str
    equity_curve: pd.Series
    weights_history: pd.DataFrame
    returns: pd.Series
    trades: pd.DataFrame
    metrics: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        """Calculate performance metrics after initialization."""
        if not self.metrics:
            self.metrics = self._calculate_metrics()

    def _calculate_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        returns = self.returns.dropna()

        if len(returns) == 0:
            return {}

        # Basic metrics
        total_return = (self.equity_curve.iloc[-1] / self.equity_curve.iloc[0]) - 1
        n_years = len(returns) / 252
        annual_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

        # Risk metrics
        annual_volatility = returns.std() * np.sqrt(252)
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0

        # Sharpe and Sortino ratios
        sharpe_ratio = (annual_return - 0.02) / annual_volatility if annual_volatility > 0 else 0
        sortino_ratio = (annual_return - 0.02) / downside_volatility if downside_volatility > 0 else 0

        # Drawdown analysis
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Calculate average drawdown duration
        is_drawdown = drawdown < -0.01  # 1% threshold
        drawdown_periods = []
        current_period = 0

        for is_dd in is_drawdown:
            if is_dd:
                current_period += 1
            elif current_period > 0:
                drawdown_periods.append(current_period)
                current_period = 0

        avg_drawdown_duration = np.mean(drawdown_periods) if drawdown_periods else 0

        # Trading metrics
        n_trades = len(self.trades) if len(self.trades) > 0 else 0
        total_transaction_cost = self.trades['cost'].sum() if 'cost' in self.trades.columns else 0
        turnover = self.trades['turnover'].mean() if 'turnover' in self.trades.columns else 0

        # Value at Risk (VaR)
        var_95 = returns.quantile(0.05)
        cvar_95 = returns[returns <= var_95].mean()

        # Win rate
        win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0

        # Calmar ratio (return / max drawdown)
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'avg_drawdown_duration': avg_drawdown_duration,
            'calmar_ratio': calmar_ratio,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'win_rate': win_rate,
            'n_trades': n_trades,
            'total_transaction_cost': total_transaction_cost,
            'avg_turnover': turnover,
            'n_periods': len(returns),
            'n_years': n_years
        }

    def summary(self) -> pd.DataFrame:
        """Return metrics as a formatted DataFrame."""
        metrics_formatted = {}

        for key, value in self.metrics.items():
            if 'return' in key or 'volatility' in key or 'turnover' in key:
                metrics_formatted[key] = f"{value*100:.2f}%"
            elif 'ratio' in key:
                metrics_formatted[key] = f"{value:.3f}"
            elif 'drawdown' in key and 'duration' not in key:
                metrics_formatted[key] = f"{value*100:.2f}%"
            elif 'var' in key or 'cvar' in key:
                metrics_formatted[key] = f"{value*100:.2f}%"
            elif 'win_rate' in key:
                metrics_formatted[key] = f"{value*100:.2f}%"
            elif 'cost' in key:
                metrics_formatted[key] = f"${value:,.2f}"
            else:
                metrics_formatted[key] = f"{value:.2f}"

        return pd.DataFrame.from_dict(
            metrics_formatted,
            orient='index',
            columns=[self.strategy_name]
        )


class Backtester:
    """
    Rolling window backtesting engine.

    For each rebalancing period:
    1. Train forecasting models on historical data
    2. Forecast returns and covariance
    3. Optimize portfolio weights
    4. Execute trades (simulate slippage)
    5. Record performance
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        """
        Initialize backtester.

        Args:
            config: Backtesting configuration
        """
        self.config = config or BacktestConfig()
        self.results: Optional[BacktestResults] = None

    def run_backtest(
        self,
        prices: pd.DataFrame,
        optimizer_func: Callable,
        strategy_name: str = "Strategy",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> BacktestResults:
        """
        Run backtesting with rolling window.

        Args:
            prices: DataFrame of asset prices (index: dates, columns: tickers)
            optimizer_func: Function that takes (returns, cov_matrix) and returns weights
            strategy_name: Name of the strategy
            start_date: Start date for backtest (default: first valid date)
            end_date: End date for backtest (default: last date)

        Returns:
            BacktestResults object
        """
        # Validate inputs
        if prices.empty:
            raise ValueError("Prices DataFrame is empty")

        prices = prices.sort_index()

        # Set date range
        if start_date:
            prices = prices[prices.index >= start_date]
        if end_date:
            prices = prices[prices.index <= end_date]

        # Calculate returns
        returns = prices.pct_change().dropna()

        # Initialize tracking variables
        equity_curve = []
        weights_history = []
        trades_history = []
        returns_history = []

        current_weights = pd.Series(0.0, index=prices.columns)
        capital = self.config.initial_capital

        # Get rebalancing dates
        rebalance_dates = self._get_rebalance_dates(returns.index)

        if self.config.verbose:
            print(f"\n{'='*60}")
            print(f"Backtesting: {strategy_name}")
            print(f"{'='*60}")
            print(f"Period: {returns.index[0].date()} to {returns.index[-1].date()}")
            print(f"Assets: {len(prices.columns)}")
            print(f"Rebalancing frequency: {self.config.rebalance_freq} days")
            print(f"Number of rebalances: {len(rebalance_dates)}")
            print(f"{'='*60}\n")

        # Main backtesting loop
        for i, rebal_date in enumerate(rebalance_dates):
            try:
                # Get training data
                train_data = self._get_training_data(returns, rebal_date)

                if len(train_data) < self.config.min_history:
                    if self.config.verbose:
                        print(f"Skipping {rebal_date.date()}: Insufficient history")
                    continue

                # Forecast returns and covariance
                expected_returns, cov_matrix = self._forecast_parameters(train_data)

                # Optimize portfolio
                new_weights = optimizer_func(expected_returns, cov_matrix)

                # Ensure weights is a Series
                if not isinstance(new_weights, pd.Series):
                    new_weights = pd.Series(new_weights, index=prices.columns)

                # Normalize weights
                if new_weights.sum() > 0:
                    new_weights = new_weights / new_weights.sum()
                else:
                    new_weights = pd.Series(0.0, index=prices.columns)

                # Calculate transaction costs
                turnover = (new_weights - current_weights).abs().sum()
                transaction_cost = turnover * self.config.transaction_cost
                slippage_cost = turnover * self.config.slippage
                total_cost = transaction_cost + slippage_cost

                # Apply costs to capital
                capital *= (1 - total_cost)

                # Record trade
                trades_history.append({
                    'date': rebal_date,
                    'turnover': turnover,
                    'transaction_cost': transaction_cost,
                    'slippage': slippage_cost,
                    'cost': total_cost * capital
                })

                # Update weights
                current_weights = new_weights.copy()
                weights_history.append({
                    'date': rebal_date,
                    **current_weights.to_dict()
                })

                if self.config.verbose and (i + 1) % 10 == 0:
                    print(f"Rebalance {i+1}/{len(rebalance_dates)}: "
                          f"{rebal_date.date()}, Capital: ${capital:,.2f}, "
                          f"Turnover: {turnover*100:.1f}%")

            except Exception as e:
                if self.config.verbose:
                    print(f"Error at {rebal_date.date()}: {str(e)}")
                continue

        # Calculate equity curve for all dates
        for date in returns.index:
            if date >= rebalance_dates[0]:
                # Get returns for current holdings
                daily_return = (returns.loc[date] * current_weights).sum()

                # Update capital
                capital *= (1 + daily_return)

                equity_curve.append({
                    'date': date,
                    'equity': capital
                })

                returns_history.append({
                    'date': date,
                    'return': daily_return
                })

                # Update weights for next rebalance
                if date in rebalance_dates:
                    idx = list(rebalance_dates).index(date)
                    if idx < len(weights_history):
                        weight_dict = weights_history[idx]
                        current_weights = pd.Series({
                            k: v for k, v in weight_dict.items()
                            if k != 'date'
                        })

        # Convert to DataFrames
        equity_df = pd.DataFrame(equity_curve).set_index('date')
        weights_df = pd.DataFrame(weights_history).set_index('date')
        trades_df = pd.DataFrame(trades_history).set_index('date')
        returns_df = pd.DataFrame(returns_history).set_index('date')

        # Create results
        self.results = BacktestResults(
            strategy_name=strategy_name,
            equity_curve=equity_df['equity'],
            weights_history=weights_df,
            returns=returns_df['return'],
            trades=trades_df
        )

        if self.config.verbose:
            print(f"\n{'='*60}")
            print("Backtest Complete!")
            print(f"{'='*60}")
            print(self.results.summary())
            print(f"{'='*60}\n")

        return self.results

    def _get_rebalance_dates(self, dates: pd.DatetimeIndex) -> List[pd.Timestamp]:
        """Get list of rebalancing dates."""
        rebalance_dates = []

        # Start after minimum training window
        start_idx = self.config.train_window

        for i in range(start_idx, len(dates), self.config.rebalance_freq):
            rebalance_dates.append(dates[i])

        return rebalance_dates

    def _get_training_data(
        self,
        returns: pd.DataFrame,
        current_date: pd.Timestamp
    ) -> pd.DataFrame:
        """Get training data up to current date."""
        # Get all data before current date
        historical_data = returns[returns.index < current_date]

        # Use last train_window days
        if len(historical_data) > self.config.train_window:
            return historical_data.iloc[-self.config.train_window:]

        return historical_data

    def _forecast_parameters(
        self,
        train_data: pd.DataFrame
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Forecast expected returns and covariance matrix.

        Args:
            train_data: Historical returns data

        Returns:
            Tuple of (expected_returns, covariance_matrix)
        """
        # Simple historical mean for returns
        expected_returns = train_data.mean() * 252  # Annualize

        # Use Ledoit-Wolf shrinkage for covariance
        cov_estimator = CovarianceEstimator(method='ledoit_wolf')
        cov_matrix = cov_estimator.estimate(train_data)

        # Annualize covariance
        cov_matrix = cov_matrix * 252

        return expected_returns, cov_matrix

    def compare_strategies(
        self,
        prices: pd.DataFrame,
        strategies: Dict[str, Callable],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Compare multiple strategies.

        Args:
            prices: DataFrame of asset prices
            strategies: Dict mapping strategy names to optimizer functions
            start_date: Start date for backtest
            end_date: End date for backtest

        Returns:
            DataFrame with comparison metrics
        """
        all_results = {}

        for strategy_name, optimizer_func in strategies.items():
            if self.config.verbose:
                print(f"\nRunning backtest for: {strategy_name}")

            results = self.run_backtest(
                prices=prices,
                optimizer_func=optimizer_func,
                strategy_name=strategy_name,
                start_date=start_date,
                end_date=end_date
            )

            all_results[strategy_name] = results.metrics

        # Create comparison DataFrame
        comparison_df = pd.DataFrame(all_results).T

        if self.config.verbose:
            print(f"\n{'='*60}")
            print("STRATEGY COMPARISON")
            print(f"{'='*60}")
            print(comparison_df.to_string())
            print(f"{'='*60}\n")

        return comparison_df


def equal_weight_optimizer(expected_returns: pd.Series, cov_matrix: pd.DataFrame) -> pd.Series:
    """Equal weight portfolio (1/N)."""
    n = len(expected_returns)
    return pd.Series(1.0 / n, index=expected_returns.index)


def max_sharpe_optimizer(expected_returns: pd.Series, cov_matrix: pd.DataFrame) -> pd.Series:
    """Maximum Sharpe ratio portfolio."""
    from scipy.optimize import minimize

    n = len(expected_returns)

    def neg_sharpe(weights):
        port_return = np.dot(weights, expected_returns.values)
        port_vol = np.sqrt(weights @ cov_matrix.values @ weights)
        return -port_return / port_vol if port_vol > 0 else 0

    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = tuple((0, 1) for _ in range(n))

    result = minimize(
        neg_sharpe,
        x0=np.ones(n) / n,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    return pd.Series(result.x, index=expected_returns.index)


def min_variance_optimizer(expected_returns: pd.Series, cov_matrix: pd.DataFrame) -> pd.Series:
    """Minimum variance portfolio."""
    from scipy.optimize import minimize

    n = len(expected_returns)

    def portfolio_variance(weights):
        return weights @ cov_matrix.values @ weights

    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = tuple((0, 1) for _ in range(n))

    result = minimize(
        portfolio_variance,
        x0=np.ones(n) / n,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    return pd.Series(result.x, index=expected_returns.index)


if __name__ == "__main__":
    # Example usage
    print("Backtesting Engine Demo\n")

    # Load sample data
    loader = AssetDataLoader()
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'JPM', 'V', 'WMT', 'JNJ', 'PG']

    try:
        prices = loader.fetch_prices(tickers, '2020-01-01', '2023-12-31')

        # Setup backtesting config
        config = BacktestConfig(
            train_window=252,
            rebalance_freq=21,  # Monthly rebalancing
            transaction_cost=0.001,
            initial_capital=100000,
            verbose=True
        )

        backtester = Backtester(config=config)

        # Define strategies
        strategies = {
            'Equal Weight': equal_weight_optimizer,
            'Max Sharpe': max_sharpe_optimizer,
            'Min Variance': min_variance_optimizer
        }

        # Run comparison
        comparison = backtester.compare_strategies(
            prices=prices,
            strategies=strategies,
            start_date='2021-01-01'
        )

        print("\nBacktesting completed successfully!")

    except Exception as e:
        print(f"Error in demo: {str(e)}")
