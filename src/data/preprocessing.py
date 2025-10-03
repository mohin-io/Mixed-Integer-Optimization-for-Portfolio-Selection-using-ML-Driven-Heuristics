"""
Data Preprocessing Module

Responsibilities:
- Compute rolling windows for out-of-sample testing
- Calculate risk factors (size, value, momentum)
- Generate covariance matrices
- Handle outliers and winsorization
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from scipy import stats
from sklearn.preprocessing import StandardScaler


class DataPreprocessor:
    """
    Preprocesses financial data for portfolio optimization.

    Handles rolling windows, factor computation, and outlier treatment.
    """

    def __init__(self, winsorize_quantile: float = 0.01):
        """
        Initialize preprocessor.

        Args:
            winsorize_quantile: Quantile for winsorization (default: 1%)
        """
        self.winsorize_quantile = winsorize_quantile
        self.scaler = StandardScaler()

    def compute_rolling_windows(
        self,
        prices: pd.DataFrame,
        window_size: int = 252,
        step_size: int = 21
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Create rolling train/test windows for backtesting.

        Args:
            prices: DataFrame of asset prices
            window_size: Training window size in days (default: 252 = 1 year)
            step_size: Step size for rolling window (default: 21 = 1 month)

        Returns:
            List of (train_window, test_window) tuples
        """
        windows = []
        n_samples = len(prices)

        for start_idx in range(0, n_samples - window_size - step_size, step_size):
            train_end_idx = start_idx + window_size
            test_end_idx = min(train_end_idx + step_size, n_samples)

            train_window = prices.iloc[start_idx:train_end_idx]
            test_window = prices.iloc[train_end_idx:test_end_idx]

            if len(test_window) > 0:
                windows.append((train_window, test_window))

        print(f"Created {len(windows)} rolling windows")
        return windows

    def calculate_momentum_factor(
        self,
        returns: pd.DataFrame,
        lookback: int = 126
    ) -> pd.Series:
        """
        Calculate momentum factor scores for each asset.

        Momentum = cumulative return over lookback period.

        Args:
            returns: DataFrame of asset returns
            lookback: Lookback period in days (default: 126 = 6 months)

        Returns:
            Series of momentum scores
        """
        # Cumulative return over lookback period
        momentum = (1 + returns.tail(lookback)).prod() - 1
        return momentum

    def calculate_volatility_factor(
        self,
        returns: pd.DataFrame,
        lookback: int = 126
    ) -> pd.Series:
        """
        Calculate volatility (risk) factor for each asset.

        Args:
            returns: DataFrame of asset returns
            lookback: Lookback period in days

        Returns:
            Series of annualized volatilities
        """
        volatility = returns.tail(lookback).std() * np.sqrt(252)
        return volatility

    def calculate_factors(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        market_caps: Optional[dict] = None
    ) -> pd.DataFrame:
        """
        Calculate comprehensive factor exposures for all assets.

        Factors:
        - Momentum (6-month return)
        - Volatility (annualized std dev)
        - Size (market capitalization)
        - Beta (market sensitivity)

        Args:
            prices: DataFrame of asset prices
            returns: DataFrame of asset returns
            market_caps: Optional dictionary of market capitalizations

        Returns:
            DataFrame with factor scores for each asset
        """
        factors = pd.DataFrame(index=prices.columns)

        # Momentum factor
        factors['momentum'] = self.calculate_momentum_factor(returns)

        # Volatility factor
        factors['volatility'] = self.calculate_volatility_factor(returns)

        # Size factor (if market cap data available)
        if market_caps is not None:
            factors['size'] = pd.Series(market_caps)
            factors['size'] = np.log(factors['size'].fillna(factors['size'].median()))

        # Beta factor (sensitivity to equal-weight market)
        market_returns = returns.mean(axis=1)
        betas = []
        for col in returns.columns:
            asset_returns = returns[col].dropna()
            aligned_market = market_returns.loc[asset_returns.index]

            if len(asset_returns) > 30:  # Need sufficient data
                covariance = np.cov(asset_returns, aligned_market)[0, 1]
                market_variance = np.var(aligned_market)
                beta = covariance / market_variance if market_variance > 0 else 1.0
            else:
                beta = 1.0

            betas.append(beta)

        factors['beta'] = betas

        # Standardize factors (mean=0, std=1)
        factors_standardized = pd.DataFrame(
            self.scaler.fit_transform(factors.fillna(0)),
            index=factors.index,
            columns=factors.columns
        )

        return factors_standardized

    def winsorize_returns(
        self,
        returns: pd.DataFrame,
        quantile: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Winsorize returns to limit impact of outliers.

        Caps extreme values at specified quantiles.

        Args:
            returns: DataFrame of asset returns
            quantile: Winsorization quantile (if None, use instance default)

        Returns:
            Winsorized returns DataFrame
        """
        quantile = quantile or self.winsorize_quantile

        lower_bound = returns.quantile(quantile)
        upper_bound = returns.quantile(1 - quantile)

        winsorized = returns.clip(lower=lower_bound, upper=upper_bound, axis=1)

        n_clipped = (returns != winsorized).sum().sum()
        print(f"Winsorized {n_clipped} extreme return observations")

        return winsorized

    def compute_covariance_matrix(
        self,
        returns: pd.DataFrame,
        method: str = 'sample'
    ) -> pd.DataFrame:
        """
        Compute covariance matrix of returns.

        Args:
            returns: DataFrame of asset returns
            method: Estimation method ('sample', 'exponential', 'shrinkage')

        Returns:
            Covariance matrix DataFrame
        """
        if method == 'sample':
            cov_matrix = returns.cov()

        elif method == 'exponential':
            # Exponentially weighted covariance (more weight to recent data)
            cov_matrix = returns.ewm(span=60).cov().iloc[-len(returns.columns):]

        elif method == 'shrinkage':
            # Ledoit-Wolf shrinkage (will be implemented in forecasting module)
            cov_matrix = returns.cov()
            print("Warning: Using sample covariance. Use forecasting.covariance module for shrinkage.")

        else:
            raise ValueError(f"Unknown covariance method: {method}")

        # Annualize covariance matrix (assuming daily returns)
        cov_matrix = cov_matrix * 252

        return cov_matrix

    def detect_outliers(
        self,
        returns: pd.DataFrame,
        method: str = 'zscore',
        threshold: float = 3.0
    ) -> pd.DataFrame:
        """
        Detect outlier returns.

        Args:
            returns: DataFrame of asset returns
            method: Detection method ('zscore', 'iqr')
            threshold: Threshold for outlier detection

        Returns:
            Boolean DataFrame indicating outliers
        """
        if method == 'zscore':
            z_scores = np.abs(stats.zscore(returns, nan_policy='omit'))
            outliers = z_scores > threshold

        elif method == 'iqr':
            Q1 = returns.quantile(0.25)
            Q3 = returns.quantile(0.75)
            IQR = Q3 - Q1
            outliers = (returns < (Q1 - threshold * IQR)) | (returns > (Q3 + threshold * IQR))

        else:
            raise ValueError(f"Unknown outlier detection method: {method}")

        n_outliers = outliers.sum().sum()
        print(f"Detected {n_outliers} outlier observations ({method} method)")

        return outliers

    def create_lagged_features(
        self,
        returns: pd.DataFrame,
        lags: List[int] = [1, 5, 21]
    ) -> pd.DataFrame:
        """
        Create lagged return features for ML models.

        Args:
            returns: DataFrame of asset returns
            lags: List of lag periods

        Returns:
            DataFrame with lagged features
        """
        features = pd.DataFrame(index=returns.index)

        for lag in lags:
            lagged = returns.shift(lag)
            for col in returns.columns:
                features[f'{col}_lag_{lag}'] = lagged[col]

        return features.dropna()

    def compute_correlation_matrix(
        self,
        returns: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute correlation matrix of returns.

        Args:
            returns: DataFrame of asset returns

        Returns:
            Correlation matrix DataFrame
        """
        correlation = returns.corr()
        return correlation

    def filter_assets_by_liquidity(
        self,
        prices: pd.DataFrame,
        min_trading_days: int = 200
    ) -> pd.DataFrame:
        """
        Filter assets by minimum trading history.

        Args:
            prices: DataFrame of asset prices
            min_trading_days: Minimum number of valid price observations

        Returns:
            Filtered prices DataFrame
        """
        valid_days = prices.notna().sum()
        liquid_assets = valid_days[valid_days >= min_trading_days].index

        print(f"Filtered to {len(liquid_assets)}/{len(prices.columns)} liquid assets")

        return prices[liquid_assets]


if __name__ == "__main__":
    # Example usage
    from src.data.loader import AssetDataLoader

    # Load data
    loader = AssetDataLoader()
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'JPM', 'V', 'WMT']
    prices = loader.fetch_prices(tickers, '2020-01-01', '2023-12-31')
    returns = loader.compute_returns(prices)

    # Preprocess data
    preprocessor = DataPreprocessor()

    # Factor calculation
    print("\n=== Factor Analysis ===")
    factors = preprocessor.calculate_factors(prices, returns)
    print(factors)

    # Winsorization
    print("\n=== Winsorization ===")
    winsorized_returns = preprocessor.winsorize_returns(returns)

    # Covariance matrix
    print("\n=== Covariance Matrix ===")
    cov_matrix = preprocessor.compute_covariance_matrix(returns)
    print(f"Covariance matrix shape: {cov_matrix.shape}")
    print(f"Condition number: {np.linalg.cond(cov_matrix):.2f}")

    # Rolling windows
    print("\n=== Rolling Windows ===")
    windows = preprocessor.compute_rolling_windows(prices, window_size=252, step_size=21)
    print(f"Created {len(windows)} windows for backtesting")
