"""
Factor-based risk models for portfolio optimization.

Implements:
- Fama-French 5-Factor Model (Market, Size, Value, Profitability, Investment)
- Factor loading estimation
- Factor covariance modeling
- Integration with existing optimization framework
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pandas_datareader as pdr
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import warnings


@dataclass
class FactorModelResult:
    """Results from factor model estimation."""
    factor_loadings: pd.DataFrame  # Beta coefficients for each asset
    factor_returns: pd.DataFrame   # Historical factor returns
    residual_cov: np.ndarray       # Idiosyncratic risk covariance
    factor_cov: np.ndarray         # Factor covariance matrix
    r_squared: pd.Series           # Model fit for each asset


class FamaFrenchFactors:
    """
    Fama-French 5-Factor Model implementation.

    Factors:
    - Mkt-RF: Market excess return
    - SMB: Small Minus Big (size factor)
    - HML: High Minus Low (value factor)
    - RMW: Robust Minus Weak (profitability factor)
    - CMA: Conservative Minus Aggressive (investment factor)
    """

    def __init__(self, use_local_data: bool = False):
        """
        Initialize Fama-French factor model.

        Args:
            use_local_data: If True, use locally stored factor data
        """
        self.use_local_data = use_local_data
        self.factor_data: Optional[pd.DataFrame] = None

    def fetch_factor_data(
        self,
        start_date: str,
        end_date: str,
        frequency: str = 'daily'
    ) -> pd.DataFrame:
        """
        Fetch Fama-French factor data from Kenneth French's data library.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            frequency: 'daily', 'weekly', or 'monthly'

        Returns:
            DataFrame with factor returns
        """
        try:
            # Map frequency to dataset name
            dataset_map = {
                'daily': 'F-F_Research_Data_5_Factors_2x3_daily',
                'weekly': 'F-F_Research_Data_5_Factors_2x3_weekly',
                'monthly': 'F-F_Research_Data_5_Factors_2x3'
            }

            dataset = dataset_map.get(frequency, 'F-F_Research_Data_5_Factors_2x3_daily')

            # Fetch from pandas_datareader
            ff_factors = pdr.DataReader(
                dataset,
                'famafrench',
                start=start_date,
                end=end_date
            )[0]  # First dataframe contains the factors

            # Convert from percentage to decimal
            ff_factors = ff_factors / 100.0

            self.factor_data = ff_factors
            return ff_factors

        except Exception as e:
            warnings.warn(f"Could not fetch Fama-French data: {e}. Using synthetic data.")
            return self._generate_synthetic_factors(start_date, end_date, frequency)

    def _generate_synthetic_factors(
        self,
        start_date: str,
        end_date: str,
        frequency: str = 'daily'
    ) -> pd.DataFrame:
        """Generate synthetic factor returns for testing."""
        dates = pd.date_range(start=start_date, end=end_date, freq='B' if frequency == 'daily' else 'W')

        np.random.seed(42)
        n_periods = len(dates)

        # Realistic factor parameters (annualized)
        factor_params = {
            'Mkt-RF': {'mean': 0.08/252, 'std': 0.16/np.sqrt(252)},
            'SMB': {'mean': 0.02/252, 'std': 0.12/np.sqrt(252)},
            'HML': {'mean': 0.03/252, 'std': 0.13/np.sqrt(252)},
            'RMW': {'mean': 0.025/252, 'std': 0.11/np.sqrt(252)},
            'CMA': {'mean': 0.02/252, 'std': 0.10/np.sqrt(252)},
            'RF': {'mean': 0.02/252, 'std': 0.001/np.sqrt(252)}
        }

        factors = {}
        for factor, params in factor_params.items():
            factors[factor] = np.random.normal(params['mean'], params['std'], n_periods)

        df = pd.DataFrame(factors, index=dates)
        self.factor_data = df
        return df

    def estimate_factor_loadings(
        self,
        asset_returns: pd.DataFrame,
        factor_returns: Optional[pd.DataFrame] = None
    ) -> FactorModelResult:
        """
        Estimate factor loadings (betas) for assets using regression.

        Args:
            asset_returns: DataFrame of asset returns (assets as columns)
            factor_returns: DataFrame of factor returns. If None, uses self.factor_data

        Returns:
            FactorModelResult with loadings, covariances, and fit statistics
        """
        if factor_returns is None:
            if self.factor_data is None:
                raise ValueError("No factor data available. Call fetch_factor_data first.")
            factor_returns = self.factor_data

        # Align dates
        common_dates = asset_returns.index.intersection(factor_returns.index)
        asset_returns = asset_returns.loc[common_dates]
        factor_returns = factor_returns.loc[common_dates]

        # Prepare factor matrix (exclude RF for regression)
        factor_cols = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
        X = factor_returns[factor_cols].values

        # Store results
        betas = {}
        r_squared_vals = {}
        residuals = {}

        # Run regression for each asset
        for asset in asset_returns.columns:
            y = asset_returns[asset].values

            # Linear regression
            model = LinearRegression(fit_intercept=True)
            model.fit(X, y)

            # Store betas
            betas[asset] = model.coef_

            # Calculate R-squared
            y_pred = model.predict(X)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared_vals[asset] = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            # Store residuals
            residuals[asset] = y - y_pred

        # Create DataFrames
        factor_loadings = pd.DataFrame(betas, index=factor_cols).T
        residual_df = pd.DataFrame(residuals, index=common_dates)

        # Calculate covariance matrices
        factor_cov = np.cov(X.T)
        residual_cov = np.cov(residual_df.T)

        # Create result object
        result = FactorModelResult(
            factor_loadings=factor_loadings,
            factor_returns=factor_returns[factor_cols],
            residual_cov=residual_cov,
            factor_cov=factor_cov,
            r_squared=pd.Series(r_squared_vals)
        )

        return result

    def compute_total_covariance(
        self,
        factor_result: FactorModelResult
    ) -> np.ndarray:
        """
        Compute total covariance matrix using factor model decomposition.

        Cov(R) = B * Cov(F) * B' + Cov(ε)

        where:
        - B: factor loadings matrix
        - Cov(F): factor covariance
        - Cov(ε): residual covariance

        Args:
            factor_result: Result from estimate_factor_loadings

        Returns:
            Total covariance matrix
        """
        B = factor_result.factor_loadings.values
        F_cov = factor_result.factor_cov
        epsilon_cov = factor_result.residual_cov

        # Factor contribution: B * Cov(F) * B'
        factor_contribution = B @ F_cov @ B.T

        # Total covariance
        total_cov = factor_contribution + epsilon_cov

        return total_cov

    def forecast_returns(
        self,
        factor_result: FactorModelResult,
        factor_forecasts: Optional[pd.Series] = None,
        use_historical_mean: bool = True
    ) -> pd.Series:
        """
        Forecast asset returns using factor model.

        E[R_i] = α_i + Σ(β_ij * E[F_j])

        Args:
            factor_result: Result from estimate_factor_loadings
            factor_forecasts: Forecasted factor returns. If None, uses historical means
            use_historical_mean: Use historical factor means for forecasting

        Returns:
            Series of forecasted returns for each asset
        """
        if factor_forecasts is None:
            if use_historical_mean:
                factor_forecasts = factor_result.factor_returns.mean()
            else:
                # Use zero factor forecasts (market neutral)
                factor_forecasts = pd.Series(0, index=factor_result.factor_loadings.columns)

        # Calculate expected returns
        expected_returns = factor_result.factor_loadings @ factor_forecasts

        return expected_returns


class BarraFactorModel:
    """
    Simplified Barra-style factor model.

    Uses industry and style factors for risk decomposition.
    """

    def __init__(self):
        """Initialize Barra-style factor model."""
        self.industry_factors: Optional[pd.DataFrame] = None
        self.style_factors: Optional[pd.DataFrame] = None

    def create_industry_factors(
        self,
        tickers: List[str],
        industry_mapping: Dict[str, str]
    ) -> pd.DataFrame:
        """
        Create industry factor exposures (dummy variables).

        Args:
            tickers: List of asset tickers
            industry_mapping: Dict mapping ticker to industry

        Returns:
            DataFrame with industry exposures
        """
        industries = list(set(industry_mapping.values()))
        exposures = pd.DataFrame(0, index=tickers, columns=industries)

        for ticker in tickers:
            if ticker in industry_mapping:
                exposures.loc[ticker, industry_mapping[ticker]] = 1

        self.industry_factors = exposures
        return exposures

    def create_style_factors(
        self,
        characteristics: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create style factor exposures (e.g., value, momentum, quality).

        Args:
            characteristics: DataFrame with asset characteristics
                            (e.g., P/E ratio, momentum, ROE)

        Returns:
            DataFrame with standardized style exposures
        """
        # Standardize characteristics (z-scores)
        style_factors = (characteristics - characteristics.mean()) / characteristics.std()

        self.style_factors = style_factors
        return style_factors

    def estimate_factor_model(
        self,
        returns: pd.DataFrame,
        industry_exposures: pd.DataFrame,
        style_exposures: pd.DataFrame
    ) -> FactorModelResult:
        """
        Estimate Barra-style factor model.

        Args:
            returns: Asset returns
            industry_exposures: Industry factor exposures
            style_exposures: Style factor exposures

        Returns:
            FactorModelResult with estimated parameters
        """
        # Combine all factors
        all_factors = pd.concat([industry_exposures, style_exposures], axis=1)

        # Align dates
        common_dates = returns.index

        # Estimate using cross-sectional regression
        factor_returns_list = []

        for date in common_dates:
            if date in returns.index:
                # Cross-sectional regression: r_t = B * f_t + ε_t
                y = returns.loc[date].values
                X = all_factors.values

                model = LinearRegression(fit_intercept=False)
                model.fit(X.T, y)

                factor_returns_list.append(model.coef_)

        factor_returns_df = pd.DataFrame(
            factor_returns_list,
            index=common_dates,
            columns=all_factors.columns
        )

        # Calculate covariances
        factor_cov = np.cov(factor_returns_df.T)

        # Calculate residuals and residual covariance
        predicted = all_factors.T @ factor_returns_df.T
        residuals = returns.T - predicted
        residual_cov = np.cov(residuals)

        # Create result
        result = FactorModelResult(
            factor_loadings=all_factors,
            factor_returns=factor_returns_df,
            residual_cov=residual_cov,
            factor_cov=factor_cov,
            r_squared=pd.Series(1.0, index=returns.columns)  # Placeholder
        )

        return result


if __name__ == "__main__":
    # Example usage
    print("Testing Fama-French 5-Factor Model...")

    # Initialize model
    ff_model = FamaFrenchFactors()

    # Fetch factor data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=252*2)  # 2 years

    factors = ff_model.fetch_factor_data(
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d')
    )

    print(f"\nFactor data shape: {factors.shape}")
    print(f"\nFactor summary statistics:")
    print(factors.describe())

    # Generate synthetic asset returns for testing
    n_assets = 5
    asset_returns = pd.DataFrame(
        np.random.randn(len(factors), n_assets) * 0.01,
        index=factors.index,
        columns=[f'Asset_{i}' for i in range(n_assets)]
    )

    # Estimate factor loadings
    result = ff_model.estimate_factor_loadings(asset_returns)

    print(f"\nFactor loadings:")
    print(result.factor_loadings)

    print(f"\nR-squared values:")
    print(result.r_squared)

    # Compute total covariance
    total_cov = ff_model.compute_total_covariance(result)
    print(f"\nTotal covariance matrix shape: {total_cov.shape}")

    # Forecast returns
    forecasted_returns = ff_model.forecast_returns(result)
    print(f"\nForecasted returns:")
    print(forecasted_returns)

    print("\n✅ Fama-French factor model implementation complete!")
