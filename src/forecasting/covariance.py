"""
Covariance Matrix Estimation Module

Implements various covariance estimation methods:
- Sample covariance (baseline)
- Ledoit-Wolf shrinkage
- Exponentially weighted covariance
- Factor models (simplified)
"""

import pandas as pd
import numpy as np
from typing import Optional
from sklearn.covariance import LedoitWolf, OAS


class CovarianceEstimator:
    """
    Estimates covariance matrices using various methods.
    """

    def __init__(self, method: str = 'sample'):
        """
        Initialize covariance estimator.

        Args:
            method: Estimation method ('sample', 'ledoit_wolf', 'oas', 'exponential', 'constant_correlation')
        """
        self.method = method
        self.covariance = None

    def estimate(
        self,
        returns: pd.DataFrame,
        **kwargs
    ) -> pd.DataFrame:
        """
        Estimate covariance matrix.

        Args:
            returns: DataFrame of asset returns
            **kwargs: Method-specific parameters

        Returns:
            Estimated covariance matrix (annualized)
        """
        returns_clean = returns.dropna()

        if self.method == 'sample':
            cov_matrix = self._sample_covariance(returns_clean)

        elif self.method == 'ledoit_wolf':
            cov_matrix = self._ledoit_wolf(returns_clean)

        elif self.method == 'oas':
            cov_matrix = self._oas_shrinkage(returns_clean)

        elif self.method == 'exponential':
            span = kwargs.get('span', 60)
            cov_matrix = self._exponential_covariance(returns_clean, span)

        elif self.method == 'constant_correlation':
            cov_matrix = self._constant_correlation(returns_clean)

        else:
            raise ValueError(f"Unknown covariance method: {self.method}")

        # Annualize (assuming daily returns)
        cov_matrix = cov_matrix * 252

        # Ensure positive definite
        cov_matrix = self._ensure_positive_definite(cov_matrix)

        self.covariance = cov_matrix
        return cov_matrix

    def _sample_covariance(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Standard sample covariance matrix."""
        return returns.cov()

    def _ledoit_wolf(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Ledoit-Wolf shrinkage estimator.

        Shrinks sample covariance towards constant correlation matrix.
        """
        lw = LedoitWolf()
        cov_matrix = lw.fit(returns.values).covariance_

        return pd.DataFrame(
            cov_matrix,
            index=returns.columns,
            columns=returns.columns
        )

    def _oas_shrinkage(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Oracle Approximating Shrinkage estimator.

        Alternative shrinkage method.
        """
        oas = OAS()
        cov_matrix = oas.fit(returns.values).covariance_

        return pd.DataFrame(
            cov_matrix,
            index=returns.columns,
            columns=returns.columns
        )

    def _exponential_covariance(
        self,
        returns: pd.DataFrame,
        span: int = 60
    ) -> pd.DataFrame:
        """
        Exponentially weighted covariance matrix.

        Gives more weight to recent observations.
        """
        return returns.ewm(span=span).cov().iloc[-len(returns.columns):]

    def _constant_correlation(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Constant correlation model.

        Assumes all pairs have the same correlation.
        """
        # Calculate sample variances
        variances = returns.var()

        # Calculate average correlation
        corr_matrix = returns.corr()
        # Extract upper triangle (excluding diagonal)
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        avg_corr = upper_tri.stack().mean()

        # Build constant correlation matrix
        n = len(returns.columns)
        const_corr = np.full((n, n), avg_corr)
        np.fill_diagonal(const_corr, 1.0)

        # Convert correlation to covariance
        std_devs = np.sqrt(variances.values)
        cov_matrix = const_corr * np.outer(std_devs, std_devs)

        return pd.DataFrame(
            cov_matrix,
            index=returns.columns,
            columns=returns.columns
        )

    def _ensure_positive_definite(
        self,
        cov_matrix: pd.DataFrame,
        epsilon: float = 1e-8
    ) -> pd.DataFrame:
        """
        Ensure covariance matrix is positive definite.

        Args:
            cov_matrix: Covariance matrix
            epsilon: Small value to add to diagonal if needed

        Returns:
            Positive definite covariance matrix
        """
        # Check if already positive definite
        eigenvalues = np.linalg.eigvalsh(cov_matrix.values)

        if np.all(eigenvalues > 0):
            return cov_matrix

        # Add small value to diagonal
        print(f"Warning: Covariance matrix not positive definite. Adding {epsilon} to diagonal.")

        cov_values = cov_matrix.values.copy()
        min_eigenvalue = np.min(eigenvalues)

        if min_eigenvalue < 0:
            cov_values += np.eye(len(cov_values)) * (abs(min_eigenvalue) + epsilon)

        return pd.DataFrame(
            cov_values,
            index=cov_matrix.index,
            columns=cov_matrix.columns
        )

    def compute_condition_number(self) -> float:
        """
        Compute condition number of covariance matrix.

        Lower is better (less ill-conditioned).
        """
        if self.covariance is None:
            raise RuntimeError("Must estimate covariance first")

        return np.linalg.cond(self.covariance.values)

    def compare_methods(
        self,
        returns: pd.DataFrame,
        methods: list = ['sample', 'ledoit_wolf', 'exponential']
    ) -> pd.DataFrame:
        """
        Compare different covariance estimation methods.

        Args:
            returns: Asset returns
            methods: List of methods to compare

        Returns:
            DataFrame with comparison metrics
        """
        results = []

        for method in methods:
            estimator = CovarianceEstimator(method=method)
            cov_matrix = estimator.estimate(returns)

            # Calculate metrics
            condition_number = estimator.compute_condition_number()
            eigenvalues = np.linalg.eigvalsh(cov_matrix.values)
            min_eigenvalue = np.min(eigenvalues)
            max_eigenvalue = np.max(eigenvalues)

            results.append({
                'method': method,
                'condition_number': condition_number,
                'min_eigenvalue': min_eigenvalue,
                'max_eigenvalue': max_eigenvalue,
                'mean_variance': np.diag(cov_matrix.values).mean(),
                'mean_covariance': cov_matrix.values[np.triu_indices_from(cov_matrix.values, k=1)].mean()
            })

        return pd.DataFrame(results)


if __name__ == "__main__":
    # Example usage
    from src.data.loader import AssetDataLoader

    # Load data
    loader = AssetDataLoader()
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'JPM', 'V', 'WMT']
    prices = loader.fetch_prices(tickers, '2020-01-01', '2023-12-31')
    returns = loader.compute_returns(prices)

    # Compare covariance estimation methods
    estimator = CovarianceEstimator()
    comparison = estimator.compare_methods(returns)

    print("=== Covariance Estimation Methods Comparison ===\n")
    print(comparison.to_string(index=False))

    # Detailed look at Ledoit-Wolf
    print("\n=== Ledoit-Wolf Shrinkage Covariance ===")
    lw_estimator = CovarianceEstimator(method='ledoit_wolf')
    lw_cov = lw_estimator.estimate(returns)

    print(f"Condition number: {lw_estimator.compute_condition_number():.2f}")
    print(f"\nCovariance matrix (annualized):")
    print(lw_cov)
