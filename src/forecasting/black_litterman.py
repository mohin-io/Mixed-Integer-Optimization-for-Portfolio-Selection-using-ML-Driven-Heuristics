"""
Black-Litterman Model for Portfolio Optimization.

Combines market equilibrium returns with investor views to generate
superior return forecasts.

Reference:
- Black, F., & Litterman, R. (1992). Global Portfolio Optimization.
  Financial Analysts Journal.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from scipy.linalg import inv


@dataclass
class InvestorView:
    """Represents a single investor view (forecast)."""
    assets: List[str]  # Assets involved in the view
    weights: List[float]  # Weights for each asset (sum to 0 for relative views)
    expected_return: float  # Expected return for this view
    confidence: float = 0.5  # Confidence level (0 to 1)


class BlackLittermanModel:
    """
    Black-Litterman model for combining market equilibrium with investor views.

    The model blends:
    1. Market-implied equilibrium returns (from reverse optimization)
    2. Investor views (subjective forecasts)

    Output: Posterior expected returns that can be used in portfolio optimization
    """

    def __init__(
        self,
        risk_aversion: float = 2.5,
        tau: float = 0.05,
        market_weight: Optional[pd.Series] = None
    ):
        """
        Initialize Black-Litterman model.

        Args:
            risk_aversion: Market risk aversion parameter (δ)
            tau: Uncertainty scaling parameter (typically 0.01 to 0.05)
            market_weight: Market capitalization weights (if None, use equal weights)
        """
        self.risk_aversion = risk_aversion
        self.tau = tau
        self.market_weight = market_weight

    def compute_equilibrium_returns(
        self,
        covariance: pd.DataFrame,
        market_weights: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Compute market-implied equilibrium returns using reverse optimization.

        Π = δ * Σ * w_mkt

        where:
        - Π: implied equilibrium returns
        - δ: risk aversion
        - Σ: covariance matrix
        - w_mkt: market capitalization weights

        Args:
            covariance: Covariance matrix of asset returns
            market_weights: Market cap weights (if None, uses equal weights)

        Returns:
            Series of equilibrium returns
        """
        if market_weights is None:
            if self.market_weight is not None:
                market_weights = self.market_weight
            else:
                # Use equal weights as default
                n_assets = len(covariance)
                market_weights = pd.Series(1.0 / n_assets, index=covariance.index)

        # Ensure alignment
        market_weights = market_weights.reindex(covariance.index, fill_value=0)

        # Compute equilibrium returns: Π = δ * Σ * w
        equilibrium_returns = self.risk_aversion * (covariance @ market_weights)

        return equilibrium_returns

    def create_view_matrix(
        self,
        views: List[InvestorView],
        asset_names: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create view matrices P, Q, and Ω for Black-Litterman model.

        P: Pick matrix (k x n) - specifies which assets are in each view
        Q: View returns vector (k x 1) - expected returns for each view
        Ω: Uncertainty matrix (k x k) - diagonal matrix of view uncertainties

        Args:
            views: List of InvestorView objects
            asset_names: List of all asset names

        Returns:
            Tuple of (P, Q, Omega) matrices
        """
        k = len(views)  # Number of views
        n = len(asset_names)  # Number of assets

        P = np.zeros((k, n))
        Q = np.zeros(k)
        omega = np.zeros(k)

        for i, view in enumerate(views):
            # Build pick matrix row
            for asset, weight in zip(view.assets, view.weights):
                if asset in asset_names:
                    idx = asset_names.index(asset)
                    P[i, idx] = weight

            # Expected return for this view
            Q[i] = view.expected_return

            # View uncertainty (inversely proportional to confidence)
            # Higher confidence -> lower uncertainty
            omega[i] = (1.0 - view.confidence) / view.confidence if view.confidence > 0 else 1.0

        # Omega is diagonal matrix
        Omega = np.diag(omega)

        return P, Q, Omega

    def compute_posterior_returns(
        self,
        equilibrium_returns: pd.Series,
        covariance: pd.DataFrame,
        views: List[InvestorView]
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Compute posterior (blended) returns using Black-Litterman formula.

        E[R] = [(τΣ)^(-1) + P'Ω^(-1)P]^(-1) * [(τΣ)^(-1)Π + P'Ω^(-1)Q]

        Args:
            equilibrium_returns: Market equilibrium returns (Π)
            covariance: Covariance matrix (Σ)
            views: List of investor views

        Returns:
            Tuple of (posterior_returns, posterior_covariance)
        """
        asset_names = equilibrium_returns.index.tolist()
        n = len(asset_names)

        # Convert to numpy arrays
        Pi = equilibrium_returns.values.reshape(-1, 1)
        Sigma = covariance.values

        # Create view matrices
        P, Q, Omega = self.create_view_matrix(views, asset_names)
        Q = Q.reshape(-1, 1)

        # Scaled covariance: τΣ
        tau_Sigma = self.tau * Sigma

        # Compute posterior expected returns
        # Term 1: (τΣ)^(-1)
        tau_Sigma_inv = inv(tau_Sigma)

        # Term 2: P'Ω^(-1)P
        Omega_inv = inv(Omega)
        P_Omega_P = P.T @ Omega_inv @ P

        # Posterior precision: (τΣ)^(-1) + P'Ω^(-1)P
        posterior_precision = tau_Sigma_inv + P_Omega_P

        # Posterior covariance
        posterior_cov = inv(posterior_precision)

        # Posterior mean: M^(-1) * [(τΣ)^(-1)Π + P'Ω^(-1)Q]
        posterior_mean = posterior_cov @ (tau_Sigma_inv @ Pi + P.T @ Omega_inv @ Q)

        # Convert back to pandas
        posterior_returns = pd.Series(posterior_mean.flatten(), index=asset_names)
        posterior_covariance = pd.DataFrame(posterior_cov, index=asset_names, columns=asset_names)

        return posterior_returns, posterior_covariance

    def run(
        self,
        covariance: pd.DataFrame,
        views: List[InvestorView],
        market_weights: Optional[pd.Series] = None
    ) -> Dict:
        """
        Run complete Black-Litterman model.

        Args:
            covariance: Covariance matrix
            views: List of investor views
            market_weights: Market cap weights (optional)

        Returns:
            Dictionary with equilibrium returns, posterior returns, and covariances
        """
        # Step 1: Compute equilibrium returns
        equilibrium_returns = self.compute_equilibrium_returns(covariance, market_weights)

        # Step 2: Compute posterior returns
        posterior_returns, posterior_cov = self.compute_posterior_returns(
            equilibrium_returns,
            covariance,
            views
        )

        return {
            'equilibrium_returns': equilibrium_returns,
            'posterior_returns': posterior_returns,
            'posterior_covariance': posterior_cov,
            'views': views,
            'tau': self.tau,
            'risk_aversion': self.risk_aversion
        }


class BlackLittermanOptimizer:
    """
    Portfolio optimizer that uses Black-Litterman expected returns.
    """

    def __init__(
        self,
        bl_model: BlackLittermanModel,
        optimization_method: str = 'mean_variance'
    ):
        """
        Initialize BL optimizer.

        Args:
            bl_model: Black-Litterman model instance
            optimization_method: 'mean_variance' or 'max_sharpe'
        """
        self.bl_model = bl_model
        self.optimization_method = optimization_method

    def optimize(
        self,
        covariance: pd.DataFrame,
        views: List[InvestorView],
        market_weights: Optional[pd.Series] = None,
        risk_aversion: Optional[float] = None
    ) -> pd.Series:
        """
        Optimize portfolio using Black-Litterman expected returns.

        Args:
            covariance: Covariance matrix
            views: Investor views
            market_weights: Market cap weights
            risk_aversion: Risk aversion (overrides model default)

        Returns:
            Optimal portfolio weights
        """
        # Run Black-Litterman model
        bl_result = self.bl_model.run(covariance, views, market_weights)

        posterior_returns = bl_result['posterior_returns']
        posterior_cov = bl_result['posterior_covariance']

        # Optimize portfolio
        if risk_aversion is None:
            risk_aversion = self.bl_model.risk_aversion

        if self.optimization_method == 'mean_variance':
            # Analytical solution: w = (1/δ) * Σ^(-1) * μ
            cov_inv = inv(posterior_cov.values)
            weights = (1.0 / risk_aversion) * (cov_inv @ posterior_returns.values)

            # Normalize to sum to 1
            weights = weights / weights.sum()

            # Ensure long-only
            weights = np.maximum(weights, 0)
            weights = weights / weights.sum()

            return pd.Series(weights, index=posterior_returns.index)

        elif self.optimization_method == 'max_sharpe':
            # Maximum Sharpe ratio
            cov_inv = inv(posterior_cov.values)
            ones = np.ones(len(posterior_returns))

            # w = Σ^(-1) * μ / (1' * Σ^(-1) * μ)
            weights = cov_inv @ posterior_returns.values
            weights = weights / (ones @ weights)

            # Ensure long-only
            weights = np.maximum(weights, 0)
            weights = weights / weights.sum()

            return pd.Series(weights, index=posterior_returns.index)

        else:
            raise ValueError(f"Unknown optimization method: {self.optimization_method}")


def create_absolute_view(
    asset: str,
    expected_return: float,
    confidence: float = 0.5
) -> InvestorView:
    """Helper function to create an absolute view on a single asset."""
    return InvestorView(
        assets=[asset],
        weights=[1.0],
        expected_return=expected_return,
        confidence=confidence
    )


def create_relative_view(
    asset_long: str,
    asset_short: str,
    expected_outperformance: float,
    confidence: float = 0.5
) -> InvestorView:
    """Helper function to create a relative view (asset A outperforms asset B)."""
    return InvestorView(
        assets=[asset_long, asset_short],
        weights=[1.0, -1.0],
        expected_return=expected_outperformance,
        confidence=confidence
    )


if __name__ == "__main__":
    # Example usage
    print("Testing Black-Litterman Model...")

    # Sample data
    assets = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    n = len(assets)

    # Create sample covariance matrix
    np.random.seed(42)
    volatilities = np.array([0.20, 0.18, 0.22, 0.25, 0.30])
    correlation = np.array([
        [1.00, 0.50, 0.40, 0.35, 0.30],
        [0.50, 1.00, 0.45, 0.40, 0.25],
        [0.40, 0.45, 1.00, 0.50, 0.35],
        [0.35, 0.40, 0.50, 1.00, 0.40],
        [0.30, 0.25, 0.35, 0.40, 1.00]
    ])
    covariance = np.outer(volatilities, volatilities) * correlation
    cov_df = pd.DataFrame(covariance, index=assets, columns=assets)

    # Market cap weights (example)
    market_caps = pd.Series([3000, 2800, 1800, 1600, 800], index=assets)
    market_weights = market_caps / market_caps.sum()

    print(f"\nMarket Weights:")
    print(market_weights.round(4))

    # Initialize Black-Litterman model
    bl_model = BlackLittermanModel(risk_aversion=2.5, tau=0.05)

    # Compute equilibrium returns
    eq_returns = bl_model.compute_equilibrium_returns(cov_df, market_weights)
    print(f"\nEquilibrium Returns:")
    print(eq_returns.round(4))

    # Create investor views
    views = [
        # Absolute view: AAPL will return 15%
        create_absolute_view('AAPL', 0.15, confidence=0.7),

        # Relative view: TSLA will outperform AMZN by 5%
        create_relative_view('TSLA', 'AMZN', 0.05, confidence=0.6),

        # Relative view: MSFT will outperform GOOGL by 3%
        create_relative_view('MSFT', 'GOOGL', 0.03, confidence=0.5)
    ]

    print(f"\nInvestor Views:")
    for i, view in enumerate(views, 1):
        print(f"  View {i}: {view}")

    # Run Black-Litterman model
    bl_result = bl_model.run(cov_df, views, market_weights)

    print(f"\nPosterior Returns:")
    print(bl_result['posterior_returns'].round(4))

    # Compare equilibrium vs posterior
    comparison = pd.DataFrame({
        'Equilibrium': bl_result['equilibrium_returns'],
        'Posterior': bl_result['posterior_returns'],
        'Difference': bl_result['posterior_returns'] - bl_result['equilibrium_returns']
    })

    print(f"\nReturn Comparison:")
    print(comparison.round(4))

    # Optimize portfolio using BL returns
    bl_optimizer = BlackLittermanOptimizer(bl_model, optimization_method='mean_variance')
    optimal_weights = bl_optimizer.optimize(cov_df, views, market_weights)

    print(f"\nOptimal Portfolio Weights (Black-Litterman):")
    print(optimal_weights.round(4))

    # Compare with market weights
    weight_comparison = pd.DataFrame({
        'Market': market_weights,
        'BL Optimal': optimal_weights,
        'Difference': optimal_weights - market_weights
    })

    print(f"\nWeight Comparison:")
    print(weight_comparison.round(4))

    # Calculate portfolio metrics
    bl_return = (optimal_weights * bl_result['posterior_returns']).sum()
    bl_variance = optimal_weights @ bl_result['posterior_covariance'] @ optimal_weights
    bl_volatility = np.sqrt(bl_variance)
    bl_sharpe = bl_return / bl_volatility

    print(f"\nPortfolio Metrics:")
    print(f"  Expected Return: {bl_return:.4f}")
    print(f"  Volatility: {bl_volatility:.4f}")
    print(f"  Sharpe Ratio: {bl_sharpe:.4f}")

    print("\n✅ Black-Litterman model implementation complete!")
