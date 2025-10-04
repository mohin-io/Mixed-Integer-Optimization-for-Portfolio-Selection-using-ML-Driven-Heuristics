"""
CVaR (Conditional Value-at-Risk) Portfolio Optimization.

Implements:
- CVaR risk measure (Expected Shortfall)
- CVaR minimization with return constraints
- CVaR-based portfolio optimization
- Integration with MIO framework
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple
import cvxpy as cp
from scipy.optimize import minimize
import warnings


class CVaROptimizer:
    """
    Conditional Value-at-Risk (CVaR) portfolio optimizer.

    CVaR is the expected loss given that the loss exceeds VaR.
    Also known as Expected Shortfall (ES) or Tail VaR.
    """

    def __init__(
        self,
        confidence_level: float = 0.95,
        scenario_based: bool = True,
        n_scenarios: int = 1000
    ):
        """
        Initialize CVaR optimizer.

        Args:
            confidence_level: Confidence level for VaR/CVaR (e.g., 0.95 for 95%)
            scenario_based: If True, use scenario-based optimization
            n_scenarios: Number of scenarios for Monte Carlo simulation
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        self.scenario_based = scenario_based
        self.n_scenarios = n_scenarios

    def calculate_cvar_historical(
        self,
        returns: np.ndarray,
        weights: np.ndarray,
        confidence_level: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Calculate CVaR using historical returns.

        Args:
            returns: Historical returns matrix (T x N)
            weights: Portfolio weights (N,)
            confidence_level: Confidence level (overrides default)

        Returns:
            Tuple of (VaR, CVaR)
        """
        if confidence_level is None:
            confidence_level = self.confidence_level

        # Calculate portfolio returns
        portfolio_returns = returns @ weights

        # Calculate VaR (negative of percentile since we look at losses)
        var = -np.percentile(portfolio_returns, (1 - confidence_level) * 100)

        # Calculate CVaR (mean of returns below VaR threshold)
        losses = -portfolio_returns
        cvar = np.mean(losses[losses >= var])

        return var, cvar

    def optimize_cvar_cvxpy(
        self,
        expected_returns: np.ndarray,
        scenarios: np.ndarray,
        min_return: Optional[float] = None,
        max_weight: float = 1.0,
        min_weight: float = 0.0,
        cardinality: Optional[int] = None
    ) -> Dict:
        """
        Optimize portfolio to minimize CVaR using CVXPY.

        Formulation:
        minimize: CVaR_α(w)
        subject to: E[R'w] >= min_return
                   Σw = 1
                   w >= 0

        Args:
            expected_returns: Expected returns for each asset
            scenarios: Return scenarios matrix (S x N)
            min_return: Minimum required portfolio return
            max_weight: Maximum weight per asset
            min_weight: Minimum weight per asset
            cardinality: Maximum number of assets (if specified)

        Returns:
            Dict with optimal weights, CVaR, VaR, and other metrics
        """
        n_assets = len(expected_returns)
        n_scenarios = scenarios.shape[0]

        # Decision variables
        w = cp.Variable(n_assets)  # Portfolio weights
        z = cp.Variable(n_scenarios)  # Auxiliary variables for CVaR
        var = cp.Variable()  # Value-at-Risk

        # CVaR objective: VaR + (1/α) * E[max(0, -R'w - VaR)]
        cvar = var + (1 / self.alpha) * cp.sum(z) / n_scenarios

        # Constraints
        constraints = [
            cp.sum(w) == 1,  # Budget constraint
            w >= min_weight,  # Lower bound
            w <= max_weight,  # Upper bound
            z >= 0,  # Auxiliary variables non-negative
        ]

        # CVaR constraint: z_s >= -R_s'w - VaR
        for s in range(n_scenarios):
            constraints.append(z[s] >= -scenarios[s, :] @ w - var)

        # Minimum return constraint
        if min_return is not None:
            constraints.append(expected_returns @ w >= min_return)

        # Cardinality constraint (approximation using L1 norm)
        if cardinality is not None:
            # This is a relaxation; for exact cardinality, use MIO
            constraints.append(cp.norm(w, 1) <= cardinality * max_weight)

        # Solve problem
        problem = cp.Problem(cp.Minimize(cvar), constraints)

        try:
            problem.solve(solver=cp.ECOS)

            if problem.status not in ["optimal", "optimal_inaccurate"]:
                warnings.warn(f"Optimization status: {problem.status}")

            # Extract results
            optimal_weights = w.value
            optimal_cvar = cvar.value
            optimal_var = var.value
            expected_return = expected_returns @ optimal_weights

            return {
                'weights': optimal_weights,
                'cvar': optimal_cvar,
                'var': optimal_var,
                'expected_return': expected_return,
                'status': problem.status
            }

        except Exception as e:
            warnings.warn(f"CVaR optimization failed: {e}")
            # Return equal weights as fallback
            equal_weights = np.ones(n_assets) / n_assets
            return {
                'weights': equal_weights,
                'cvar': None,
                'var': None,
                'expected_return': expected_returns @ equal_weights,
                'status': 'failed'
            }

    def generate_scenarios(
        self,
        expected_returns: np.ndarray,
        covariance: np.ndarray,
        n_scenarios: Optional[int] = None,
        method: str = 'monte_carlo'
    ) -> np.ndarray:
        """
        Generate return scenarios for CVaR optimization.

        Args:
            expected_returns: Expected returns vector
            covariance: Covariance matrix
            n_scenarios: Number of scenarios (overrides default)
            method: 'monte_carlo', 'historical', or 'bootstrap'

        Returns:
            Scenarios matrix (S x N)
        """
        if n_scenarios is None:
            n_scenarios = self.n_scenarios

        if method == 'monte_carlo':
            # Monte Carlo simulation from normal distribution
            scenarios = np.random.multivariate_normal(
                expected_returns,
                covariance,
                size=n_scenarios
            )

        elif method == 'bootstrap':
            # Bootstrap resampling (requires historical returns)
            # Placeholder: use Monte Carlo if historical data not available
            scenarios = np.random.multivariate_normal(
                expected_returns,
                covariance,
                size=n_scenarios
            )

        else:
            raise ValueError(f"Unknown scenario generation method: {method}")

        return scenarios

    def optimize(
        self,
        expected_returns: np.ndarray,
        covariance: np.ndarray,
        min_return: Optional[float] = None,
        max_weight: float = 1.0,
        min_weight: float = 0.0,
        cardinality: Optional[int] = None,
        scenarios: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Main optimization method.

        Args:
            expected_returns: Expected returns for each asset
            covariance: Covariance matrix
            min_return: Minimum required portfolio return
            max_weight: Maximum weight per asset
            min_weight: Minimum weight per asset
            cardinality: Maximum number of assets
            scenarios: Pre-generated scenarios (if None, will generate)

        Returns:
            Optimization results dictionary
        """
        # Generate scenarios if not provided
        if scenarios is None:
            scenarios = self.generate_scenarios(
                expected_returns,
                covariance,
                n_scenarios=self.n_scenarios
            )

        # Run optimization
        result = self.optimize_cvar_cvxpy(
            expected_returns,
            scenarios,
            min_return=min_return,
            max_weight=max_weight,
            min_weight=min_weight,
            cardinality=cardinality
        )

        return result

    def efficient_frontier_cvar(
        self,
        expected_returns: np.ndarray,
        covariance: np.ndarray,
        n_points: int = 50,
        scenarios: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Compute CVaR efficient frontier.

        Args:
            expected_returns: Expected returns for each asset
            covariance: Covariance matrix
            n_points: Number of points on the frontier
            scenarios: Pre-generated scenarios

        Returns:
            DataFrame with returns, CVaR, and weights for each frontier point
        """
        # Generate scenarios if not provided
        if scenarios is None:
            scenarios = self.generate_scenarios(expected_returns, covariance)

        # Range of target returns
        min_ret = np.min(expected_returns)
        max_ret = np.max(expected_returns)
        target_returns = np.linspace(min_ret, max_ret, n_points)

        results = []

        for target_ret in target_returns:
            result = self.optimize_cvar_cvxpy(
                expected_returns,
                scenarios,
                min_return=target_ret
            )

            if result['status'] in ['optimal', 'optimal_inaccurate']:
                results.append({
                    'return': result['expected_return'],
                    'cvar': result['cvar'],
                    'var': result['var'],
                    'weights': result['weights']
                })

        return pd.DataFrame(results)


class RobustCVaROptimizer(CVaROptimizer):
    """
    Robust CVaR optimizer that accounts for parameter uncertainty.

    Uses worst-case CVaR optimization under uncertainty sets.
    """

    def __init__(
        self,
        confidence_level: float = 0.95,
        uncertainty_set: str = 'box',
        robustness_param: float = 0.1
    ):
        """
        Initialize robust CVaR optimizer.

        Args:
            confidence_level: Confidence level for CVaR
            uncertainty_set: Type of uncertainty set ('box', 'ellipsoidal')
            robustness_param: Size of uncertainty set
        """
        super().__init__(confidence_level=confidence_level)
        self.uncertainty_set = uncertainty_set
        self.robustness_param = robustness_param

    def optimize_robust_cvar(
        self,
        expected_returns: np.ndarray,
        covariance: np.ndarray,
        min_return: Optional[float] = None
    ) -> Dict:
        """
        Optimize portfolio using robust CVaR under parameter uncertainty.

        Args:
            expected_returns: Nominal expected returns
            covariance: Nominal covariance matrix
            min_return: Minimum required return

        Returns:
            Optimization results
        """
        n_assets = len(expected_returns)

        # Decision variables
        w = cp.Variable(n_assets)

        # Uncertainty set for expected returns
        if self.uncertainty_set == 'box':
            # Box uncertainty: |μ - μ_nominal| <= δ
            delta = self.robustness_param * np.abs(expected_returns)
            worst_case_return = expected_returns - delta

        elif self.uncertainty_set == 'ellipsoidal':
            # Ellipsoidal uncertainty: ||μ - μ_nominal||_2 <= δ
            # Approximation: worst case in direction of -w
            worst_case_return = expected_returns * (1 - self.robustness_param)

        else:
            worst_case_return = expected_returns

        # Generate scenarios with perturbed parameters
        scenarios = self.generate_scenarios(
            worst_case_return,
            covariance * (1 + self.robustness_param)
        )

        # Solve robust optimization
        result = self.optimize_cvar_cvxpy(
            worst_case_return,
            scenarios,
            min_return=min_return
        )

        result['robustness'] = self.robustness_param
        return result


if __name__ == "__main__":
    # Example usage
    print("Testing CVaR Optimization...")

    # Generate sample data
    np.random.seed(42)
    n_assets = 5

    expected_returns = np.array([0.10, 0.12, 0.08, 0.15, 0.09])
    volatilities = np.array([0.15, 0.20, 0.12, 0.25, 0.18])

    # Create covariance matrix
    correlation = np.array([
        [1.0, 0.5, 0.3, 0.2, 0.4],
        [0.5, 1.0, 0.4, 0.3, 0.5],
        [0.3, 0.4, 1.0, 0.2, 0.3],
        [0.2, 0.3, 0.2, 1.0, 0.4],
        [0.4, 0.5, 0.3, 0.4, 1.0]
    ])

    covariance = np.outer(volatilities, volatilities) * correlation

    # Initialize optimizer
    cvar_opt = CVaROptimizer(confidence_level=0.95, n_scenarios=1000)

    # Optimize portfolio
    result = cvar_opt.optimize(
        expected_returns,
        covariance,
        min_return=0.10
    )

    print(f"\nOptimal Weights:")
    for i, w in enumerate(result['weights']):
        print(f"  Asset {i+1}: {w:.4f}")

    print(f"\nExpected Return: {result['expected_return']:.4f}")
    print(f"VaR (95%): {result['var']:.4f}")
    print(f"CVaR (95%): {result['cvar']:.4f}")
    print(f"Optimization Status: {result['status']}")

    # Compute efficient frontier
    print("\nComputing CVaR efficient frontier...")
    frontier = cvar_opt.efficient_frontier_cvar(
        expected_returns,
        covariance,
        n_points=20
    )

    print(f"\nEfficient Frontier (first 5 points):")
    print(frontier[['return', 'cvar', 'var']].head())

    # Test robust optimization
    print("\nTesting Robust CVaR Optimization...")
    robust_opt = RobustCVaROptimizer(
        confidence_level=0.95,
        robustness_param=0.1
    )

    robust_result = robust_opt.optimize_robust_cvar(
        expected_returns,
        covariance,
        min_return=0.10
    )

    print(f"\nRobust Optimal Weights:")
    for i, w in enumerate(robust_result['weights']):
        print(f"  Asset {i+1}: {w:.4f}")

    print(f"\nRobust CVaR: {robust_result['cvar']:.4f}")
    print(f"Robustness Parameter: {robust_result['robustness']}")

    print("\n✅ CVaR optimization implementation complete!")
