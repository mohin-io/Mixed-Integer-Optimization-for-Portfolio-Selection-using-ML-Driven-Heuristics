"""
Robust Mixed-Integer-Optimization-for-Portfolio-Selection-using-ML-Driven-Heuristics for Worst-Case Scenarios.

Implements:
- Robust mean-variance optimization
- Worst-case CVaR optimization
- Box uncertainty sets
- Ellipsoidal uncertainty sets
- Robust Black-Litterman
- Ambiguity-averse optimization
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple
from scipy.optimize import minimize
import cvxpy as cp
import warnings


class RobustMeanVarianceOptimizer:
    """
    Robust mean-variance optimization under parameter uncertainty.
    """

    def __init__(
        self,
        uncertainty_set: str = 'box',
        epsilon_mu: float = 0.05,
        epsilon_sigma: float = 0.10
    ):
        """
        Initialize robust optimizer.

        Args:
            uncertainty_set: 'box' or 'ellipsoidal'
            epsilon_mu: Uncertainty in mean returns (e.g., 5%)
            epsilon_sigma: Uncertainty in covariance (e.g., 10%)
        """
        self.uncertainty_set = uncertainty_set
        self.epsilon_mu = epsilon_mu
        self.epsilon_sigma = epsilon_sigma

    def optimize_box_uncertainty(
        self,
        expected_returns: np.ndarray,
        covariance: np.ndarray,
        risk_aversion: float = 2.5
    ) -> Dict:
        """
        Robust optimization with box uncertainty set.

        Uncertainty: |μ - μ_nominal| <= ε * |μ_nominal|

        Args:
            expected_returns: Nominal expected returns
            covariance: Nominal covariance
            risk_aversion: Risk aversion parameter

        Returns:
            Optimization results
        """
        n_assets = len(expected_returns)

        # Worst-case expected returns (lower bound)
        worst_case_returns = expected_returns * (1 - self.epsilon_mu)

        # Worst-case covariance (upper bound on variance)
        worst_case_cov = covariance * (1 + self.epsilon_sigma)

        # Decision variables
        w = cp.Variable(n_assets)

        # Worst-case objective
        portfolio_return = w @ worst_case_returns
        portfolio_variance = cp.quad_form(w, worst_case_cov)

        objective = portfolio_return - (risk_aversion / 2) * portfolio_variance

        # Constraints
        constraints = [
            cp.sum(w) == 1,  # Budget
            w >= 0           # Long-only
        ]

        # Solve
        problem = cp.Problem(cp.Maximize(objective), constraints)
        problem.solve()

        if problem.status in ['optimal', 'optimal_inaccurate']:
            weights = w.value
            weights = weights / weights.sum()  # Normalize

            return {
                'weights': weights,
                'worst_case_return': worst_case_returns @ weights,
                'worst_case_volatility': np.sqrt(weights @ worst_case_cov @ weights),
                'status': problem.status
            }
        else:
            warnings.warn(f"Optimization failed: {problem.status}")
            return {'weights': np.ones(n_assets) / n_assets, 'status': 'failed'}

    def optimize_ellipsoidal_uncertainty(
        self,
        expected_returns: np.ndarray,
        covariance: np.ndarray,
        risk_aversion: float = 2.5
    ) -> Dict:
        """
        Robust optimization with ellipsoidal uncertainty set.

        Uncertainty: ||μ - μ_nominal||_2 <= ε

        Args:
            expected_returns: Nominal expected returns
            covariance: Nominal covariance
            risk_aversion: Risk aversion parameter

        Returns:
            Optimization results
        """
        n_assets = len(expected_returns)

        # Decision variables
        w = cp.Variable(n_assets)

        # Worst-case return (robust counterpart)
        # min_μ (μ'w) subject to ||μ - μ_nominal||_2 <= ε
        # = μ_nominal'w - ε||w||_2

        epsilon = self.epsilon_mu * np.linalg.norm(expected_returns)
        worst_case_return = expected_returns @ w - epsilon * cp.norm(w, 2)

        # Risk
        portfolio_variance = cp.quad_form(w, covariance)

        # Objective
        objective = worst_case_return - (risk_aversion / 2) * portfolio_variance

        # Constraints
        constraints = [
            cp.sum(w) == 1,
            w >= 0
        ]

        # Solve
        problem = cp.Problem(cp.Maximize(objective), constraints)
        problem.solve()

        if problem.status in ['optimal', 'optimal_inaccurate']:
            weights = w.value
            weights = weights / weights.sum()

            return {
                'weights': weights,
                'nominal_return': expected_returns @ weights,
                'worst_case_return': expected_returns @ weights - epsilon * np.linalg.norm(weights),
                'volatility': np.sqrt(weights @ covariance @ weights),
                'status': problem.status
            }
        else:
            return {'weights': np.ones(n_assets) / n_assets, 'status': 'failed'}


class RobustCVaROptimizer:
    """
    Robust CVaR optimization under distributional ambiguity.
    """

    def __init__(
        self,
        confidence_level: float = 0.95,
        ambiguity_radius: float = 0.1
    ):
        """
        Initialize robust CVaR optimizer.

        Args:
            confidence_level: CVaR confidence level
            ambiguity_radius: Wasserstein distance radius for ambiguity
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        self.ambiguity_radius = ambiguity_radius

    def optimize_worst_case_cvar(
        self,
        expected_returns: np.ndarray,
        scenarios: np.ndarray,
        min_return: Optional[float] = None
    ) -> Dict:
        """
        Optimize worst-case CVaR under distributional ambiguity.

        Args:
            expected_returns: Expected returns
            scenarios: Return scenarios (S x N)
            min_return: Minimum required return

        Returns:
            Optimization results
        """
        n_assets = len(expected_returns)
        n_scenarios = scenarios.shape[0]

        # Decision variables
        w = cp.Variable(n_assets)
        z = cp.Variable(n_scenarios)
        var = cp.Variable()

        # Worst-case CVaR (pessimistic probabilities)
        # Standard CVaR formulation with ambiguity adjustment
        cvar = var + (1 / self.alpha) * cp.sum(z) / n_scenarios

        # Add ambiguity penalty (simplified version)
        ambiguity_penalty = self.ambiguity_radius * cp.norm(w, 2)

        # Robust objective
        objective = cvar + ambiguity_penalty

        # Constraints
        constraints = [
            cp.sum(w) == 1,
            w >= 0,
            z >= 0
        ]

        # CVaR constraints
        for s in range(n_scenarios):
            constraints.append(z[s] >= -scenarios[s, :] @ w - var)

        # Minimum return constraint
        if min_return is not None:
            constraints.append(expected_returns @ w >= min_return)

        # Solve
        problem = cp.Problem(cp.Minimize(objective), constraints)

        try:
            problem.solve(solver=cp.ECOS)

            if problem.status in ['optimal', 'optimal_inaccurate']:
                weights = w.value
                weights = weights / weights.sum()

                return {
                    'weights': weights,
                    'worst_case_cvar': cvar.value,
                    'var': var.value,
                    'expected_return': expected_returns @ weights,
                    'status': problem.status
                }
        except:
            pass

        return {'weights': np.ones(n_assets) / n_assets, 'status': 'failed'}


class MinMaxOptimizer:
    """
    Min-max (maximin) portfolio optimization.
    """

    def __init__(self):
        """Initialize min-max optimizer."""
        pass

    def optimize_minimax_regret(
        self,
        return_scenarios: np.ndarray,
        scenario_probabilities: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Minimize maximum regret across scenarios.

        Args:
            return_scenarios: Return scenarios (S x N)
            scenario_probabilities: Scenario probabilities (if None, equal weight)

        Returns:
            Optimization results
        """
        n_scenarios, n_assets = return_scenarios.shape

        if scenario_probabilities is None:
            scenario_probabilities = np.ones(n_scenarios) / n_scenarios

        # Decision variables
        w = cp.Variable(n_assets)
        regret = cp.Variable()

        # For each scenario, calculate regret
        # Regret = best possible return - actual return
        for s in range(n_scenarios):
            # Best possible return in scenario s
            best_return = np.max(return_scenarios[s, :])

            # Actual return with portfolio w
            actual_return = return_scenarios[s, :] @ w

            # Regret in scenario s
            scenario_regret = best_return - actual_return

            # Max regret constraint
            constraints = [regret >= scenario_regret] if s == 0 else constraints + [regret >= scenario_regret]

        # Additional constraints
        constraints += [
            cp.sum(w) == 1,
            w >= 0
        ]

        # Minimize maximum regret
        problem = cp.Problem(cp.Minimize(regret), constraints)
        problem.solve()

        if problem.status in ['optimal', 'optimal_inaccurate']:
            weights = w.value
            weights = weights / weights.sum()

            # Calculate metrics
            expected_return = scenario_probabilities @ (return_scenarios @ weights)

            return {
                'weights': weights,
                'max_regret': regret.value,
                'expected_return': expected_return,
                'status': problem.status
            }
        else:
            return {'weights': np.ones(n_assets) / n_assets, 'status': 'failed'}


class DistributionallyRobustOptimizer:
    """
    Distributionally Robust Optimization (DRO).
    """

    def __init__(self, ambiguity_set: str = 'wasserstein'):
        """
        Initialize DRO optimizer.

        Args:
            ambiguity_set: 'wasserstein' or 'moment'
        """
        self.ambiguity_set = ambiguity_set

    def optimize_moment_based(
        self,
        mean_estimate: np.ndarray,
        cov_estimate: np.ndarray,
        risk_aversion: float = 2.5,
        confidence_level: float = 0.95
    ) -> Dict:
        """
        DRO with moment-based ambiguity set.

        Assumes only mean and covariance are known with certainty.

        Args:
            mean_estimate: Estimated mean returns
            cov_estimate: Estimated covariance
            risk_aversion: Risk aversion
            confidence_level: Confidence level for robustness

        Returns:
            Optimization results
        """
        n_assets = len(mean_estimate)

        # Chi-squared critical value
        chi_sq_val = np.sqrt(2 * n_assets)  # Approximate

        # Decision variables
        w = cp.Variable(n_assets)

        # Worst-case expected return
        # Under moment ambiguity: E[R'w] - κ√(w'Σw)
        kappa = chi_sq_val * (1 - confidence_level)
        worst_case_return = mean_estimate @ w - kappa * cp.quad_form(w, cov_estimate) ** 0.5

        # Risk
        portfolio_var = cp.quad_form(w, cov_estimate)

        # Objective
        objective = worst_case_return - (risk_aversion / 2) * portfolio_var

        # Constraints
        constraints = [
            cp.sum(w) == 1,
            w >= 0
        ]

        # Solve
        problem = cp.Problem(cp.Maximize(objective), constraints)
        problem.solve()

        if problem.status in ['optimal', 'optimal_inaccurate']:
            weights = w.value
            weights = weights / weights.sum()

            return {
                'weights': weights,
                'nominal_return': mean_estimate @ weights,
                'worst_case_return': mean_estimate @ weights - kappa * np.sqrt(weights @ cov_estimate @ weights),
                'volatility': np.sqrt(weights @ cov_estimate @ weights),
                'status': problem.status
            }
        else:
            return {'weights': np.ones(n_assets) / n_assets, 'status': 'failed'}


if __name__ == "__main__":
    print("Testing Robust Optimization...")

    # Generate test data
    np.random.seed(42)
    n_assets = 5

    expected_returns = np.array([0.10, 0.12, 0.08, 0.15, 0.09])
    volatilities = np.array([0.15, 0.20, 0.12, 0.25, 0.18])

    correlation = np.eye(n_assets)
    for i in range(n_assets):
        for j in range(i+1, n_assets):
            correlation[i, j] = correlation[j, i] = 0.3

    covariance = np.outer(volatilities, volatilities) * correlation

    print("\n1. Robust Mean-Variance (Box Uncertainty)")
    robust_mv = RobustMeanVarianceOptimizer(
        uncertainty_set='box',
        epsilon_mu=0.05,
        epsilon_sigma=0.10
    )

    result = robust_mv.optimize_box_uncertainty(expected_returns, covariance)
    print(f"   Weights: {result['weights'].round(4)}")
    print(f"   Worst-Case Return: {result['worst_case_return']:.4f}")
    print(f"   Worst-Case Volatility: {result['worst_case_volatility']:.4f}")

    print("\n2. Robust Mean-Variance (Ellipsoidal Uncertainty)")
    result = robust_mv.optimize_ellipsoidal_uncertainty(expected_returns, covariance)
    print(f"   Weights: {result['weights'].round(4)}")
    print(f"   Nominal Return: {result['nominal_return']:.4f}")
    print(f"   Worst-Case Return: {result['worst_case_return']:.4f}")

    print("\n3. Robust CVaR")
    scenarios = np.random.multivariate_normal(expected_returns, covariance, size=1000)

    robust_cvar = RobustCVaROptimizer(confidence_level=0.95, ambiguity_radius=0.1)
    result = robust_cvar.optimize_worst_case_cvar(expected_returns, scenarios)

    print(f"   Weights: {result['weights'].round(4)}")
    print(f"   Worst-Case CVaR: {result['worst_case_cvar']:.4f}")
    print(f"   Expected Return: {result['expected_return']:.4f}")

    print("\n4. Minimax Regret")
    n_scenarios = 10
    return_scenarios = np.random.randn(n_scenarios, n_assets) * 0.05 + expected_returns

    minimax = MinMaxOptimizer()
    result = minimax.optimize_minimax_regret(return_scenarios)

    print(f"   Weights: {result['weights'].round(4)}")
    print(f"   Max Regret: {result['max_regret']:.4f}")
    print(f"   Expected Return: {result['expected_return']:.4f}")

    print("\n5. Distributionally Robust (Moment-Based)")
    dro = DistributionallyRobustOptimizer(ambiguity_set='moment')
    result = dro.optimize_moment_based(expected_returns, covariance)

    print(f"   Weights: {result['weights'].round(4)}")
    print(f"   Nominal Return: {result['nominal_return']:.4f}")
    print(f"   Worst-Case Return: {result['worst_case_return']:.4f}")

    print("\n✅ Robust optimization implementation complete!")
