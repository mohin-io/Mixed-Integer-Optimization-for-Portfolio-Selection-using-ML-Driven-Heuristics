"""
Multi-Period Portfolio Optimization with Dynamic Programming.

Implements:
- Dynamic programming for multi-period optimization
- Stochastic dynamic programming with scenario trees
- Transaction costs across multiple periods
- Rebalancing policies
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple, Callable
from dataclasses import dataclass
from scipy.optimize import minimize
import warnings


@dataclass
class MultiPeriodConfig:
    """Configuration for multi-period optimization."""
    n_periods: int = 12  # Number of periods to optimize
    risk_aversion: float = 2.5
    transaction_cost: float = 0.001  # Proportional transaction cost
    max_weight: float = 0.5  # Maximum weight per asset
    min_weight: float = 0.0  # Minimum weight per asset
    rebalance_threshold: float = 0.05  # Trigger rebalancing if drift > threshold
    discount_factor: float = 0.99  # Time discount factor


@dataclass
class ScenarioTree:
    """Scenario tree for stochastic optimization."""
    returns: np.ndarray  # (n_scenarios x n_periods x n_assets)
    probabilities: np.ndarray  # (n_scenarios,)
    n_scenarios: int
    n_periods: int
    n_assets: int


class MultiPeriodOptimizer:
    """
    Multi-period portfolio optimization using dynamic programming.

    Solves:
    max E[Σ_t γ^t U(W_t)]

    where:
    - W_t: wealth at time t
    - γ: discount factor
    - U: utility function
    - Subject to transaction costs and constraints
    """

    def __init__(self, config: Optional[MultiPeriodConfig] = None):
        """
        Initialize multi-period optimizer.

        Args:
            config: Optimization configuration
        """
        self.config = config or MultiPeriodConfig()

    def generate_scenarios(
        self,
        expected_returns: np.ndarray,
        covariance: np.ndarray,
        n_scenarios: int = 100,
        method: str = 'monte_carlo'
    ) -> ScenarioTree:
        """
        Generate return scenarios for stochastic optimization.

        Args:
            expected_returns: Expected returns per period (n_assets,)
            covariance: Covariance matrix (n_assets x n_assets)
            n_scenarios: Number of scenarios
            method: 'monte_carlo' or 'lattice'

        Returns:
            ScenarioTree object
        """
        n_assets = len(expected_returns)
        n_periods = self.config.n_periods

        if method == 'monte_carlo':
            # Monte Carlo simulation
            scenarios = np.random.multivariate_normal(
                expected_returns,
                covariance,
                size=(n_scenarios, n_periods)
            )

            # Equal probabilities
            probabilities = np.ones(n_scenarios) / n_scenarios

        elif method == 'lattice':
            # Simple binomial lattice (up/down scenarios)
            # For each period, create 2^period scenarios
            scenarios = []
            probabilities = []

            sigma = np.sqrt(np.diag(covariance))
            up_return = expected_returns + sigma
            down_return = expected_returns - sigma

            # Generate all combinations
            for i in range(n_scenarios):
                scenario_path = []
                for t in range(n_periods):
                    if np.random.rand() > 0.5:
                        scenario_path.append(up_return)
                    else:
                        scenario_path.append(down_return)

                scenarios.append(scenario_path)
                probabilities.append(1.0 / n_scenarios)

            scenarios = np.array(scenarios)
            probabilities = np.array(probabilities)

        else:
            raise ValueError(f"Unknown scenario generation method: {method}")

        return ScenarioTree(
            returns=scenarios,
            probabilities=probabilities,
            n_scenarios=n_scenarios,
            n_periods=n_periods,
            n_assets=n_assets
        )

    def utility_function(
        self,
        wealth: float,
        utility_type: str = 'power'
    ) -> float:
        """
        Compute utility of wealth.

        Args:
            wealth: Portfolio wealth
            utility_type: 'power', 'exponential', or 'quadratic'

        Returns:
            Utility value
        """
        if utility_type == 'power':
            # Power utility: U(W) = W^(1-γ) / (1-γ)
            gamma = self.config.risk_aversion
            if gamma == 1:
                return np.log(wealth)
            else:
                return (wealth ** (1 - gamma)) / (1 - gamma)

        elif utility_type == 'exponential':
            # Exponential utility: U(W) = -exp(-γW)
            gamma = self.config.risk_aversion
            return -np.exp(-gamma * wealth)

        elif utility_type == 'quadratic':
            # Quadratic utility: U(W) = W - (γ/2)W^2
            gamma = self.config.risk_aversion
            return wealth - (gamma / 2) * (wealth ** 2)

        else:
            raise ValueError(f"Unknown utility type: {utility_type}")

    def compute_transaction_costs(
        self,
        weights_old: np.ndarray,
        weights_new: np.ndarray,
        wealth: float
    ) -> float:
        """
        Compute transaction costs for rebalancing.

        Args:
            weights_old: Old portfolio weights
            weights_new: New portfolio weights
            wealth: Current wealth

        Returns:
            Transaction cost
        """
        # Proportional cost based on turnover
        turnover = np.sum(np.abs(weights_new - weights_old))
        cost = self.config.transaction_cost * turnover * wealth

        return cost

    def backward_induction(
        self,
        scenario_tree: ScenarioTree,
        initial_weights: Optional[np.ndarray] = None,
        initial_wealth: float = 1.0
    ) -> Dict:
        """
        Solve multi-period problem using backward induction (dynamic programming).

        Args:
            scenario_tree: Scenario tree with return paths
            initial_weights: Initial portfolio weights
            initial_wealth: Initial wealth

        Returns:
            Dictionary with optimal policy and value function
        """
        n_scenarios = scenario_tree.n_scenarios
        n_periods = scenario_tree.n_periods
        n_assets = scenario_tree.n_assets

        if initial_weights is None:
            initial_weights = np.ones(n_assets) / n_assets

        # Value function: V[scenario, period, state]
        # For simplicity, discretize wealth states
        n_wealth_states = 20
        wealth_grid = np.linspace(0.5, 2.0, n_wealth_states)

        # Policy: optimal weights at each (scenario, period, wealth_state)
        policy = {}

        # Terminal value function (period T)
        V_terminal = np.zeros((n_scenarios, n_wealth_states))
        for s in range(n_scenarios):
            for w_idx, wealth in enumerate(wealth_grid):
                V_terminal[s, w_idx] = self.utility_function(wealth)

        # Backward induction
        V_current = V_terminal

        for t in range(n_periods - 1, -1, -1):
            V_next = np.zeros((n_scenarios, n_wealth_states))

            for s in range(n_scenarios):
                for w_idx, wealth in enumerate(wealth_grid):
                    # Optimize portfolio weights for this state
                    def objective(weights):
                        # Ensure weights sum to 1
                        weights = weights / weights.sum()

                        # Transaction cost
                        prev_weights = initial_weights if t == 0 else policy.get((s, t-1, w_idx), initial_weights)
                        tc = self.compute_transaction_costs(prev_weights, weights, wealth)

                        # Expected value next period
                        returns_next = scenario_tree.returns[s, t, :]
                        wealth_next = wealth * (1 + np.dot(weights, returns_next)) - tc

                        # Interpolate value function
                        if wealth_next < wealth_grid[0]:
                            v_next = V_current[s, 0]
                        elif wealth_next > wealth_grid[-1]:
                            v_next = V_current[s, -1]
                        else:
                            v_next = np.interp(wealth_next, wealth_grid, V_current[s, :])

                        # Bellman equation: current utility + discounted future value
                        current_utility = self.utility_function(wealth)
                        total_value = current_utility + self.config.discount_factor * v_next

                        return -total_value  # Minimize negative value

                    # Optimization constraints
                    constraints = [
                        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Budget
                    ]

                    bounds = [(self.config.min_weight, self.config.max_weight) for _ in range(n_assets)]

                    # Solve for optimal weights
                    result = minimize(
                        objective,
                        x0=initial_weights,
                        method='SLSQP',
                        bounds=bounds,
                        constraints=constraints,
                        options={'maxiter': 100, 'disp': False}
                    )

                    optimal_weights = result.x / result.x.sum()
                    policy[(s, t, w_idx)] = optimal_weights

                    # Update value function
                    V_next[s, w_idx] = -result.fun

            V_current = V_next

        return {
            'policy': policy,
            'value_function': V_current,
            'wealth_grid': wealth_grid,
            'initial_weights': initial_weights
        }

    def deterministic_multi_period(
        self,
        expected_returns_path: np.ndarray,
        covariance_path: np.ndarray,
        initial_weights: Optional[np.ndarray] = None,
        initial_wealth: float = 1.0
    ) -> Dict:
        """
        Solve deterministic multi-period optimization (no uncertainty).

        Args:
            expected_returns_path: Expected returns for each period (n_periods x n_assets)
            covariance_path: Covariance for each period (n_periods x n_assets x n_assets)
            initial_weights: Initial portfolio weights
            initial_wealth: Initial wealth

        Returns:
            Dictionary with optimal weights and wealth trajectory
        """
        n_periods = expected_returns_path.shape[0]
        n_assets = expected_returns_path.shape[1]

        if initial_weights is None:
            initial_weights = np.ones(n_assets) / n_assets

        # Storage
        weights_trajectory = [initial_weights]
        wealth_trajectory = [initial_wealth]

        current_weights = initial_weights
        current_wealth = initial_wealth

        for t in range(n_periods):
            # Expected return and risk for period t
            mu = expected_returns_path[t]
            Sigma = covariance_path[t]

            # Single-period mean-variance optimization
            def objective(weights):
                portfolio_return = np.dot(weights, mu)
                portfolio_variance = weights @ Sigma @ weights

                # Transaction cost
                tc = self.compute_transaction_costs(current_weights, weights, current_wealth)

                # Mean-variance utility - transaction cost
                utility = portfolio_return - (self.config.risk_aversion / 2) * portfolio_variance
                utility_with_cost = utility - tc / current_wealth

                return -utility_with_cost

            constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
            bounds = [(self.config.min_weight, self.config.max_weight) for _ in range(n_assets)]

            result = minimize(
                objective,
                x0=current_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )

            optimal_weights = result.x / result.x.sum()

            # Update wealth
            tc = self.compute_transaction_costs(current_weights, optimal_weights, current_wealth)
            returns = np.dot(optimal_weights, mu)
            next_wealth = current_wealth * (1 + returns) - tc

            # Store
            weights_trajectory.append(optimal_weights)
            wealth_trajectory.append(next_wealth)

            # Update state
            current_weights = optimal_weights
            current_wealth = next_wealth

        return {
            'weights_trajectory': weights_trajectory,
            'wealth_trajectory': wealth_trajectory,
            'final_wealth': wealth_trajectory[-1],
            'total_return': (wealth_trajectory[-1] - initial_wealth) / initial_wealth
        }


class ThresholdRebalancingPolicy:
    """
    Threshold-based rebalancing policy.

    Rebalances only when portfolio drift exceeds threshold.
    """

    def __init__(
        self,
        target_weights: np.ndarray,
        threshold: float = 0.05,
        transaction_cost: float = 0.001
    ):
        """
        Initialize threshold rebalancing.

        Args:
            target_weights: Target portfolio weights
            threshold: Rebalancing threshold (fraction)
            transaction_cost: Transaction cost rate
        """
        self.target_weights = target_weights
        self.threshold = threshold
        self.transaction_cost = transaction_cost

    def should_rebalance(self, current_weights: np.ndarray) -> bool:
        """Check if rebalancing is needed."""
        max_drift = np.max(np.abs(current_weights - self.target_weights))
        return max_drift > self.threshold

    def rebalance(
        self,
        current_weights: np.ndarray,
        wealth: float
    ) -> Tuple[np.ndarray, float]:
        """
        Rebalance to target weights if threshold exceeded.

        Returns:
            Tuple of (new_weights, transaction_cost)
        """
        if self.should_rebalance(current_weights):
            # Rebalance to target
            turnover = np.sum(np.abs(self.target_weights - current_weights))
            cost = self.transaction_cost * turnover * wealth
            return self.target_weights, cost
        else:
            # No rebalancing
            return current_weights, 0.0


if __name__ == "__main__":
    print("Testing Multi-Period Optimization...")

    # Setup
    n_assets = 3
    n_periods = 6

    # Expected returns and covariance (constant over time for simplicity)
    expected_returns = np.array([0.08, 0.10, 0.06]) / 12  # Monthly
    volatilities = np.array([0.15, 0.20, 0.12]) / np.sqrt(12)

    correlation = np.array([
        [1.0, 0.5, 0.3],
        [0.5, 1.0, 0.4],
        [0.3, 0.4, 1.0]
    ])

    covariance = np.outer(volatilities, volatilities) * correlation

    # Test 1: Deterministic multi-period
    print("\n1. Deterministic Multi-Period Optimization")
    print("-" * 50)

    config = MultiPeriodConfig(
        n_periods=n_periods,
        risk_aversion=2.5,
        transaction_cost=0.002
    )

    optimizer = MultiPeriodOptimizer(config)

    # Create return path (same for all periods)
    returns_path = np.tile(expected_returns, (n_periods, 1))
    cov_path = np.tile(covariance, (n_periods, 1, 1))

    result = optimizer.deterministic_multi_period(
        returns_path,
        cov_path,
        initial_wealth=100.0
    )

    print(f"\nFinal Wealth: ${result['final_wealth']:.2f}")
    print(f"Total Return: {result['total_return']*100:.2f}%")

    print(f"\nWeights Trajectory:")
    for t, weights in enumerate(result['weights_trajectory'][:5]):
        print(f"  Period {t}: {weights.round(3)}")

    print(f"\nWealth Trajectory:")
    for t, wealth in enumerate(result['wealth_trajectory'][:5]):
        print(f"  Period {t}: ${wealth:.2f}")

    # Test 2: Scenario-based optimization
    print("\n2. Scenario-Based Optimization")
    print("-" * 50)

    scenario_tree = optimizer.generate_scenarios(
        expected_returns,
        covariance,
        n_scenarios=10,
        method='monte_carlo'
    )

    print(f"\nScenario Tree:")
    print(f"  Scenarios: {scenario_tree.n_scenarios}")
    print(f"  Periods: {scenario_tree.n_periods}")
    print(f"  Assets: {scenario_tree.n_assets}")

    print(f"\nSample Scenario Returns (Scenario 0, Period 0):")
    print(scenario_tree.returns[0, 0, :].round(4))

    # Test 3: Threshold rebalancing
    print("\n3. Threshold Rebalancing Policy")
    print("-" * 50)

    target = np.array([0.4, 0.4, 0.2])
    rebalancer = ThresholdRebalancingPolicy(
        target_weights=target,
        threshold=0.05,
        transaction_cost=0.002
    )

    # Simulate drift
    current = np.array([0.45, 0.38, 0.17])

    should_rebalance = rebalancer.should_rebalance(current)
    print(f"\nCurrent Weights: {current}")
    print(f"Target Weights: {target}")
    print(f"Should Rebalance: {should_rebalance}")

    if should_rebalance:
        new_weights, cost = rebalancer.rebalance(current, wealth=100.0)
        print(f"New Weights: {new_weights}")
        print(f"Transaction Cost: ${cost:.2f}")

    print("\n✅ Multi-period optimization implementation complete!")
