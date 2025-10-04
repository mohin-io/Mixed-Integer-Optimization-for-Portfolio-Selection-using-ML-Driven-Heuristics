"""
Simulated Annealing for Mixed-Integer-Optimization-for-Portfolio-Selection-using-ML-Driven-Heuristics

Probabilistic optimization technique for escaping local optima:
1. Start with random solution
2. Iteratively propose neighbor solutions
3. Accept improvements, sometimes accept worse solutions (with probability)
4. Gradually reduce temperature (acceptance probability)
5. Converge to near-optimal solution

Advantages:
- Escapes local optima in non-convex landscapes
- Handles transaction costs and discrete constraints
- Simple to implement and tune
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Callable
from dataclasses import dataclass
import warnings


@dataclass
class SAConfig:
    """Configuration for simulated annealing."""
    initial_temp: float = 1000.0  # Initial temperature
    final_temp: float = 0.01  # Final temperature
    cooling_rate: float = 0.95  # Cooling rate (exponential decay)
    iterations_per_temp: int = 100  # Iterations at each temperature
    max_iterations: int = 10000  # Maximum total iterations
    min_weight: float = 0.01  # Minimum weight threshold
    max_assets: Optional[int] = None  # Maximum number of assets
    neighbor_std: float = 0.1  # Standard deviation for neighbor generation
    random_seed: Optional[int] = 42


class SimulatedAnnealingOptimizer:
    """
    Simulated Annealing optimizer for portfolio selection.

    Minimizes: -Sharpe ratio (or custom objective)
    Constraints: Budget, cardinality, min/max weights
    """

    def __init__(self, config: Optional[SAConfig] = None):
        """
        Initialize simulated annealing optimizer.

        Args:
            config: SA configuration
        """
        self.config = config or SAConfig()

        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)

        self.best_solution = None
        self.best_energy = float('inf')
        self.energy_history = []
        self.temperature_history = []
        self.acceptance_history = []
        self.current_iteration = 0

    def optimize(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        objective: str = 'sharpe',
        risk_aversion: float = 2.5
    ) -> pd.Series:
        """
        Run simulated annealing optimization.

        Args:
            expected_returns: Expected returns
            covariance_matrix: Covariance matrix
            objective: Objective function ('sharpe', 'return', 'variance', 'utility')
            risk_aversion: Risk aversion parameter (for utility objective)

        Returns:
            Optimal portfolio weights
        """
        n_assets = len(expected_returns)
        assets = expected_returns.index.tolist()

        # Initialize with random solution
        current_solution = self._initialize_solution(n_assets)
        current_energy = self._evaluate_energy(
            current_solution,
            expected_returns,
            covariance_matrix,
            objective,
            risk_aversion
        )

        # Track best solution
        self.best_solution = current_solution.copy()
        self.best_energy = current_energy

        # Initialize temperature
        temperature = self.config.initial_temp

        # Simulated annealing loop
        iteration = 0
        n_accepted = 0

        print(f"\n{'='*60}")
        print("Simulated Annealing Optimization")
        print(f"{'='*60}")
        print(f"Objective: {objective}")
        print(f"Assets: {n_assets}")
        print(f"Initial temperature: {self.config.initial_temp}")
        print(f"Cooling rate: {self.config.cooling_rate}")
        print(f"{'='*60}\n")

        while temperature > self.config.final_temp and iteration < self.config.max_iterations:
            temp_accepted = 0

            for _ in range(self.config.iterations_per_temp):
                # Generate neighbor solution
                neighbor = self._generate_neighbor(current_solution)

                # Evaluate neighbor
                neighbor_energy = self._evaluate_energy(
                    neighbor,
                    expected_returns,
                    covariance_matrix,
                    objective,
                    risk_aversion
                )

                # Acceptance criterion
                delta_energy = neighbor_energy - current_energy

                if delta_energy < 0:
                    # Better solution: always accept
                    accept = True
                else:
                    # Worse solution: accept with probability
                    accept_prob = np.exp(-delta_energy / temperature)
                    accept = np.random.random() < accept_prob

                if accept:
                    current_solution = neighbor
                    current_energy = neighbor_energy
                    temp_accepted += 1
                    n_accepted += 1

                    # Update best solution
                    if current_energy < self.best_energy:
                        self.best_solution = current_solution.copy()
                        self.best_energy = current_energy

                # Track history
                self.energy_history.append(current_energy)
                self.temperature_history.append(temperature)
                self.acceptance_history.append(1 if accept else 0)

                iteration += 1

                if iteration >= self.config.max_iterations:
                    break

            # Cool down
            temperature *= self.config.cooling_rate

            # Progress update
            if iteration % 500 == 0 or temperature <= self.config.final_temp:
                acceptance_rate = temp_accepted / self.config.iterations_per_temp
                print(f"Iter {iteration:5d} | Temp: {temperature:8.2f} | "
                      f"Energy: {current_energy:8.4f} | Best: {self.best_energy:8.4f} | "
                      f"Accept: {acceptance_rate*100:5.1f}%")

        self.current_iteration = iteration

        # Convert best solution to Series
        weights = pd.Series(self.best_solution, index=assets)

        # Normalize
        if weights.sum() > 0:
            weights = weights / weights.sum()

        print(f"\n{'='*60}")
        print("Optimization Complete!")
        print(f"{'='*60}")
        print(f"Total iterations: {iteration}")
        print(f"Acceptance rate: {n_accepted/iteration*100:.2f}%")
        print(f"Best energy: {self.best_energy:.4f}")
        print(f"Active assets: {(weights > 0.01).sum()}")
        print(f"{'='*60}\n")

        return weights

    def _initialize_solution(self, n_assets: int) -> np.ndarray:
        """
        Initialize random portfolio solution.

        Args:
            n_assets: Number of assets

        Returns:
            Random weight vector
        """
        # Determine number of assets to include
        if self.config.max_assets:
            k = np.random.randint(1, min(self.config.max_assets, n_assets) + 1)
        else:
            k = np.random.randint(1, n_assets + 1)

        # Random selection
        selected = np.random.choice(n_assets, size=k, replace=False)

        # Random weights using Dirichlet distribution
        weights = np.zeros(n_assets)
        random_weights = np.random.dirichlet(np.ones(k))
        weights[selected] = random_weights

        return weights

    def _generate_neighbor(self, current: np.ndarray) -> np.ndarray:
        """
        Generate neighbor solution.

        Strategies:
        1. Perturb existing weights
        2. Add new asset
        3. Remove asset
        4. Swap assets

        Args:
            current: Current weight vector

        Returns:
            Neighbor weight vector
        """
        neighbor = current.copy()
        n_assets = len(current)
        active_assets = current > self.config.min_weight

        # Choose neighbor generation strategy
        strategy = np.random.choice(['perturb', 'add', 'remove', 'swap'], p=[0.5, 0.2, 0.2, 0.1])

        if strategy == 'perturb':
            # Perturb existing weights
            active_indices = np.where(active_assets)[0]

            if len(active_indices) > 0:
                # Add random noise
                noise = np.random.normal(0, self.config.neighbor_std, size=len(active_indices))
                neighbor[active_indices] += noise
                neighbor = np.maximum(neighbor, 0)  # No negative weights

        elif strategy == 'add':
            # Add a new asset
            inactive_assets = ~active_assets

            if inactive_assets.sum() > 0:
                # Check cardinality constraint
                current_count = active_assets.sum()

                if self.config.max_assets is None or current_count < self.config.max_assets:
                    # Select random inactive asset
                    inactive_indices = np.where(inactive_assets)[0]
                    new_asset = np.random.choice(inactive_indices)

                    # Allocate weight
                    new_weight = np.random.uniform(0.05, 0.2)
                    neighbor[new_asset] = new_weight

        elif strategy == 'remove':
            # Remove an asset
            active_indices = np.where(active_assets)[0]

            if len(active_indices) > 1:  # Keep at least 1 asset
                # Select random asset to remove
                remove_asset = np.random.choice(active_indices)
                neighbor[remove_asset] = 0

        elif strategy == 'swap':
            # Swap two assets
            active_indices = np.where(active_assets)[0]
            inactive_indices = np.where(~active_assets)[0]

            if len(active_indices) > 0 and len(inactive_indices) > 0:
                old_asset = np.random.choice(active_indices)
                new_asset = np.random.choice(inactive_indices)

                # Transfer weight
                neighbor[new_asset] = neighbor[old_asset]
                neighbor[old_asset] = 0

        # Repair solution
        neighbor = self._repair_solution(neighbor)

        return neighbor

    def _repair_solution(self, weights: np.ndarray) -> np.ndarray:
        """
        Repair solution to satisfy constraints.

        Args:
            weights: Potentially invalid weights

        Returns:
            Valid weight vector
        """
        # Remove tiny weights
        weights[weights < self.config.min_weight] = 0

        # Enforce cardinality constraint
        if self.config.max_assets and (weights > 0).sum() > self.config.max_assets:
            # Keep only top max_assets
            top_indices = np.argsort(weights)[-self.config.max_assets:]
            new_weights = np.zeros_like(weights)
            new_weights[top_indices] = weights[top_indices]
            weights = new_weights

        # Normalize to sum to 1
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            # Fallback: single random asset
            random_asset = np.random.randint(len(weights))
            weights = np.zeros_like(weights)
            weights[random_asset] = 1.0

        return weights

    def _evaluate_energy(
        self,
        weights: np.ndarray,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        objective: str,
        risk_aversion: float
    ) -> float:
        """
        Evaluate energy (objective function) for a solution.

        Lower energy = better solution

        Args:
            weights: Portfolio weights
            expected_returns: Expected returns
            covariance_matrix: Covariance matrix
            objective: Objective function type
            risk_aversion: Risk aversion parameter

        Returns:
            Energy value
        """
        # Portfolio return
        port_return = np.dot(weights, expected_returns.values)

        # Portfolio variance
        port_variance = weights @ covariance_matrix.values @ weights

        # Portfolio volatility
        port_vol = np.sqrt(port_variance) if port_variance > 0 else 1e-8

        # Calculate energy based on objective
        if objective == 'sharpe':
            # Minimize negative Sharpe ratio
            sharpe = port_return / port_vol if port_vol > 0 else -1e10
            energy = -sharpe

        elif objective == 'return':
            # Minimize negative return
            energy = -port_return

        elif objective == 'variance':
            # Minimize variance
            energy = port_variance

        elif objective == 'utility':
            # Minimize negative utility
            utility = port_return - (risk_aversion / 2) * port_variance
            energy = -utility

        else:
            raise ValueError(f"Unknown objective: {objective}")

        # Penalty for constraint violations (already handled in repair)
        return energy

    def get_convergence_data(self) -> pd.DataFrame:
        """
        Get convergence history for visualization.

        Returns:
            DataFrame with iteration, energy, temperature, acceptance
        """
        return pd.DataFrame({
            'iteration': range(len(self.energy_history)),
            'energy': self.energy_history,
            'temperature': self.temperature_history,
            'accepted': self.acceptance_history
        })


if __name__ == "__main__":
    # Example usage
    from ..data.loader import AssetDataLoader
    from ..forecasting.covariance import CovarianceEstimator

    print("Simulated Annealing Demo\n")

    # Load data
    loader = AssetDataLoader()
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'JPM', 'V', 'WMT']

    try:
        prices = loader.fetch_prices(tickers, '2020-01-01', '2023-12-31')
        returns = loader.compute_returns(prices)

        # Forecast parameters
        expected_returns = returns.mean() * 252
        cov_estimator = CovarianceEstimator(method='ledoit_wolf')
        cov_matrix = cov_estimator.estimate(returns) * 252

        # Configure SA
        config = SAConfig(
            initial_temp=1000,
            final_temp=0.1,
            cooling_rate=0.95,
            iterations_per_temp=50,
            max_iterations=5000,
            max_assets=5
        )

        # Run optimization
        optimizer = SimulatedAnnealingOptimizer(config=config)

        weights = optimizer.optimize(
            expected_returns=expected_returns,
            covariance_matrix=cov_matrix,
            objective='sharpe'
        )

        # Display results
        print("\nOptimal Portfolio Weights:")
        print(weights[weights > 0.01].sort_values(ascending=False))

        # Portfolio metrics
        port_return = np.dot(weights, expected_returns)
        port_vol = np.sqrt(weights @ cov_matrix @ weights)
        sharpe = port_return / port_vol

        print(f"\nPortfolio Metrics:")
        print(f"Expected Return: {port_return*100:.2f}%")
        print(f"Volatility: {port_vol*100:.2f}%")
        print(f"Sharpe Ratio: {sharpe:.3f}")
        print(f"Active Assets: {(weights > 0.01).sum()}")

        print("\nSimulated Annealing completed successfully!")

    except Exception as e:
        print(f"Error in demo: {str(e)}")
        import traceback
        traceback.print_exc()
