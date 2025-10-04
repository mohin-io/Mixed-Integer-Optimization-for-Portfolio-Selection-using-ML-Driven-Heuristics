"""
Genetic Algorithm for Mixed-Integer-Optimization-for-Portfolio-Selection-using-ML-Driven-Heuristics

Evolutionary approach to finding near-optimal portfolios:
1. Initialize population of random portfolios
2. Evaluate fitness (Sharpe ratio - costs)
3. Selection (tournament selection)
4. Crossover (blend weights from parents)
5. Mutation (randomly perturb weights)
6. Iterate for multiple generations
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class GAConfig:
    """Configuration for genetic algorithm."""
    population_size: int = 100
    generations: int = 50
    tournament_size: int = 5
    crossover_rate: float = 0.8
    mutation_rate: float = 0.2
    mutation_strength: float = 0.1
    max_assets: Optional[int] = None
    min_weight: float = 0.05
    elite_size: int = 5


class GeneticOptimizer:
    """
    Genetic Algorithm for portfolio optimization.
    """

    def __init__(self, config: Optional[GAConfig] = None):
        """
        Initialize genetic optimizer.

        Args:
            config: GA configuration
        """
        self.config = config or GAConfig()
        self.best_solution = None
        self.fitness_history = []

    def optimize(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        max_assets: Optional[int] = None
    ) -> pd.Series:
        """
        Run genetic algorithm to find optimal portfolio.

        Args:
            expected_returns: Expected returns
            covariance_matrix: Covariance matrix
            max_assets: Maximum number of assets (overrides config if provided)

        Returns:
            Best portfolio weights found
        """
        if max_assets is not None:
            self.config.max_assets = max_assets

        n_assets = len(expected_returns)
        assets = expected_returns.index.tolist()

        # Initialize population
        population = self._initialize_population(n_assets)

        # Evolution loop
        for generation in range(self.config.generations):
            # Evaluate fitness
            fitness_scores = [
                self._evaluate_fitness(individual, expected_returns, covariance_matrix)
                for individual in population
            ]

            # Track best solution
            best_idx = np.argmax(fitness_scores)
            best_fitness = fitness_scores[best_idx]
            self.fitness_history.append(best_fitness)

            if self.best_solution is None or best_fitness > self.best_solution['fitness']:
                self.best_solution = {
                    'weights': pd.Series(population[best_idx], index=assets),
                    'fitness': best_fitness,
                    'generation': generation
                }

            # Selection and reproduction
            new_population = []

            # Elitism: keep best solutions
            elite_indices = np.argsort(fitness_scores)[-self.config.elite_size:]
            for idx in elite_indices:
                new_population.append(population[idx].copy())

            # Generate rest of population
            while len(new_population) < self.config.population_size:
                # Tournament selection
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)

                # Crossover
                if np.random.random() < self.config.crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()

                # Mutation
                if np.random.random() < self.config.mutation_rate:
                    child1 = self._mutate(child1)
                if np.random.random() < self.config.mutation_rate:
                    child2 = self._mutate(child2)

                # Repair (ensure valid portfolio)
                child1 = self._repair_portfolio(child1)
                child2 = self._repair_portfolio(child2)

                new_population.extend([child1, child2])

            population = new_population[:self.config.population_size]

            if (generation + 1) % 10 == 0:
                print(f"Generation {generation+1}/{self.config.generations}, "
                      f"Best Fitness: {best_fitness:.4f}")

        return self.best_solution['weights']

    def _initialize_population(self, n_assets: int) -> List[np.ndarray]:
        """
        Initialize random population of portfolios.

        Args:
            n_assets: Number of assets

        Returns:
            List of weight vectors
        """
        population = []

        for _ in range(self.config.population_size):
            # Determine number of assets to include
            if self.config.max_assets:
                k = np.random.randint(1, self.config.max_assets + 1)
            else:
                k = np.random.randint(1, n_assets + 1)

            # Random selection of assets
            selected = np.random.choice(n_assets, size=k, replace=False)

            # Random weights
            weights = np.zeros(n_assets)
            random_weights = np.random.dirichlet(np.ones(k))
            weights[selected] = random_weights

            # Ensure minimum weight
            weights[weights > 0] = np.maximum(weights[weights > 0], self.config.min_weight)
            weights = weights / weights.sum()  # Renormalize

            population.append(weights)

        return population

    def _evaluate_fitness(
        self,
        weights: np.ndarray,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame
    ) -> float:
        """
        Evaluate fitness of a portfolio (Sharpe ratio).

        Args:
            weights: Portfolio weights
            expected_returns: Expected returns
            covariance_matrix: Covariance matrix

        Returns:
            Fitness score
        """
        portfolio_return = np.dot(weights, expected_returns.values)
        portfolio_variance = weights @ covariance_matrix.values @ weights
        portfolio_volatility = np.sqrt(portfolio_variance)

        if portfolio_volatility > 1e-8:
            sharpe_ratio = portfolio_return / portfolio_volatility
        else:
            sharpe_ratio = 0.0

        return sharpe_ratio

    def _tournament_selection(
        self,
        population: List[np.ndarray],
        fitness_scores: List[float]
    ) -> np.ndarray:
        """
        Select parent using tournament selection.

        Args:
            population: Current population
            fitness_scores: Fitness of each individual

        Returns:
            Selected parent
        """
        tournament_indices = np.random.choice(
            len(population),
            size=self.config.tournament_size,
            replace=False
        )

        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]

        return population[winner_idx].copy()

    def _crossover(
        self,
        parent1: np.ndarray,
        parent2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Blend crossover: children are weighted averages of parents.

        Args:
            parent1: First parent
            parent2: Second parent

        Returns:
            Two children
        """
        alpha = np.random.random()

        child1 = alpha * parent1 + (1 - alpha) * parent2
        child2 = (1 - alpha) * parent1 + alpha * parent2

        return child1, child2

    def _mutate(self, individual: np.ndarray) -> np.ndarray:
        """
        Mutate individual by perturbing weights.

        Args:
            individual: Weight vector

        Returns:
            Mutated weights
        """
        mutation = np.random.normal(
            0,
            self.config.mutation_strength,
            size=len(individual)
        )

        mutated = individual + mutation
        mutated = np.maximum(mutated, 0)  # No negative weights

        return mutated

    def _repair_portfolio(self, weights: np.ndarray) -> np.ndarray:
        """
        Repair portfolio to satisfy constraints.

        Args:
            weights: Potentially invalid weights

        Returns:
            Valid portfolio weights
        """
        # Remove very small weights
        weights[weights < self.config.min_weight] = 0

        # Enforce cardinality constraint
        if self.config.max_assets and np.sum(weights > 0) > self.config.max_assets:
            # Keep only top max_assets
            top_indices = np.argsort(weights)[-self.config.max_assets:]
            new_weights = np.zeros_like(weights)
            new_weights[top_indices] = weights[top_indices]
            weights = new_weights

        # Normalize
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            # Fallback: equal weight
            weights = np.ones_like(weights) / len(weights)

        return weights


if __name__ == "__main__":
    # Example usage
    from src.data.loader import AssetDataLoader
    from src.forecasting.returns_forecast import ReturnsForecast
    from src.forecasting.covariance import CovarianceEstimator

    # Load data
    loader = AssetDataLoader()
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'JPM', 'V', 'WMT']
    prices = loader.fetch_prices(tickers, '2020-01-01', '2023-12-31')
    returns = loader.compute_returns(prices)

    # Forecast inputs
    returns_forecaster = ReturnsForecast(method='historical')
    returns_forecaster.fit(returns)
    expected_returns = returns_forecaster.predict()

    cov_estimator = CovarianceEstimator(method='ledoit_wolf')
    cov_matrix = cov_estimator.estimate(returns)

    print("=== Genetic Algorithm Optimization ===\n")

    # Run GA
    config = GAConfig(
        population_size=100,
        generations=30,
        max_assets=5
    )

    ga_optimizer = GeneticOptimizer(config=config)
    ga_weights = ga_optimizer.optimize(expected_returns, cov_matrix)

    print(f"\nBest Solution Found:")
    print(f"Generation: {ga_optimizer.best_solution['generation']}")
    print(f"Fitness (Sharpe): {ga_optimizer.best_solution['fitness']:.4f}")
    print(f"\nWeights:")
    print(ga_weights[ga_weights > 1e-6].sort_values(ascending=False))
