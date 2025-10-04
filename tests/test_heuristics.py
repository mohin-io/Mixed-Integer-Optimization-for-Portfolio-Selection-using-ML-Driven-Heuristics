"""
Unit Tests for ML-Driven Heuristics

Tests:
- Asset Clustering (K-Means, Hierarchical)
- Genetic Algorithm
- Simulated Annealing
- Constraint Predictor
- Convergence behavior
"""

import pytest
import pandas as pd
import numpy as np
from src.heuristics.clustering import AssetClusterer, compute_linkage_matrix
from src.heuristics.genetic_algorithm import GeneticOptimizer, GAConfig
from src.heuristics.simulated_annealing import SimulatedAnnealingOptimizer, SAConfig
from src.heuristics.constraint_predictor import ConstraintPredictor, ConstraintPredictorConfig


# Fixtures
@pytest.fixture
def sample_returns():
    """Generate sample returns for testing."""
    np.random.seed(42)
    n_periods = 252
    n_assets = 10

    # Generate returns with some correlation structure
    returns = np.random.multivariate_normal(
        mean=np.random.uniform(0.0005, 0.0015, n_assets),
        cov=np.eye(n_assets) * 0.0004 + 0.0001,
        size=n_periods
    )

    dates = pd.date_range('2023-01-01', periods=n_periods, freq='D')
    tickers = [f'Asset_{i}' for i in range(n_assets)]

    return pd.DataFrame(returns, index=dates, columns=tickers)


@pytest.fixture
def expected_returns_cov(sample_returns):
    """Generate expected returns and covariance matrix."""
    expected_returns = sample_returns.mean() * 252
    cov_matrix = sample_returns.cov() * 252

    return expected_returns, cov_matrix


class TestAssetClusterer:
    """Test suite for asset clustering."""

    def test_kmeans_clustering(self, sample_returns):
        """Test K-Means clustering."""
        clusterer = AssetClusterer(method='kmeans')
        clusterer.fit(sample_returns, n_clusters=3)

        assert clusterer.labels_ is not None, "Labels should be assigned"
        assert len(clusterer.labels_) == len(sample_returns.columns), "Label for each asset"
        assert len(np.unique(clusterer.labels_)) <= 3, "Should have at most 3 clusters"
        assert (clusterer.labels_ >= 0).all(), "Labels should be non-negative"

    def test_hierarchical_clustering(self, sample_returns):
        """Test hierarchical clustering."""
        clusterer = AssetClusterer(method='hierarchical')
        clusterer.fit(sample_returns, n_clusters=4)

        assert clusterer.labels_ is not None, "Labels should be assigned"
        assert len(clusterer.labels_) == len(sample_returns.columns)
        assert len(np.unique(clusterer.labels_)) <= 4

    def test_select_representatives(self, sample_returns):
        """Test representative asset selection."""
        clusterer = AssetClusterer(method='kmeans')
        clusterer.fit(sample_returns, n_clusters=3)

        selected = clusterer.select_representatives(sample_returns, n_per_cluster=2)

        assert isinstance(selected, list), "Should return list"
        assert len(selected) > 0, "Should select some assets"
        assert len(selected) <= 6, "Should select at most n_clusters * n_per_cluster assets"
        assert all(asset in sample_returns.columns for asset in selected), "Selected assets should be valid"

    def test_cluster_summary(self, sample_returns):
        """Test cluster summary statistics."""
        clusterer = AssetClusterer(method='kmeans')
        clusterer.fit(sample_returns, n_clusters=3)

        summary = clusterer.get_cluster_summary(sample_returns)

        assert isinstance(summary, pd.DataFrame), "Summary should be DataFrame"
        assert 'cluster_id' in summary.columns, "Should have cluster_id"
        assert 'n_assets' in summary.columns, "Should have asset count"
        assert len(summary) <= 3, "Should have summary for each cluster"

    def test_different_cluster_numbers(self, sample_returns):
        """Test clustering with different numbers of clusters."""
        for n_clusters in [2, 3, 5]:
            clusterer = AssetClusterer(method='kmeans')
            clusterer.fit(sample_returns, n_clusters=n_clusters)

            unique_labels = len(np.unique(clusterer.labels_))
            assert unique_labels <= n_clusters, f"Should have at most {n_clusters} clusters"

    def test_linkage_matrix_computation(self, sample_returns):
        """Test linkage matrix computation for dendrogram."""
        linkage_matrix = compute_linkage_matrix(sample_returns, method='ward')

        assert isinstance(linkage_matrix, np.ndarray), "Should return numpy array"
        assert linkage_matrix.shape[0] == len(sample_returns.columns) - 1, "Correct linkage shape"
        assert linkage_matrix.shape[1] == 4, "Linkage should have 4 columns"

    def test_reproducibility(self, sample_returns):
        """Test that clustering is reproducible with same random state."""
        clusterer1 = AssetClusterer(method='kmeans')
        clusterer2 = AssetClusterer(method='kmeans')

        clusterer1.fit(sample_returns, n_clusters=3)
        clusterer2.fit(sample_returns, n_clusters=3)

        # Should produce same result (K-means uses fixed random_state=42)
        assert np.array_equal(clusterer1.labels_, clusterer2.labels_), "Should be reproducible"


class TestGeneticAlgorithm:
    """Test suite for genetic algorithm."""

    def test_ga_optimization(self, expected_returns_cov):
        """Test basic GA optimization."""
        expected_returns, cov_matrix = expected_returns_cov

        config = GAConfig(
            population_size=20,
            generations=10,
            max_assets=5
        )

        optimizer = GeneticOptimizer(config=config)
        weights = optimizer.optimize(expected_returns, cov_matrix)

        assert isinstance(weights, pd.Series), "Should return Series"
        assert len(weights) == len(expected_returns), "Correct length"
        assert np.isclose(weights.sum(), 1.0, atol=1e-6), "Weights should sum to 1"
        assert (weights >= 0).all(), "Weights should be non-negative"

    def test_ga_cardinality_constraint(self, expected_returns_cov):
        """Test that GA respects cardinality constraint."""
        expected_returns, cov_matrix = expected_returns_cov

        max_assets = 4
        config = GAConfig(
            population_size=20,
            generations=10,
            max_assets=max_assets
        )

        optimizer = GeneticOptimizer(config=config)
        weights = optimizer.optimize(expected_returns, cov_matrix)

        n_active = (weights > 1e-6).sum()
        assert n_active <= max_assets, f"Should have at most {max_assets} assets"

    def test_ga_convergence(self, expected_returns_cov):
        """Test that GA fitness improves over generations."""
        expected_returns, cov_matrix = expected_returns_cov

        config = GAConfig(
            population_size=30,
            generations=20,
            max_assets=5
        )

        optimizer = GeneticOptimizer(config=config)
        weights = optimizer.optimize(expected_returns, cov_matrix)

        # Check fitness history
        assert len(optimizer.fitness_history) > 0, "Should track fitness history"

        # Fitness should generally improve (best fitness should be non-decreasing)
        # Allow for some noise but overall trend should be upward
        first_half_avg = np.mean(optimizer.fitness_history[:10])
        second_half_avg = np.mean(optimizer.fitness_history[-10:])

        assert second_half_avg >= first_half_avg * 0.9, "Fitness should generally improve"

    def test_ga_tournament_selection(self, expected_returns_cov):
        """Test GA with different tournament sizes."""
        expected_returns, cov_matrix = expected_returns_cov

        for tournament_size in [3, 5, 7]:
            config = GAConfig(
                population_size=20,
                generations=5,
                tournament_size=tournament_size
            )

            optimizer = GeneticOptimizer(config=config)
            weights = optimizer.optimize(expected_returns, cov_matrix)

            assert weights is not None, f"Should work with tournament_size={tournament_size}"

    def test_ga_best_solution_tracking(self, expected_returns_cov):
        """Test that GA tracks best solution."""
        expected_returns, cov_matrix = expected_returns_cov

        config = GAConfig(population_size=20, generations=10)

        optimizer = GeneticOptimizer(config=config)
        optimizer.optimize(expected_returns, cov_matrix)

        assert optimizer.best_solution is not None, "Should track best solution"
        assert 'weights' in optimizer.best_solution, "Best solution should have weights"
        assert 'fitness' in optimizer.best_solution, "Best solution should have fitness"


class TestSimulatedAnnealing:
    """Test suite for simulated annealing."""

    def test_sa_optimization(self, expected_returns_cov):
        """Test basic SA optimization."""
        expected_returns, cov_matrix = expected_returns_cov

        config = SAConfig(
            initial_temp=100,
            final_temp=0.1,
            cooling_rate=0.9,
            iterations_per_temp=20,
            max_iterations=500,
            max_assets=5,
            random_seed=42
        )

        optimizer = SimulatedAnnealingOptimizer(config=config)
        weights = optimizer.optimize(expected_returns, cov_matrix, objective='sharpe')

        assert isinstance(weights, pd.Series), "Should return Series"
        assert np.isclose(weights.sum(), 1.0, atol=1e-6), "Weights should sum to 1"
        assert (weights >= 0).all(), "Weights should be non-negative"

    def test_sa_cardinality_constraint(self, expected_returns_cov):
        """Test SA cardinality constraint."""
        expected_returns, cov_matrix = expected_returns_cov

        max_assets = 3
        config = SAConfig(
            initial_temp=100,
            final_temp=1,
            max_iterations=300,
            max_assets=max_assets,
            random_seed=42
        )

        optimizer = SimulatedAnnealingOptimizer(config=config)
        weights = optimizer.optimize(expected_returns, cov_matrix)

        n_active = (weights > 0.01).sum()
        assert n_active <= max_assets, f"Should have at most {max_assets} assets"

    def test_sa_temperature_cooling(self, expected_returns_cov):
        """Test that temperature decreases over iterations."""
        expected_returns, cov_matrix = expected_returns_cov

        config = SAConfig(
            initial_temp=1000,
            final_temp=1,
            cooling_rate=0.95,
            max_iterations=500,
            random_seed=42
        )

        optimizer = SimulatedAnnealingOptimizer(config=config)
        optimizer.optimize(expected_returns, cov_matrix)

        # Check temperature history
        temp_history = optimizer.temperature_history

        assert len(temp_history) > 0, "Should track temperature"
        assert temp_history[0] >= temp_history[-1], "Temperature should decrease"

    def test_sa_different_objectives(self, expected_returns_cov):
        """Test SA with different objective functions."""
        expected_returns, cov_matrix = expected_returns_cov

        config = SAConfig(
            initial_temp=100,
            final_temp=1,
            max_iterations=200,
            random_seed=42
        )

        objectives = ['sharpe', 'return', 'variance', 'utility']

        for obj in objectives:
            optimizer = SimulatedAnnealingOptimizer(config=config)
            weights = optimizer.optimize(expected_returns, cov_matrix, objective=obj)

            assert weights is not None, f"Should work with objective={obj}"
            assert np.isclose(weights.sum(), 1.0, atol=1e-6), f"Weights should sum to 1 for {obj}"

    def test_sa_reproducibility(self, expected_returns_cov):
        """Test SA reproducibility with same seed."""
        expected_returns, cov_matrix = expected_returns_cov

        config = SAConfig(
            initial_temp=100,
            final_temp=1,
            max_iterations=100,
            random_seed=123
        )

        optimizer1 = SimulatedAnnealingOptimizer(config=config)
        weights1 = optimizer1.optimize(expected_returns, cov_matrix)

        optimizer2 = SimulatedAnnealingOptimizer(config=config)
        weights2 = optimizer2.optimize(expected_returns, cov_matrix)

        assert np.allclose(weights1, weights2, atol=1e-6), "Should be reproducible with same seed"

    def test_sa_convergence_data(self, expected_returns_cov):
        """Test that SA tracks convergence data."""
        expected_returns, cov_matrix = expected_returns_cov

        config = SAConfig(
            initial_temp=100,
            final_temp=1,
            max_iterations=200,
            random_seed=42
        )

        optimizer = SimulatedAnnealingOptimizer(config=config)
        optimizer.optimize(expected_returns, cov_matrix)

        convergence_data = optimizer.get_convergence_data()

        assert isinstance(convergence_data, pd.DataFrame), "Should return DataFrame"
        assert 'iteration' in convergence_data.columns
        assert 'energy' in convergence_data.columns
        assert 'temperature' in convergence_data.columns
        assert len(convergence_data) > 0


class TestConstraintPredictor:
    """Test suite for constraint predictor."""

    def test_predictor_training(self):
        """Test training constraint predictor."""
        # Generate synthetic training data
        from src.heuristics.constraint_predictor import generate_synthetic_training_data

        training_data = generate_synthetic_training_data(n_samples=50, n_assets=10)

        config = ConstraintPredictorConfig(
            model_type='random_forest',
            n_estimators=50,
            random_state=42
        )

        predictor = ConstraintPredictor(config=config)
        metrics = predictor.train(training_data, verbose=False)

        assert 'train_accuracy' in metrics, "Should return train accuracy"
        assert 'test_accuracy' in metrics, "Should return test accuracy"
        assert metrics['train_accuracy'] > 0.5, "Should have reasonable accuracy"
        assert predictor.is_fitted, "Should be marked as fitted"

    def test_predictor_prediction(self):
        """Test making predictions."""
        from src.heuristics.constraint_predictor import generate_synthetic_training_data

        training_data = generate_synthetic_training_data(n_samples=50, n_assets=10)

        predictor = ConstraintPredictor()
        predictor.train(training_data, verbose=False)

        # Make prediction on new data
        test_returns = pd.Series(
            np.random.normal(0.10, 0.05, 10),
            index=[f'Asset_{i}' for i in range(10)]
        )

        test_cov = pd.DataFrame(
            np.eye(10) * 0.04,
            index=test_returns.index,
            columns=test_returns.index
        )

        predictions = predictor.predict_constraints(test_returns, test_cov, max_assets=5)

        assert 'selected_assets' in predictions, "Should return selected assets"
        assert 'probabilities' in predictions, "Should return probabilities"
        assert 'weights_hint' in predictions, "Should return weight hints"
        assert len(predictions['selected_assets']) <= 5, "Should respect max_assets"

    def test_predictor_requires_training(self):
        """Test that prediction requires training first."""
        predictor = ConstraintPredictor()

        test_returns = pd.Series([0.1] * 10, index=[f'Asset_{i}' for i in range(10)])
        test_cov = pd.DataFrame(np.eye(10) * 0.04, index=test_returns.index, columns=test_returns.index)

        with pytest.raises(RuntimeError):
            predictor.predict_constraints(test_returns, test_cov)

    def test_predictor_feature_extraction(self):
        """Test that features are extracted correctly."""
        from src.heuristics.constraint_predictor import generate_synthetic_training_data

        training_data = generate_synthetic_training_data(n_samples=20, n_assets=10)

        predictor = ConstraintPredictor()
        X, y, asset_names = predictor._extract_features_labels(training_data)

        assert X.shape[0] == 20 * 10, "Should have features for all samples x assets"
        assert y.shape[0] == 20 * 10, "Should have labels for all"
        assert len(asset_names) == 10, "Should track asset names"


class TestHeuristicsIntegration:
    """Integration tests for heuristics."""

    def test_clustering_with_optimization(self, sample_returns, expected_returns_cov):
        """Test using clustering to pre-select assets for optimization."""
        expected_returns, cov_matrix = expected_returns_cov

        # Cluster assets
        clusterer = AssetClusterer(method='kmeans')
        clusterer.fit(sample_returns, n_clusters=3)

        selected_assets = clusterer.select_representatives(sample_returns, n_per_cluster=2)

        # Optimize on selected assets
        subset_returns = expected_returns[selected_assets]
        subset_cov = cov_matrix.loc[selected_assets, selected_assets]

        config = GAConfig(population_size=10, generations=5)
        optimizer = GeneticOptimizer(config=config)
        weights = optimizer.optimize(subset_returns, subset_cov)

        assert len(weights) == len(selected_assets), "Weights for selected assets"
        assert np.isclose(weights.sum(), 1.0, atol=1e-6), "Weights should sum to 1"

    def test_heuristic_comparison(self, expected_returns_cov):
        """Compare GA and SA on same problem."""
        expected_returns, cov_matrix = expected_returns_cov

        # GA
        ga_config = GAConfig(
            population_size=20,
            generations=10,
            max_assets=5
        )
        ga_optimizer = GeneticOptimizer(config=ga_config)
        ga_weights = ga_optimizer.optimize(expected_returns, cov_matrix)

        # SA
        sa_config = SAConfig(
            initial_temp=100,
            final_temp=1,
            max_iterations=200,
            max_assets=5,
            random_seed=42
        )
        sa_optimizer = SimulatedAnnealingOptimizer(config=sa_config)
        sa_weights = sa_optimizer.optimize(expected_returns, cov_matrix, objective='sharpe')

        # Both should produce valid solutions
        assert np.isclose(ga_weights.sum(), 1.0, atol=1e-6), "GA weights valid"
        assert np.isclose(sa_weights.sum(), 1.0, atol=1e-6), "SA weights valid"

        # Both should respect cardinality
        assert (ga_weights > 0.01).sum() <= 5, "GA respects cardinality"
        assert (sa_weights > 0.01).sum() <= 5, "SA respects cardinality"


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, '-v'])
