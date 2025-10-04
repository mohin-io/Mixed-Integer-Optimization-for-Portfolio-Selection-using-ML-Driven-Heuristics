"""
ML-Based Constraint Prediction for Portfolio Optimization

Predicts which constraints will be active/binding to guide optimization:
1. Cardinality constraints (which assets to select)
2. Upper/lower bound constraints (which limits will bind)
3. Budget constraint utilization

Uses historical optimization solutions to train predictive models.

Benefits:
- Reduces search space for heuristic algorithms
- Warm-starts optimization solvers
- Improves convergence speed
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings


@dataclass
class ConstraintPredictorConfig:
    """Configuration for constraint predictor."""
    model_type: str = 'random_forest'  # 'random_forest' or 'gradient_boosting'
    n_estimators: int = 100
    max_depth: int = 10
    min_samples_split: int = 5
    random_state: int = 42
    test_size: float = 0.2


class ConstraintPredictor:
    """
    ML model to predict which constraints will be active.

    Features:
    - Asset-level: Sharpe ratio, volatility, correlation with portfolio
    - Market-level: Market volatility, dispersion, concentration
    - Historical: Past weight, weight changes, selection frequency

    Target:
    - Binary: Will asset be selected? (cardinality)
    - Continuous: Predicted weight level
    """

    def __init__(self, config: Optional[ConstraintPredictorConfig] = None):
        """
        Initialize constraint predictor.

        Args:
            config: Predictor configuration
        """
        self.config = config or ConstraintPredictorConfig()

        # Initialize model
        if self.config.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                min_samples_split=self.config.min_samples_split,
                random_state=self.config.random_state,
                n_jobs=-1
            )
        elif self.config.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                min_samples_split=self.config.min_samples_split,
                random_state=self.config.random_state
            )
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")

        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_fitted = False

    def train(
        self,
        historical_data: List[Dict],
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Train predictor on historical optimization results.

        Args:
            historical_data: List of dicts with keys:
                - 'returns': pd.DataFrame of returns
                - 'weights': pd.Series of optimal weights
                - 'expected_returns': pd.Series
                - 'cov_matrix': pd.DataFrame
            verbose: Print training progress

        Returns:
            Training metrics (accuracy, precision, recall)
        """
        if len(historical_data) < 10:
            raise ValueError("Need at least 10 historical samples for training")

        # Extract features and labels
        X, y, asset_names = self._extract_features_labels(historical_data)

        if len(X) == 0:
            raise ValueError("No features extracted from historical data")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y if len(np.unique(y)) > 1 else None
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model
        if verbose:
            print(f"\n{'='*60}")
            print("Training Constraint Predictor")
            print(f"{'='*60}")
            print(f"Model: {self.config.model_type}")
            print(f"Training samples: {len(X_train)}")
            print(f"Test samples: {len(X_test)}")
            print(f"Features: {X_train.shape[1]}")
            print(f"Positive class ratio: {y_train.mean()*100:.1f}%")
            print(f"{'='*60}\n")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(X_train_scaled, y_train)

        self.is_fitted = True

        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)

        # Predictions for precision/recall
        y_pred = self.model.predict(X_test_scaled)

        # Calculate metrics
        from sklearn.metrics import precision_score, recall_score, f1_score

        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        metrics = {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

        if verbose:
            print("Training Results:")
            print(f"  Train Accuracy: {train_score*100:.2f}%")
            print(f"  Test Accuracy:  {test_score*100:.2f}%")
            print(f"  Precision:      {precision*100:.2f}%")
            print(f"  Recall:         {recall*100:.2f}%")
            print(f"  F1 Score:       {f1:.3f}")
            print(f"{'='*60}\n")

            # Feature importance
            if hasattr(self.model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': self.model.feature_importances_
                }).sort_values('importance', ascending=False)

                print("Top 10 Important Features:")
                print(importance_df.head(10).to_string(index=False))
                print()

        return metrics

    def predict_constraints(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        max_assets: Optional[int] = None
    ) -> Dict[str, any]:
        """
        Predict which assets should be selected.

        Args:
            expected_returns: Expected returns
            cov_matrix: Covariance matrix
            max_assets: Maximum number of assets to select

        Returns:
            Dict with:
                - 'selected_assets': List of asset names
                - 'probabilities': Selection probabilities
                - 'weights_hint': Suggested initial weights
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before prediction")

        # Extract features for current market state
        features = self._compute_features(expected_returns, cov_matrix)

        # Scale features
        features_scaled = self.scaler.transform(features)

        # Predict probabilities
        probabilities = self.model.predict_proba(features_scaled)[:, 1]

        # Create results DataFrame
        results_df = pd.DataFrame({
            'asset': expected_returns.index,
            'probability': probabilities
        }).sort_values('probability', ascending=False)

        # Select top assets
        if max_assets:
            selected = results_df.head(max_assets)['asset'].tolist()
        else:
            # Use probability threshold (e.g., > 0.5)
            selected = results_df[results_df['probability'] > 0.5]['asset'].tolist()

        # Generate weight hints (proportional to probability)
        selected_probs = results_df[results_df['asset'].isin(selected)]
        weight_hints = selected_probs.set_index('asset')['probability']
        weight_hints = weight_hints / weight_hints.sum()  # Normalize

        return {
            'selected_assets': selected,
            'probabilities': results_df.set_index('asset')['probability'],
            'weights_hint': weight_hints
        }

    def _extract_features_labels(
        self,
        historical_data: List[Dict]
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Extract features and labels from historical data."""
        all_features = []
        all_labels = []
        all_asset_names = []

        for sample in historical_data:
            expected_returns = sample.get('expected_returns')
            cov_matrix = sample.get('cov_matrix')
            weights = sample.get('weights')

            if expected_returns is None or cov_matrix is None or weights is None:
                continue

            # Compute features
            features = self._compute_features(expected_returns, cov_matrix)

            # Labels: was asset selected? (weight > threshold)
            labels = (weights.values > 0.01).astype(int)

            all_features.append(features)
            all_labels.append(labels)

            if len(all_asset_names) == 0:
                all_asset_names = expected_returns.index.tolist()

        if len(all_features) == 0:
            return np.array([]), np.array([]), []

        # Stack all samples
        X = np.vstack(all_features)
        y = np.concatenate(all_labels)

        return X, y, all_asset_names

    def _compute_features(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame
    ) -> np.ndarray:
        """
        Compute asset-level features for prediction.

        Args:
            expected_returns: Expected returns
            cov_matrix: Covariance matrix

        Returns:
            Feature matrix (n_assets x n_features)
        """
        n_assets = len(expected_returns)

        # Asset-level features
        features_list = []

        for i, asset in enumerate(expected_returns.index):
            asset_features = {}

            # 1. Expected return
            asset_features['expected_return'] = expected_returns.iloc[i]

            # 2. Volatility
            asset_features['volatility'] = np.sqrt(cov_matrix.iloc[i, i])

            # 3. Sharpe ratio (individual)
            if asset_features['volatility'] > 0:
                asset_features['sharpe_ratio'] = (
                    asset_features['expected_return'] / asset_features['volatility']
                )
            else:
                asset_features['sharpe_ratio'] = 0

            # 4. Average correlation with other assets
            correlations = cov_matrix.iloc[i, :] / (
                np.sqrt(cov_matrix.iloc[i, i]) *
                np.sqrt(np.diag(cov_matrix))
            )
            asset_features['avg_correlation'] = correlations.mean()

            # 5. Max correlation with other assets
            asset_features['max_correlation'] = correlations.drop(asset).max()

            # 6. Min correlation with other assets
            asset_features['min_correlation'] = correlations.drop(asset).min()

            # Market-level features (same for all assets in this sample)

            # 7. Market dispersion (std of returns)
            asset_features['market_dispersion'] = expected_returns.std()

            # 8. Market average return
            asset_features['market_avg_return'] = expected_returns.mean()

            # 9. Market average volatility
            asset_features['market_avg_vol'] = np.sqrt(np.diag(cov_matrix)).mean()

            # 10. Return rank (percentile)
            asset_features['return_rank'] = (
                (expected_returns < expected_returns.iloc[i]).sum() / n_assets
            )

            # 11. Volatility rank
            vols = np.sqrt(np.diag(cov_matrix))
            asset_features['vol_rank'] = (vols < vols[i]).sum() / n_assets

            features_list.append(list(asset_features.values()))

        # Store feature names (only once)
        if len(self.feature_names) == 0:
            self.feature_names = list(asset_features.keys())

        return np.array(features_list)


def generate_synthetic_training_data(n_samples: int = 100, n_assets: int = 10) -> List[Dict]:
    """
    Generate synthetic training data for demonstration.

    Args:
        n_samples: Number of samples
        n_assets: Number of assets per sample

    Returns:
        List of training samples
    """
    training_data = []

    for _ in range(n_samples):
        # Random returns
        expected_returns = pd.Series(
            np.random.normal(0.08, 0.15, n_assets),
            index=[f'Asset_{i}' for i in range(n_assets)]
        )

        # Random covariance (ensure positive semi-definite)
        A = np.random.randn(n_assets, n_assets)
        cov_matrix = pd.DataFrame(
            A @ A.T / n_assets,
            index=expected_returns.index,
            columns=expected_returns.index
        )

        # Random optimal weights (sparse)
        k = np.random.randint(3, min(7, n_assets))
        selected = np.random.choice(n_assets, k, replace=False)
        weights = pd.Series(0.0, index=expected_returns.index)
        weights.iloc[selected] = np.random.dirichlet(np.ones(k))

        training_data.append({
            'expected_returns': expected_returns,
            'cov_matrix': cov_matrix,
            'weights': weights
        })

    return training_data


if __name__ == "__main__":
    print("Constraint Predictor Demo\n")

    # Generate synthetic training data
    print("Generating synthetic training data...")
    training_data = generate_synthetic_training_data(n_samples=200, n_assets=10)

    # Initialize and train predictor
    config = ConstraintPredictorConfig(
        model_type='random_forest',
        n_estimators=100,
        max_depth=10
    )

    predictor = ConstraintPredictor(config=config)

    # Train
    metrics = predictor.train(training_data, verbose=True)

    # Make prediction on new data
    print("\nMaking prediction on new market state...")

    test_returns = pd.Series(
        np.random.normal(0.10, 0.12, 10),
        index=[f'Asset_{i}' for i in range(10)]
    )

    A = np.random.randn(10, 10)
    test_cov = pd.DataFrame(
        A @ A.T / 10,
        index=test_returns.index,
        columns=test_returns.index
    )

    predictions = predictor.predict_constraints(
        expected_returns=test_returns,
        cov_matrix=test_cov,
        max_assets=5
    )

    print(f"\nPredicted Selected Assets: {predictions['selected_assets']}")
    print(f"\nSelection Probabilities:")
    print(predictions['probabilities'].sort_values(ascending=False))
    print(f"\nWeight Hints:")
    print(predictions['weights_hint'].sort_values(ascending=False))

    print("\nConstraint Predictor demo completed successfully!")
