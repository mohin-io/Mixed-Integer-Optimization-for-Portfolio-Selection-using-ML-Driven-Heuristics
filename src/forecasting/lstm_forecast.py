"""
LSTM (Long Short-Term Memory) Neural Network for Return Forecasting.

Implements deep learning models for time series prediction:
- Univariate LSTM for single asset prediction
- Multivariate LSTM for multiple assets
- Attention-enhanced LSTM
- Ensemble LSTM models
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, List
import warnings

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Attention, Input
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from sklearn.preprocessing import StandardScaler
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    warnings.warn("TensorFlow not available. LSTM forecasting will use fallback methods.")


class LSTMForecaster:
    """
    LSTM-based return forecaster for portfolio optimization.
    """

    def __init__(
        self,
        lookback_window: int = 60,
        forecast_horizon: int = 1,
        hidden_units: List[int] = [64, 32],
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 32,
        use_bidirectional: bool = False
    ):
        """
        Initialize LSTM forecaster.

        Args:
            lookback_window: Number of historical periods to use
            forecast_horizon: Number of periods to forecast ahead
            hidden_units: List of hidden units for each LSTM layer
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
            epochs: Maximum training epochs
            batch_size: Batch size for training
            use_bidirectional: Use bidirectional LSTM
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM forecasting. Install with: pip install tensorflow")

        self.lookback_window = lookback_window
        self.forecast_horizon = forecast_horizon
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.use_bidirectional = use_bidirectional

        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False

    def create_sequences(
        self,
        data: np.ndarray,
        lookback: int,
        forecast_horizon: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training.

        Args:
            data: Input time series data
            lookback: Number of past observations to use
            forecast_horizon: Number of steps to predict ahead

        Returns:
            Tuple of (X, y) where X is sequences and y is targets
        """
        X, y = [], []

        for i in range(len(data) - lookback - forecast_horizon + 1):
            X.append(data[i:(i + lookback)])
            y.append(data[i + lookback + forecast_horizon - 1])

        return np.array(X), np.array(y)

    def build_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """
        Build LSTM model architecture.

        Args:
            input_shape: Shape of input (lookback_window, n_features)

        Returns:
            Compiled Keras model
        """
        model = Sequential()

        # First LSTM layer
        if self.use_bidirectional:
            model.add(Bidirectional(
                LSTM(self.hidden_units[0], return_sequences=len(self.hidden_units) > 1),
                input_shape=input_shape
            ))
        else:
            model.add(LSTM(
                self.hidden_units[0],
                return_sequences=len(self.hidden_units) > 1,
                input_shape=input_shape
            ))

        model.add(Dropout(self.dropout_rate))

        # Additional LSTM layers
        for i, units in enumerate(self.hidden_units[1:]):
            is_last = (i == len(self.hidden_units) - 2)

            if self.use_bidirectional:
                model.add(Bidirectional(LSTM(units, return_sequences=not is_last)))
            else:
                model.add(LSTM(units, return_sequences=not is_last))

            model.add(Dropout(self.dropout_rate))

        # Output layer
        model.add(Dense(input_shape[1]))  # Predict all features

        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )

        return model

    def fit(
        self,
        returns: pd.DataFrame,
        validation_split: float = 0.2,
        verbose: int = 0
    ) -> dict:
        """
        Train LSTM model on historical returns.

        Args:
            returns: DataFrame of historical returns (time x assets)
            validation_split: Fraction of data to use for validation
            verbose: Verbosity level (0, 1, or 2)

        Returns:
            Training history dictionary
        """
        # Normalize data
        returns_scaled = self.scaler.fit_transform(returns.values)

        # Create sequences
        X, y = self.create_sequences(
            returns_scaled,
            self.lookback_window,
            self.forecast_horizon
        )

        # Build model
        input_shape = (self.lookback_window, returns.shape[1])
        self.model = self.build_model(input_shape)

        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]

        # Train model
        history = self.model.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose
        )

        self.is_fitted = True

        return history.history

    def predict(
        self,
        returns: pd.DataFrame,
        n_steps: int = 1
    ) -> pd.DataFrame:
        """
        Forecast future returns.

        Args:
            returns: Recent historical returns
            n_steps: Number of steps to forecast

        Returns:
            DataFrame of forecasted returns
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Use last lookback_window observations
        recent_data = returns.values[-self.lookback_window:]

        # Normalize
        recent_scaled = self.scaler.transform(recent_data)

        predictions = []

        # Multi-step prediction
        current_sequence = recent_scaled.copy()

        for _ in range(n_steps):
            # Predict next step
            X_pred = current_sequence.reshape(1, self.lookback_window, -1)
            y_pred = self.model.predict(X_pred, verbose=0)[0]

            predictions.append(y_pred)

            # Update sequence (rolling window)
            current_sequence = np.vstack([current_sequence[1:], y_pred])

        # Inverse transform predictions
        predictions = np.array(predictions)
        predictions = self.scaler.inverse_transform(predictions)

        # Create DataFrame
        forecast_df = pd.DataFrame(
            predictions,
            columns=returns.columns,
            index=pd.RangeIndex(1, n_steps + 1)
        )

        return forecast_df


class AttentionLSTMForecaster(LSTMForecaster):
    """
    LSTM with attention mechanism for better long-term dependencies.
    """

    def build_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """Build LSTM model with attention layer."""
        # Input layer
        inputs = Input(shape=input_shape)

        # LSTM layers
        x = inputs
        for i, units in enumerate(self.hidden_units):
            x = LSTM(units, return_sequences=True)(x)
            x = Dropout(self.dropout_rate)(x)

        # Attention layer
        attention = Attention()([x, x])

        # Global pooling
        x = tf.keras.layers.GlobalAveragePooling1D()(attention)

        # Output layer
        outputs = Dense(input_shape[1])(x)

        # Create model
        model = Model(inputs=inputs, outputs=outputs)

        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )

        return model


class EnsembleLSTMForecaster:
    """
    Ensemble of multiple LSTM models for robust predictions.
    """

    def __init__(
        self,
        n_models: int = 5,
        lookback_window: int = 60,
        **lstm_kwargs
    ):
        """
        Initialize ensemble forecaster.

        Args:
            n_models: Number of models in ensemble
            lookback_window: Lookback window for each model
            **lstm_kwargs: Additional arguments for LSTMForecaster
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM forecasting.")

        self.n_models = n_models
        self.lookback_window = lookback_window
        self.lstm_kwargs = lstm_kwargs

        self.models = []

    def fit(
        self,
        returns: pd.DataFrame,
        validation_split: float = 0.2,
        verbose: int = 0
    ):
        """Train ensemble of models."""
        for i in range(self.n_models):
            print(f"Training model {i+1}/{self.n_models}...")

            # Create model with different random seed
            tf.random.set_seed(i)

            model = LSTMForecaster(
                lookback_window=self.lookback_window,
                **self.lstm_kwargs
            )

            model.fit(returns, validation_split=validation_split, verbose=verbose)
            self.models.append(model)

    def predict(
        self,
        returns: pd.DataFrame,
        n_steps: int = 1
    ) -> pd.DataFrame:
        """Forecast using ensemble average."""
        predictions = []

        for model in self.models:
            pred = model.predict(returns, n_steps=n_steps)
            predictions.append(pred.values)

        # Average predictions
        ensemble_pred = np.mean(predictions, axis=0)

        # Create DataFrame
        forecast_df = pd.DataFrame(
            ensemble_pred,
            columns=returns.columns,
            index=pd.RangeIndex(1, n_steps + 1)
        )

        return forecast_df


class FallbackLSTMForecaster:
    """
    Fallback forecaster when TensorFlow is not available.
    Uses simple exponential weighted moving average.
    """

    def __init__(self, alpha: float = 0.1):
        """
        Initialize fallback forecaster.

        Args:
            alpha: Smoothing parameter for EWMA
        """
        self.alpha = alpha
        self.last_values = None

    def fit(self, returns: pd.DataFrame):
        """Store last values."""
        self.last_values = returns.ewm(alpha=self.alpha).mean().iloc[-1]

    def predict(
        self,
        returns: pd.DataFrame,
        n_steps: int = 1
    ) -> pd.DataFrame:
        """Predict using EWMA."""
        if self.last_values is None:
            self.fit(returns)

        # Simply repeat last EWMA value
        predictions = np.tile(self.last_values.values, (n_steps, 1))

        forecast_df = pd.DataFrame(
            predictions,
            columns=returns.columns,
            index=pd.RangeIndex(1, n_steps + 1)
        )

        return forecast_df


if __name__ == "__main__":
    print("Testing LSTM Forecasting...")

    if not TENSORFLOW_AVAILABLE:
        print("⚠️ TensorFlow not available. Using fallback method.")

        # Test fallback
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=252, freq='D')
        returns = pd.DataFrame(
            np.random.randn(252, 3) * 0.01,
            index=dates,
            columns=['Asset_A', 'Asset_B', 'Asset_C']
        )

        forecaster = FallbackLSTMForecaster(alpha=0.1)
        forecaster.fit(returns)
        predictions = forecaster.predict(returns, n_steps=5)

        print(f"\nFallback Predictions (5 steps):")
        print(predictions)

    else:
        print("✅ TensorFlow available. Testing LSTM models...")

        # Generate synthetic data
        np.random.seed(42)
        n_periods = 500
        n_assets = 3

        dates = pd.date_range('2020-01-01', periods=n_periods, freq='D')

        # Create synthetic returns with some autocorrelation
        returns_data = []
        for _ in range(n_assets):
            ar_param = 0.1
            noise = np.random.randn(n_periods) * 0.01
            returns_series = [noise[0]]

            for i in range(1, n_periods):
                returns_series.append(ar_param * returns_series[-1] + noise[i])

            returns_data.append(returns_series)

        returns = pd.DataFrame(
            np.array(returns_data).T,
            index=dates,
            columns=['Tech', 'Finance', 'Energy']
        )

        print(f"\nData shape: {returns.shape}")
        print(f"Sample data:\n{returns.head()}")

        # Test basic LSTM
        print("\n1. Testing Basic LSTM...")
        lstm = LSTMForecaster(
            lookback_window=30,
            forecast_horizon=1,
            hidden_units=[32, 16],
            epochs=50,
            batch_size=16
        )

        history = lstm.fit(returns.iloc[:-50], verbose=0)
        print(f"   Training loss: {history['loss'][-1]:.6f}")

        predictions = lstm.predict(returns.iloc[-60:], n_steps=5)
        print(f"\n   Predictions (5 steps):")
        print(predictions.round(4))

        # Test Attention LSTM
        print("\n2. Testing Attention LSTM...")
        attention_lstm = AttentionLSTMForecaster(
            lookback_window=30,
            hidden_units=[32, 16],
            epochs=50,
            batch_size=16
        )

        history = attention_lstm.fit(returns.iloc[:-50], verbose=0)
        print(f"   Training loss: {history['loss'][-1]:.6f}")

        predictions = attention_lstm.predict(returns.iloc[-60:], n_steps=5)
        print(f"\n   Predictions (5 steps):")
        print(predictions.round(4))

        # Test Ensemble
        print("\n3. Testing Ensemble LSTM...")
        ensemble = EnsembleLSTMForecaster(
            n_models=3,
            lookback_window=30,
            hidden_units=[32],
            epochs=30,
            batch_size=16
        )

        ensemble.fit(returns.iloc[:-50], verbose=0)

        predictions = ensemble.predict(returns.iloc[-60:], n_steps=5)
        print(f"\n   Ensemble Predictions (5 steps):")
        print(predictions.round(4))

        print("\n✅ LSTM forecasting implementation complete!")
