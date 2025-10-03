"""
Returns Forecasting Module

Implements multiple methods for forecasting asset returns:
- ARIMA (Autoregressive Integrated Moving Average)
- VAR (Vector Autoregression)
- ML-based models (Random Forest, XGBoost)
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple
import warnings
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.vector_ar.var_model import VAR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings('ignore')


class ReturnsForecast:
    """
    Forecasts future asset returns using time series and ML models.
    """

    def __init__(self, method: str = 'historical'):
        """
        Initialize returns forecasting model.

        Args:
            method: Forecasting method ('historical', 'arima', 'var', 'ml')
        """
        self.method = method
        self.model = None
        self.fitted = False

    def fit(
        self,
        returns: pd.DataFrame,
        arima_order: Tuple[int, int, int] = (1, 0, 1),
        var_lags: int = 5,
        **kwargs
    ):
        """
        Fit the forecasting model.

        Args:
            returns: DataFrame of historical returns
            arima_order: (p, d, q) order for ARIMA models
            var_lags: Number of lags for VAR model
            **kwargs: Additional parameters for specific models
        """
        if self.method == 'historical':
            # Simple historical mean
            self.historical_mean = returns.mean()
            self.fitted = True

        elif self.method == 'arima':
            # Fit ARIMA model for each asset separately
            self.model = {}
            for col in returns.columns:
                try:
                    model = ARIMA(returns[col].dropna(), order=arima_order)
                    self.model[col] = model.fit()
                except Exception as e:
                    print(f"ARIMA fit failed for {col}: {str(e)}")
                    self.model[col] = None
            self.fitted = True

        elif self.method == 'var':
            # Vector Autoregression (captures cross-asset dynamics)
            try:
                returns_clean = returns.dropna()
                var_model = VAR(returns_clean)
                self.model = var_model.fit(maxlags=var_lags, ic='aic')
                self.fitted = True
            except Exception as e:
                print(f"VAR fit failed: {str(e)}")
                # Fallback to historical mean
                self.historical_mean = returns.mean()
                self.method = 'historical'
                self.fitted = True

        elif self.method == 'ml':
            # Machine Learning approach with lagged features
            self.model = {}
            for col in returns.columns:
                model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=5,
                    random_state=42,
                    n_jobs=-1
                )
                self.model[col] = self._fit_ml_model(returns[col], model)
            self.fitted = True

        else:
            raise ValueError(f"Unknown forecasting method: {self.method}")

    def predict(
        self,
        horizon: int = 1,
        returns: Optional[pd.DataFrame] = None
    ) -> pd.Series:
        """
        Forecast returns for specified horizon.

        Args:
            horizon: Forecasting horizon (days ahead)
            returns: Recent returns data (for ML/VAR methods)

        Returns:
            Series of forecasted returns for each asset
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before prediction")

        if self.method == 'historical':
            # Return historical mean (annualized)
            return self.historical_mean * 252

        elif self.method == 'arima':
            forecasts = {}
            for asset, model in self.model.items():
                if model is not None:
                    try:
                        forecast = model.forecast(steps=horizon)
                        # Return annualized forecast
                        forecasts[asset] = forecast.mean() * 252
                    except:
                        forecasts[asset] = 0.0
                else:
                    forecasts[asset] = 0.0

            return pd.Series(forecasts)

        elif self.method == 'var':
            try:
                forecast = self.model.forecast(returns.values[-self.model.k_ar:], steps=horizon)
                # Average over horizon and annualize
                forecasted_returns = pd.Series(
                    forecast.mean(axis=0) * 252,
                    index=returns.columns
                )
                return forecasted_returns
            except:
                # Fallback
                return returns.mean() * 252

        elif self.method == 'ml':
            forecasts = {}
            for asset, model in self.model.items():
                if model is not None and returns is not None:
                    features = self._create_features(returns[asset])
                    if features is not None and len(features) > 0:
                        pred = model.predict(features[-1:])
                        forecasts[asset] = pred[0] * 252
                    else:
                        forecasts[asset] = 0.0
                else:
                    forecasts[asset] = 0.0

            return pd.Series(forecasts)

        else:
            return pd.Series()

    def _fit_ml_model(self, returns_series: pd.Series, model) -> Optional[object]:
        """
        Fit ML model with lagged features.

        Args:
            returns_series: Time series of returns for one asset
            model: Sklearn model instance

        Returns:
            Fitted model or None if fitting fails
        """
        try:
            # Create lagged features
            features = self._create_features(returns_series)

            if features is None or len(features) < 50:
                return None

            # Target: next day return
            y = returns_series.iloc[5:].values  # Skip first 5 due to lags

            # Train model
            X_train, X_test, y_train, y_test = train_test_split(
                features, y, test_size=0.2, shuffle=False
            )

            model.fit(X_train, y_train)

            return model

        except Exception as e:
            print(f"ML model fit failed: {str(e)}")
            return None

    def _create_features(self, returns_series: pd.Series) -> Optional[pd.DataFrame]:
        """
        Create lagged features for ML models.

        Args:
            returns_series: Time series of returns

        Returns:
            DataFrame of features or None
        """
        if len(returns_series) < 10:
            return None

        features = pd.DataFrame()
        # Lagged returns
        for lag in [1, 2, 3, 5]:
            features[f'lag_{lag}'] = returns_series.shift(lag)

        # Rolling statistics
        features['rolling_mean_5'] = returns_series.rolling(5).mean()
        features['rolling_std_5'] = returns_series.rolling(5).std()

        return features.dropna()

    def evaluate(
        self,
        train_returns: pd.DataFrame,
        test_returns: pd.DataFrame
    ) -> dict:
        """
        Evaluate forecasting performance.

        Args:
            train_returns: Training data
            test_returns: Testing data

        Returns:
            Dictionary of evaluation metrics
        """
        # Fit on training data
        self.fit(train_returns)

        # Predict
        predictions = self.predict(horizon=len(test_returns), returns=train_returns)

        # Actual realized returns (annualized)
        actual = test_returns.mean() * 252

        # Calculate metrics
        mse = mean_squared_error(actual, predictions)
        mae = mean_absolute_error(actual, predictions)
        correlation = np.corrcoef(actual, predictions)[0, 1]

        return {
            'mse': mse,
            'mae': mae,
            'rmse': np.sqrt(mse),
            'correlation': correlation,
            'method': self.method
        }


class EnsembleReturnsForecast:
    """
    Combines multiple forecasting methods for robust predictions.
    """

    def __init__(self, methods: list = ['historical', 'arima', 'var']):
        """
        Initialize ensemble forecast.

        Args:
            methods: List of forecasting methods to combine
        """
        self.forecasters = [ReturnsForecast(method=m) for m in methods]
        self.methods = methods

    def fit(self, returns: pd.DataFrame, **kwargs):
        """Fit all forecasters."""
        for forecaster in self.forecasters:
            try:
                forecaster.fit(returns, **kwargs)
            except Exception as e:
                print(f"Failed to fit {forecaster.method}: {str(e)}")

    def predict(
        self,
        horizon: int = 1,
        returns: Optional[pd.DataFrame] = None,
        weights: Optional[list] = None
    ) -> pd.Series:
        """
        Generate ensemble forecast.

        Args:
            horizon: Forecasting horizon
            returns: Recent returns data
            weights: Optional weights for each method (default: equal weight)

        Returns:
            Weighted average forecast
        """
        if weights is None:
            weights = [1.0 / len(self.forecasters)] * len(self.forecasters)

        forecasts = []
        for forecaster in self.forecasters:
            try:
                pred = forecaster.predict(horizon=horizon, returns=returns)
                forecasts.append(pred)
            except:
                # If a method fails, use zeros
                forecasts.append(pd.Series(0, index=returns.columns))

        # Weighted average
        ensemble_forecast = sum(w * f for w, f in zip(weights, forecasts))

        return ensemble_forecast


if __name__ == "__main__":
    # Example usage
    from src.data.loader import AssetDataLoader

    # Load data
    loader = AssetDataLoader()
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
    prices = loader.fetch_prices(tickers, '2020-01-01', '2023-12-31')
    returns = loader.compute_returns(prices)

    # Split into train/test
    split_idx = int(len(returns) * 0.8)
    train_returns = returns.iloc[:split_idx]
    test_returns = returns.iloc[split_idx:]

    # Test different forecasting methods
    methods = ['historical', 'arima', 'var']

    print("=== Returns Forecasting Evaluation ===\n")

    for method in methods:
        forecaster = ReturnsForecast(method=method)
        metrics = forecaster.evaluate(train_returns, test_returns)

        print(f"{method.upper()} Method:")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  MAE: {metrics['mae']:.4f}")
        print(f"  Correlation: {metrics['correlation']:.4f}\n")

    # Test ensemble
    print("=== Ensemble Forecast ===")
    ensemble = EnsembleReturnsForecast(methods=['historical', 'arima'])
    ensemble.fit(train_returns)
    forecast = ensemble.predict(returns=train_returns)
    print(f"Forecasted annual returns:\n{forecast}")
