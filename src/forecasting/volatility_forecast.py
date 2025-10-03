"""
Volatility Forecasting Module

Implements GARCH family models for volatility forecasting:
- GARCH(1,1): Standard GARCH model
- EGARCH: Exponential GARCH (captures asymmetric shocks)
- Historical volatility as baseline
"""

import pandas as pd
import numpy as np
from typing import Optional
import warnings
from arch import arch_model

warnings.filterwarnings('ignore')


class VolatilityForecast:
    """
    Forecasts asset volatility using GARCH models.
    """

    def __init__(self, method: str = 'garch'):
        """
        Initialize volatility forecasting model.

        Args:
            method: Forecasting method ('historical', 'garch', 'egarch', 'ewma')
        """
        self.method = method
        self.models = {}
        self.fitted = False

    def fit(
        self,
        returns: pd.DataFrame,
        p: int = 1,
        q: int = 1,
        **kwargs
    ):
        """
        Fit volatility model.

        Args:
            returns: DataFrame of asset returns (not percentage)
            p: GARCH lag order
            q: ARCH lag order
        """
        if self.method == 'historical':
            # Simple rolling standard deviation
            self.historical_vol = returns.std() * np.sqrt(252)
            self.fitted = True

        elif self.method == 'ewma':
            # Exponentially weighted moving average
            self.ewma_vol = returns.ewm(span=60).std() * np.sqrt(252)
            self.fitted = True

        elif self.method == 'garch':
            # Fit GARCH model for each asset
            for col in returns.columns:
                try:
                    # Convert to percentage returns for ARCH package
                    ret_pct = returns[col].dropna() * 100

                    model = arch_model(
                        ret_pct,
                        vol='Garch',
                        p=p,
                        q=q,
                        rescale=False
                    )

                    self.models[col] = model.fit(disp='off', show_warning=False)
                except Exception as e:
                    print(f"GARCH fit failed for {col}: {str(e)}")
                    self.models[col] = None

            self.fitted = True

        elif self.method == 'egarch':
            # Exponential GARCH (captures leverage effects)
            for col in returns.columns:
                try:
                    ret_pct = returns[col].dropna() * 100

                    model = arch_model(
                        ret_pct,
                        vol='EGARCH',
                        p=p,
                        q=q,
                        rescale=False
                    )

                    self.models[col] = model.fit(disp='off', show_warning=False)
                except Exception as e:
                    print(f"EGARCH fit failed for {col}: {str(e)}")
                    self.models[col] = None

            self.fitted = True

        else:
            raise ValueError(f"Unknown volatility method: {self.method}")

    def predict(
        self,
        horizon: int = 1,
        returns: Optional[pd.DataFrame] = None
    ) -> pd.Series:
        """
        Forecast volatility for specified horizon.

        Args:
            horizon: Forecasting horizon (days ahead)
            returns: Recent returns (for EWMA method)

        Returns:
            Series of annualized volatility forecasts
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before prediction")

        if self.method == 'historical':
            return self.historical_vol

        elif self.method == 'ewma':
            if returns is not None:
                # Use most recent EWMA volatility
                return returns.ewm(span=60).std().iloc[-1] * np.sqrt(252)
            else:
                return self.ewma_vol.iloc[-1]

        elif self.method in ['garch', 'egarch']:
            forecasts = {}

            for asset, model in self.models.items():
                if model is not None:
                    try:
                        # Forecast variance
                        forecast = model.forecast(horizon=horizon)
                        # Extract variance and convert to volatility
                        variance = forecast.variance.values[-1, -1]
                        # Convert from percentage to decimal and annualize
                        volatility = np.sqrt(variance) / 100 * np.sqrt(252)
                        forecasts[asset] = volatility
                    except Exception as e:
                        # Fallback to historical vol
                        forecasts[asset] = self.historical_vol[asset]
                else:
                    forecasts[asset] = self.historical_vol[asset]

            return pd.Series(forecasts)

        else:
            raise ValueError(f"Unknown method: {self.method}")

    def evaluate(
        self,
        train_returns: pd.DataFrame,
        test_returns: pd.DataFrame,
        horizon: int = 21
    ) -> dict:
        """
        Evaluate volatility forecasting accuracy.

        Args:
            train_returns: Training period returns
            test_returns: Testing period returns
            horizon: Forecast horizon

        Returns:
            Dictionary of evaluation metrics
        """
        # Fit on training data
        self.fit(train_returns)

        # Forecast volatility
        predicted_vol = self.predict(horizon=horizon, returns=train_returns)

        # Realized volatility in test period
        realized_vol = test_returns.std() * np.sqrt(252)

        # Calculate errors
        errors = predicted_vol - realized_vol
        mse = (errors ** 2).mean()
        mae = errors.abs().mean()

        return {
            'mse': mse,
            'mae': mae,
            'rmse': np.sqrt(mse),
            'mean_predicted': predicted_vol.mean(),
            'mean_realized': realized_vol.mean(),
            'method': self.method
        }


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

    # Test different volatility methods
    methods = ['historical', 'ewma', 'garch']

    print("=== Volatility Forecasting Evaluation ===\n")

    for method in methods:
        forecaster = VolatilityForecast(method=method)
        metrics = forecaster.evaluate(train_returns, test_returns)

        print(f"{method.upper()} Method:")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  MAE: {metrics['mae']:.4f}")
        print(f"  Mean Predicted Vol: {metrics['mean_predicted']:.4f}")
        print(f"  Mean Realized Vol: {metrics['mean_realized']:.4f}\n")

    # Example forecast
    forecaster = VolatilityForecast(method='garch')
    forecaster.fit(train_returns)
    vol_forecast = forecaster.predict(horizon=21)

    print("=== GARCH Volatility Forecasts (Annualized) ===")
    print(vol_forecast)
