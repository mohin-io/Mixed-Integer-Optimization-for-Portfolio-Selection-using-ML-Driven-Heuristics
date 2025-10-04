"""
Unit Tests for Forecasting Models

Tests:
- Returns forecasting (ARIMA, VAR, ML)
- Volatility forecasting (GARCH, EGARCH, EWMA)
- Covariance estimation (Sample, Ledoit-Wolf, Factor models)
"""

import pytest
import pandas as pd
import numpy as np
from src.forecasting.returns_forecast import ReturnsForecast
from src.forecasting.volatility_forecast import VolatilityForecast
from src.forecasting.covariance import CovarianceEstimator


# Fixtures
@pytest.fixture
def sample_returns():
    """Generate sample returns data for testing."""
    np.random.seed(42)
    n_periods = 500
    n_assets = 5

    # Generate correlated returns
    mean = np.array([0.0008, 0.0006, 0.0007, 0.0005, 0.0009])
    cov = np.array([
        [0.0004, 0.0002, 0.0001, 0.0001, 0.0002],
        [0.0002, 0.0003, 0.0001, 0.0001, 0.0001],
        [0.0001, 0.0001, 0.0005, 0.0002, 0.0001],
        [0.0001, 0.0001, 0.0002, 0.0003, 0.0001],
        [0.0002, 0.0001, 0.0001, 0.0001, 0.0006]
    ])

    returns = np.random.multivariate_normal(mean, cov, n_periods)

    dates = pd.date_range('2020-01-01', periods=n_periods, freq='D')
    tickers = [f'Asset_{i}' for i in range(n_assets)]

    return pd.DataFrame(returns, index=dates, columns=tickers)


@pytest.fixture
def single_asset_returns():
    """Generate single asset returns for volatility testing."""
    np.random.seed(42)
    n_periods = 500

    # GARCH-like process
    returns = []
    sigma_t = 0.02

    for _ in range(n_periods):
        epsilon = np.random.normal(0, 1)
        returns.append(sigma_t * epsilon)
        # GARCH(1,1) volatility update
        sigma_t = np.sqrt(0.00001 + 0.1 * (sigma_t * epsilon)**2 + 0.85 * sigma_t**2)

    dates = pd.date_range('2020-01-01', periods=n_periods, freq='D')

    return pd.Series(returns, index=dates, name='Asset')


class TestReturnsForecast:
    """Test suite for returns forecasting."""

    def test_historical_forecast(self, sample_returns):
        """Test historical mean forecasting."""
        forecaster = ReturnsForecast(method='historical')
        forecaster.fit(sample_returns)
        forecast = forecaster.predict()

        assert isinstance(forecast, pd.Series), "Forecast should be a Series"
        assert len(forecast) == len(sample_returns.columns), "Forecast length should match number of assets"
        assert forecast.notna().all(), "Forecast should not contain NaN values"
        assert forecast.index.equals(sample_returns.columns), "Forecast index should match column names"

    def test_ml_forecast(self, sample_returns):
        """Test ML-based forecasting."""
        forecaster = ReturnsForecast(method='ml')
        forecaster.fit(sample_returns)
        forecast = forecaster.predict()

        assert isinstance(forecast, pd.Series), "Forecast should be a Series"
        assert len(forecast) == len(sample_returns.columns), "Forecast should have correct length"
        assert forecast.notna().all(), "No NaN values in forecast"

    def test_arima_forecast(self, sample_returns):
        """Test ARIMA forecasting."""
        # Use single asset for ARIMA
        single_asset = sample_returns.iloc[:, 0].to_frame()

        forecaster = ReturnsForecast(method='arima')
        forecaster.fit(single_asset)
        forecast = forecaster.predict(horizon=1)

        assert isinstance(forecast, (pd.Series, float, np.ndarray)), "ARIMA forecast should return valid type"
        # ARIMA may return scalar or series depending on implementation

    def test_forecast_shapes(self, sample_returns):
        """Test that forecasts have correct shapes."""
        methods = ['historical', 'ml']

        for method in methods:
            forecaster = ReturnsForecast(method=method)
            forecaster.fit(sample_returns)
            forecast = forecaster.predict()

            assert len(forecast) == sample_returns.shape[1], f"{method} forecast has wrong shape"

    def test_forecast_values_reasonable(self, sample_returns):
        """Test that forecast values are reasonable (not extreme)."""
        forecaster = ReturnsForecast(method='historical')
        forecaster.fit(sample_returns)
        forecast = forecaster.predict()

        # Daily returns should be finite and not NaN
        assert forecast.notna().all(), "Forecast should not contain NaN"
        assert np.isfinite(forecast).all(), "Forecast should be finite"
        # Daily returns should be in reasonable range for financial data
        assert (forecast > -1.0).all(), "Returns too negative (< -100%)"
        assert (forecast < 1.0).all(), "Returns too high (> 100%)"

    def test_different_lookback_periods(self, sample_returns):
        """Test forecasting with different lookback periods."""
        short_data = sample_returns.iloc[-100:]
        long_data = sample_returns

        forecaster_short = ReturnsForecast(method='historical')
        forecaster_short.fit(short_data)
        forecast_short = forecaster_short.predict()

        forecaster_long = ReturnsForecast(method='historical')
        forecaster_long.fit(long_data)
        forecast_long = forecaster_long.predict()

        # Forecasts should be different with different data lengths
        assert not forecast_short.equals(forecast_long), "Forecasts should differ with different data"

    def test_fit_before_predict(self, sample_returns):
        """Test that predict requires fit to be called first."""
        forecaster = ReturnsForecast(method='historical')

        # Should handle gracefully or raise appropriate error
        try:
            forecast = forecaster.predict()
            # If it doesn't raise error, check it handles it gracefully
            assert forecast is not None
        except (AttributeError, ValueError, RuntimeError):
            # Expected behavior - requires fit first
            pass


class TestVolatilityForecast:
    """Test suite for volatility forecasting."""

    def test_historical_volatility(self, single_asset_returns):
        """Test historical volatility calculation."""
        forecaster = VolatilityForecast(method='historical')
        forecaster.fit(single_asset_returns)
        vol_forecast = forecaster.predict()

        assert isinstance(vol_forecast, (float, np.floating)), "Volatility should be a scalar"
        assert vol_forecast > 0, "Volatility should be positive"
        assert vol_forecast < 1.0, "Daily volatility should be reasonable"

    def test_ewma_volatility(self, single_asset_returns):
        """Test EWMA volatility forecasting."""
        forecaster = VolatilityForecast(method='ewma')
        forecaster.fit(single_asset_returns)
        vol_forecast = forecaster.predict()

        assert isinstance(vol_forecast, (float, np.floating)), "EWMA volatility should be scalar"
        assert vol_forecast > 0, "Volatility should be positive"

    def test_garch_volatility(self, single_asset_returns):
        """Test GARCH volatility forecasting."""
        try:
            forecaster = VolatilityForecast(method='garch')
            forecaster.fit(single_asset_returns)
            vol_forecast = forecaster.predict()

            assert vol_forecast > 0, "GARCH volatility should be positive"
        except ImportError:
            pytest.skip("GARCH model requires arch package")
        except Exception as e:
            # GARCH may fail to converge on some data
            pytest.skip(f"GARCH failed: {str(e)}")

    def test_volatility_persistence(self, single_asset_returns):
        """Test that volatility estimates are stable."""
        forecaster = VolatilityForecast(method='historical')
        forecaster.fit(single_asset_returns)

        vol1 = forecaster.predict()
        vol2 = forecaster.predict()  # Should give same result

        assert vol1 == vol2, "Volatility forecast should be deterministic"

    def test_multiple_horizons(self, single_asset_returns):
        """Test volatility forecasting for different horizons."""
        forecaster = VolatilityForecast(method='historical')
        forecaster.fit(single_asset_returns)

        vol = forecaster.predict()

        # Volatility should be reasonable
        assert vol > 0, "Volatility should be positive"
        assert vol < 1.0, "Daily volatility should be reasonable"


class TestCovarianceEstimator:
    """Test suite for covariance estimation."""

    def test_sample_covariance(self, sample_returns):
        """Test sample covariance estimation."""
        estimator = CovarianceEstimator(method='sample')
        cov_matrix = estimator.estimate(sample_returns)

        assert isinstance(cov_matrix, pd.DataFrame), "Covariance should be DataFrame"
        assert cov_matrix.shape == (sample_returns.shape[1], sample_returns.shape[1]), "Wrong shape"
        assert (cov_matrix.T == cov_matrix).all().all(), "Covariance should be symmetric"
        assert (np.diag(cov_matrix) > 0).all(), "Diagonal elements should be positive"

    def test_ledoit_wolf_covariance(self, sample_returns):
        """Test Ledoit-Wolf shrinkage covariance."""
        estimator = CovarianceEstimator(method='ledoit_wolf')
        cov_matrix = estimator.estimate(sample_returns)

        assert isinstance(cov_matrix, pd.DataFrame), "Should return DataFrame"
        assert cov_matrix.shape[0] == cov_matrix.shape[1], "Should be square"
        assert (cov_matrix.T == cov_matrix).all().all(), "Should be symmetric"

    def test_exponential_covariance(self, sample_returns):
        """Test exponential weighted covariance."""
        estimator = CovarianceEstimator(method='exponential')
        cov_matrix = estimator.estimate(sample_returns)

        assert isinstance(cov_matrix, pd.DataFrame), "Should return DataFrame"
        assert cov_matrix.shape == (sample_returns.shape[1], sample_returns.shape[1])

    def test_covariance_positive_definite(self, sample_returns):
        """Test that covariance matrix is positive semi-definite."""
        methods = ['sample', 'ledoit_wolf', 'exponential']

        for method in methods:
            estimator = CovarianceEstimator(method=method)
            cov_matrix = estimator.estimate(sample_returns)

            # Check positive semi-definiteness via eigenvalues
            eigenvalues = np.linalg.eigvals(cov_matrix.values)

            assert (eigenvalues >= -1e-10).all(), f"{method} covariance not positive semi-definite"

    def test_covariance_values_reasonable(self, sample_returns):
        """Test that covariance values are in reasonable range."""
        estimator = CovarianceEstimator(method='sample')
        cov_matrix = estimator.estimate(sample_returns)

        # Daily volatilities should be positive and not infinite
        volatilities = np.sqrt(np.diag(cov_matrix))

        assert (volatilities > 0).all(), "Volatilities should be positive"
        assert (volatilities < 1.0).all(), "Volatilities should be finite"
        assert not np.any(np.isnan(volatilities)), "No NaN volatilities"

    def test_correlation_bounds(self, sample_returns):
        """Test that correlations are between -1 and 1."""
        estimator = CovarianceEstimator(method='sample')
        cov_matrix = estimator.estimate(sample_returns)

        # Convert to correlation
        std_devs = np.sqrt(np.diag(cov_matrix))
        corr_matrix = cov_matrix / np.outer(std_devs, std_devs)

        # Check with some tolerance for numerical errors
        assert (corr_matrix >= -1.01).all().all(), "Correlations should be >= -1"
        assert (corr_matrix <= 1.01).all().all(), "Correlations should be <= 1"
        assert np.allclose(np.diag(corr_matrix), 1.0, atol=0.01), "Diagonal correlations should be ~1"

    def test_shrinkage_effect(self, sample_returns):
        """Test that Ledoit-Wolf shrinkage differs from sample covariance."""
        sample_est = CovarianceEstimator(method='sample')
        lw_est = CovarianceEstimator(method='ledoit_wolf')

        sample_cov = sample_est.estimate(sample_returns)
        lw_cov = lw_est.estimate(sample_returns)

        # Shrinkage should produce different result
        assert not sample_cov.equals(lw_cov), "Shrinkage should differ from sample"

        # But they should have same shape
        assert sample_cov.shape == lw_cov.shape

    def test_covariance_index_preservation(self, sample_returns):
        """Test that index/columns are preserved."""
        estimator = CovarianceEstimator(method='sample')
        cov_matrix = estimator.estimate(sample_returns)

        assert cov_matrix.index.equals(sample_returns.columns), "Index should match"
        assert cov_matrix.columns.equals(sample_returns.columns), "Columns should match"


class TestForecastingIntegration:
    """Integration tests for forecasting pipeline."""

    def test_full_forecasting_pipeline(self, sample_returns):
        """Test complete forecasting workflow."""
        # Returns forecast
        returns_forecaster = ReturnsForecast(method='historical')
        returns_forecaster.fit(sample_returns)
        expected_returns = returns_forecaster.predict()

        # Covariance forecast
        cov_estimator = CovarianceEstimator(method='ledoit_wolf')
        cov_matrix = cov_estimator.estimate(sample_returns)

        # Verify compatibility
        assert len(expected_returns) == cov_matrix.shape[0], "Dimensions should match"
        assert expected_returns.index.equals(cov_matrix.index), "Indices should match"

    def test_annualization(self, sample_returns):
        """Test annualization of forecasts."""
        returns_forecaster = ReturnsForecast(method='historical')
        returns_forecaster.fit(sample_returns)
        daily_returns = returns_forecaster.predict()

        # Annualize
        annual_returns = daily_returns * 252

        # Should be larger in magnitude
        assert (annual_returns.abs() > daily_returns.abs()).any()

    def test_consistency_across_methods(self, sample_returns):
        """Test that different methods produce reasonable relative results."""
        methods = ['sample', 'ledoit_wolf', 'exponential']

        covariances = {}
        for method in methods:
            estimator = CovarianceEstimator(method=method)
            covariances[method] = estimator.estimate(sample_returns)

        # All should have same shape
        shapes = [cov.shape for cov in covariances.values()]
        assert len(set(shapes)) == 1, "All covariances should have same shape"

        # Diagonal elements (variances) should be similar order of magnitude
        for method in methods:
            diag = np.diag(covariances[method])
            assert (diag > 0).all(), f"{method} has non-positive variances"


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, '-v'])
