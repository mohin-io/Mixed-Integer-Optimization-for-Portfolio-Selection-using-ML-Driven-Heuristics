"""
Unit tests for data loading module.
"""

import pytest
import pandas as pd
import numpy as np
from src.data.loader import AssetDataLoader


def test_asset_data_loader_init():
    """Test AssetDataLoader initialization."""
    loader = AssetDataLoader()
    assert loader.cache_dir.exists()
    assert loader.data_source == "yahoo"


def test_compute_returns_log():
    """Test log returns computation."""
    loader = AssetDataLoader()

    # Create sample price data
    prices = pd.DataFrame({
        'AAPL': [100, 110, 105, 120],
        'MSFT': [200, 210, 205, 215]
    })

    returns = loader.compute_returns(prices, method='log')

    assert len(returns) == 3  # One less than prices
    assert not returns.isnull().any().any()
    assert isinstance(returns, pd.DataFrame)


def test_compute_returns_simple():
    """Test simple returns computation."""
    loader = AssetDataLoader()

    prices = pd.DataFrame({
        'AAPL': [100, 110, 100],
    })

    returns = loader.compute_returns(prices, method='simple')

    assert len(returns) == 2
    assert returns.iloc[0, 0] == pytest.approx(0.10, rel=1e-5)
    assert returns.iloc[1, 0] == pytest.approx(-0.0909, rel=1e-3)


def test_handle_missing_data():
    """Test missing data handling."""
    loader = AssetDataLoader()

    prices = pd.DataFrame({
        'AAPL': [100, np.nan, 105, 110],
        'MSFT': [200, 205, np.nan, 215]
    })

    cleaned = loader.handle_missing_data(prices, method='forward_fill')

    assert not cleaned.isnull().any().any()
    assert len(cleaned) == len(prices)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
