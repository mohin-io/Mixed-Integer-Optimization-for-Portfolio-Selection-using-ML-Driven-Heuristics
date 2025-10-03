"""
Asset Data Loader Module

Responsibilities:
- Fetch historical price data from Yahoo Finance
- Support multiple asset classes (stocks, ETFs, crypto)
- Handle missing data, splits, dividends
- Cache data locally for reproducibility
"""

import os
from datetime import datetime, timedelta
from typing import List, Optional, Dict
import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


class AssetDataLoader:
    """
    Fetches and caches historical asset price data from Yahoo Finance.

    Attributes:
        cache_dir (Path): Directory to store cached data
        data_source (str): Data source identifier (default: 'yahoo')
    """

    def __init__(self, cache_dir: str = "data/raw"):
        """
        Initialize the data loader.

        Args:
            cache_dir: Directory path for caching downloaded data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.data_source = "yahoo"

    def fetch_prices(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch historical adjusted closing prices for given tickers.

        Args:
            tickers: List of ticker symbols (e.g., ['AAPL', 'GOOGL'])
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            use_cache: Whether to use cached data if available

        Returns:
            DataFrame with dates as index and tickers as columns
        """
        cache_file = self._get_cache_filename(tickers, start_date, end_date)

        # Try to load from cache
        if use_cache and cache_file.exists():
            print(f"Loading data from cache: {cache_file.name}")
            return pd.read_csv(cache_file, index_col=0, parse_dates=True)

        # Fetch from Yahoo Finance
        print(f"Fetching data for {len(tickers)} assets from {start_date} to {end_date}...")

        try:
            data = yf.download(
                tickers,
                start=start_date,
                end=end_date,
                progress=False,
                group_by='ticker'
            )

            # Handle single ticker vs multiple tickers
            if len(tickers) == 1:
                prices = data['Adj Close'].to_frame()
                prices.columns = tickers
            else:
                # Extract adjusted close prices for all tickers
                prices = pd.DataFrame()
                for ticker in tickers:
                    if ticker in data.columns.get_level_values(0):
                        prices[ticker] = data[ticker]['Adj Close']
                    else:
                        print(f"Warning: No data found for {ticker}")

            # Handle missing data
            prices = self.handle_missing_data(prices)

            # Cache the data
            prices.to_csv(cache_file)
            print(f"Data cached to: {cache_file.name}")

            return prices

        except Exception as e:
            print(f"Error fetching data: {str(e)}")
            raise

    def compute_returns(
        self,
        prices: pd.DataFrame,
        method: str = 'log',
        frequency: str = 'daily'
    ) -> pd.DataFrame:
        """
        Compute returns from price data.

        Args:
            prices: DataFrame of asset prices
            method: 'log' for log returns or 'simple' for arithmetic returns
            frequency: Return frequency ('daily', 'weekly', 'monthly')

        Returns:
            DataFrame of returns
        """
        if frequency == 'weekly':
            prices = prices.resample('W').last()
        elif frequency == 'monthly':
            prices = prices.resample('M').last()

        if method == 'log':
            returns = np.log(prices / prices.shift(1))
        elif method == 'simple':
            returns = prices.pct_change()
        else:
            raise ValueError(f"Unknown return method: {method}")

        return returns.dropna()

    def handle_missing_data(
        self,
        prices: pd.DataFrame,
        method: str = 'forward_fill'
    ) -> pd.DataFrame:
        """
        Handle missing data in price series.

        Args:
            prices: DataFrame with potential missing values
            method: Strategy to handle missing data
                   - 'forward_fill': Propagate last valid observation
                   - 'drop': Drop columns with any missing data
                   - 'interpolate': Linear interpolation

        Returns:
            DataFrame with missing data handled
        """
        missing_count = prices.isnull().sum()

        if missing_count.sum() > 0:
            print(f"\nMissing data detected:")
            print(missing_count[missing_count > 0])

        if method == 'forward_fill':
            prices = prices.fillna(method='ffill').fillna(method='bfill')
        elif method == 'drop':
            prices = prices.dropna(axis=1)
            print(f"Dropped {len(missing_count[missing_count > 0])} assets with missing data")
        elif method == 'interpolate':
            prices = prices.interpolate(method='linear')
        else:
            raise ValueError(f"Unknown missing data method: {method}")

        return prices

    def get_market_cap_data(self, tickers: List[str]) -> Dict[str, float]:
        """
        Fetch current market capitalization for assets.

        Args:
            tickers: List of ticker symbols

        Returns:
            Dictionary mapping ticker to market cap
        """
        market_caps = {}

        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                market_caps[ticker] = info.get('marketCap', np.nan)
            except Exception as e:
                print(f"Error fetching market cap for {ticker}: {str(e)}")
                market_caps[ticker] = np.nan

        return market_caps

    def _get_cache_filename(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str
    ) -> Path:
        """Generate unique cache filename based on query parameters."""
        tickers_str = "_".join(sorted(tickers)[:5])  # Limit filename length
        if len(tickers) > 5:
            tickers_str += f"_and_{len(tickers)-5}_more"

        filename = f"{tickers_str}_{start_date}_{end_date}.csv"
        return self.cache_dir / filename

    def get_available_tickers(self, index: str = 'SP500') -> List[str]:
        """
        Get list of tickers from a major index.

        Args:
            index: Index name ('SP500', 'NASDAQ100', 'DOW30')

        Returns:
            List of ticker symbols
        """
        if index == 'SP500':
            # Fetch S&P 500 constituents from Wikipedia
            try:
                url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
                tables = pd.read_html(url)
                df = tables[0]
                return df['Symbol'].str.replace('.', '-').tolist()
            except Exception as e:
                print(f"Error fetching S&P 500 tickers: {str(e)}")
                # Fallback to hardcoded list
                return self._get_default_tickers()
        else:
            raise NotImplementedError(f"Index {index} not yet supported")

    def _get_default_tickers(self) -> List[str]:
        """Return a default set of liquid, large-cap stocks."""
        return [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B',
            'UNH', 'JNJ', 'XOM', 'V', 'JPM', 'WMT', 'PG', 'MA', 'HD', 'CVX',
            'LLY', 'MRK', 'ABBV', 'PEP', 'KO', 'COST', 'AVGO', 'TMO', 'MCD'
        ]


if __name__ == "__main__":
    # Example usage
    loader = AssetDataLoader()

    # Fetch data for tech stocks
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']
    start_date = '2020-01-01'
    end_date = '2023-12-31'

    prices = loader.fetch_prices(tickers, start_date, end_date)
    print(f"\nFetched prices shape: {prices.shape}")
    print(f"Date range: {prices.index[0]} to {prices.index[-1]}")
    print(f"\nFirst few rows:\n{prices.head()}")

    # Compute returns
    returns = loader.compute_returns(prices, method='log')
    print(f"\nReturns shape: {returns.shape}")
    print(f"\nReturns statistics:\n{returns.describe()}")
