"""
Real Market Data Loader

Fetches real stock price data from Yahoo Finance for portfolio optimization.
Supports multiple tickers and handles missing data gracefully.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple, Optional
import yfinance as yf


class RealDataLoader:
    """
    Load real market data from Yahoo Finance.

    Features:
    - Fetch historical price data for multiple tickers
    - Handle missing data and delisting
    - Compute returns and statistics
    - Cache data for performance
    """

    def __init__(self, cache_dir: str = "data/cache"):
        """
        Initialize the real data loader.

        Args:
            cache_dir: Directory to cache downloaded data
        """
        self.cache_dir = cache_dir
        self._cache = {}

    def fetch_data(
        self,
        tickers: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: str = "1y"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fetch historical price data for given tickers.

        Args:
            tickers: List of ticker symbols (e.g., ['AAPL', 'MSFT', 'GOOGL'])
            start_date: Start date in 'YYYY-MM-DD' format (optional)
            end_date: End date in 'YYYY-MM-DD' format (optional)
            period: Period string if dates not provided (e.g., '1y', '5y')

        Returns:
            prices: DataFrame with adjusted close prices
            returns: DataFrame with daily returns
        """
        # Set default dates if not provided
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        if start_date is None and period:
            end = datetime.strptime(end_date, '%Y-%m-%d')
            if period == '1mo':
                start = end - timedelta(days=30)
            elif period == '3mo':
                start = end - timedelta(days=90)
            elif period == '6mo':
                start = end - timedelta(days=180)
            elif period == '1y':
                start = end - timedelta(days=365)
            elif period == '2y':
                start = end - timedelta(days=730)
            elif period == '5y':
                start = end - timedelta(days=1825)
            else:
                start = end - timedelta(days=365)
            start_date = start.strftime('%Y-%m-%d')

        # Download data for all tickers
        try:
            if len(tickers) == 1:
                # Single ticker - simpler format
                data = yf.download(
                    tickers[0],
                    start=start_date,
                    end=end_date,
                    progress=False
                )
                if 'Close' in data.columns:
                    prices = data[['Close']].copy()
                    prices.columns = tickers
                elif 'Adj Close' in data.columns:
                    prices = data[['Adj Close']].copy()
                    prices.columns = tickers
                else:
                    raise ValueError(f"No Close or Adj Close column found for {tickers[0]}")

            else:
                # Multiple tickers
                data = yf.download(
                    tickers,
                    start=start_date,
                    end=end_date,
                    progress=False,
                    group_by='ticker'
                )

                prices_dict = {}
                for ticker in tickers:
                    try:
                        # Try different access patterns based on yfinance version
                        if isinstance(data.columns, pd.MultiIndex):
                            # Multi-index format
                            if (ticker, 'Close') in data.columns:
                                prices_dict[ticker] = data[(ticker, 'Close')]
                            elif (ticker, 'Adj Close') in data.columns:
                                prices_dict[ticker] = data[(ticker, 'Adj Close')]
                        else:
                            # Flat format
                            if f'{ticker}' in data.columns:
                                prices_dict[ticker] = data[ticker]
                    except Exception as e:
                        print(f"Warning: Could not extract data for {ticker}: {e}")

                if len(prices_dict) == 0:
                    raise ValueError("No valid data could be downloaded")

                prices = pd.DataFrame(prices_dict)

        except Exception as e:
            print(f"Error downloading data: {e}")
            raise ValueError(f"Failed to download data: {e}")

        # Handle missing data (forward fill then backward fill)
        prices = prices.ffill().bfill()

        # Drop rows with any remaining NaN
        prices = prices.dropna()

        # Compute returns
        returns = prices.pct_change().dropna()

        # Cache the data
        cache_key = f"{'-'.join(tickers)}_{start_date}_{end_date}"
        self._cache[cache_key] = (prices, returns)

        return prices, returns

    def get_popular_tickers(self, category: str = "tech") -> List[str]:
        """
        Get list of popular tickers by category.

        Args:
            category: Category name ('tech', 'finance', 'energy', 'healthcare', 'diversified')

        Returns:
            List of ticker symbols
        """
        categories = {
            'tech': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'NFLX', 'ADBE', 'CRM'],
            'finance': ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'BLK', 'SCHW', 'AXP', 'USB'],
            'energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY', 'HAL'],
            'healthcare': ['UNH', 'JNJ', 'PFE', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'LLY', 'AMGN'],
            'consumer': ['WMT', 'HD', 'PG', 'KO', 'PEP', 'COST', 'NKE', 'MCD', 'SBUX', 'TGT'],
            'diversified': ['AAPL', 'MSFT', 'JPM', 'JNJ', 'XOM', 'WMT', 'PG', 'VZ', 'T', 'DIS']
        }

        return categories.get(category, categories['diversified'])

    def fetch_sp500_subset(self, n_stocks: int = 10, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        """
        Fetch random subset of S&P 500 stocks.

        Args:
            n_stocks: Number of stocks to fetch
            seed: Random seed for reproducibility

        Returns:
            prices: DataFrame with prices
            returns: DataFrame with returns
            tickers: List of ticker symbols used
        """
        # Popular S&P 500 tickers that are reliable
        sp500_sample = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'UNH', 'JNJ',
            'JPM', 'V', 'PG', 'XOM', 'HD', 'MA', 'CVX', 'LLY', 'ABBV', 'MRK',
            'KO', 'PEP', 'COST', 'AVGO', 'WMT', 'MCD', 'DIS', 'CSCO', 'ADBE', 'NFLX'
        ]

        # Random selection
        np.random.seed(seed)
        selected = np.random.choice(sp500_sample, size=min(n_stocks, len(sp500_sample)), replace=False)
        tickers = list(selected)

        # Fetch data
        prices, returns = self.fetch_data(tickers, period='2y')

        return prices, returns, tickers

    def get_data_statistics(self, prices: pd.DataFrame, returns: pd.DataFrame) -> dict:
        """
        Compute summary statistics for the data.

        Args:
            prices: DataFrame with prices
            returns: DataFrame with returns

        Returns:
            Dictionary with statistics
        """
        stats = {
            'n_assets': len(prices.columns),
            'n_days': len(prices),
            'start_date': prices.index[0].strftime('%Y-%m-%d'),
            'end_date': prices.index[-1].strftime('%Y-%m-%d'),
            'mean_return': returns.mean().mean() * 252,  # Annualized
            'mean_volatility': returns.std().mean() * np.sqrt(252),  # Annualized
            'mean_sharpe': (returns.mean() / returns.std()).mean(),
            'correlation_mean': returns.corr().values[np.triu_indices_from(returns.corr().values, k=1)].mean(),
            'price_range': {
                'min': prices.min().min(),
                'max': prices.max().max(),
                'mean': prices.mean().mean()
            }
        }

        return stats


def example_usage():
    """Example usage of RealDataLoader."""
    loader = RealDataLoader()

    # Example 1: Fetch tech stocks
    print("Example 1: Tech Stocks")
    tech_tickers = loader.get_popular_tickers('tech')[:5]
    prices, returns = loader.fetch_data(tech_tickers, period='1y')
    stats = loader.get_data_statistics(prices, returns)

    print(f"Fetched {stats['n_assets']} assets")
    print(f"Period: {stats['start_date']} to {stats['end_date']}")
    print(f"Mean annual return: {stats['mean_return']:.2%}")
    print(f"Mean annual volatility: {stats['mean_volatility']:.2%}")
    print()

    # Example 2: Random S&P 500 subset
    print("Example 2: Random S&P 500 Subset")
    prices, returns, tickers = loader.fetch_sp500_subset(n_stocks=10, seed=42)
    stats = loader.get_data_statistics(prices, returns)

    print(f"Tickers: {', '.join(tickers)}")
    print(f"Fetched {stats['n_assets']} assets")
    print(f"Period: {stats['start_date']} to {stats['end_date']}")
    print(f"Mean correlation: {stats['correlation_mean']:.3f}")


if __name__ == '__main__':
    example_usage()
