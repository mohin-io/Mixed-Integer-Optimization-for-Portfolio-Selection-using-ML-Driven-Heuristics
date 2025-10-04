"""
Abstract interface for broker integrations.

Defines the standard interface that all broker implementations must follow.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import pandas as pd


class BrokerInterface(ABC):
    """Abstract interface for broker integrations."""

    @abstractmethod
    def connect(self) -> bool:
        """
        Connect to broker API.

        Returns:
            True if connection successful, False otherwise

        Raises:
            ConnectionError: If connection fails
        """
        pass

    @abstractmethod
    def disconnect(self) -> bool:
        """
        Disconnect from broker API.

        Returns:
            True if disconnection successful, False otherwise
        """
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """
        Check if currently connected to broker.

        Returns:
            True if connected, False otherwise
        """
        pass

    @abstractmethod
    def get_account_balance(self) -> float:
        """
        Get current account balance.

        Returns:
            Current account net liquidation value

        Raises:
            RuntimeError: If not connected or API call fails
        """
        pass

    @abstractmethod
    def get_positions(self) -> pd.DataFrame:
        """
        Get current portfolio positions.

        Returns:
            DataFrame with columns: symbol, quantity, avg_cost, market_value, unrealized_pnl

        Raises:
            RuntimeError: If not connected or API call fails
        """
        pass

    @abstractmethod
    def get_market_data(self, symbols: List[str], data_type: str = 'snapshot') -> pd.DataFrame:
        """
        Get real-time market data for symbols.

        Args:
            symbols: List of ticker symbols
            data_type: 'snapshot' for current prices, 'stream' for live updates

        Returns:
            DataFrame with columns: symbol, bid, ask, last, volume, timestamp

        Raises:
            RuntimeError: If not connected or API call fails
        """
        pass

    @abstractmethod
    def place_order(self, symbol: str, quantity: int, order_type: str = 'MARKET',
                   limit_price: Optional[float] = None) -> Optional[str]:
        """
        Place an order.

        Args:
            symbol: Ticker symbol
            quantity: Number of shares (positive for buy, negative for sell)
            order_type: 'MARKET' or 'LIMIT'
            limit_price: Limit price for LIMIT orders

        Returns:
            Order ID if successful, None otherwise

        Raises:
            ValueError: If invalid parameters
            RuntimeError: If not connected or order placement fails
        """
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if cancellation successful, False otherwise

        Raises:
            RuntimeError: If not connected or cancellation fails
        """
        pass

    @abstractmethod
    def get_order_status(self, order_id: str) -> Dict:
        """
        Get order status.

        Args:
            order_id: Order ID to check

        Returns:
            Dictionary with order details (status, filled_quantity, remaining_quantity, etc.)

        Raises:
            RuntimeError: If not connected or API call fails
        """
        pass

    @abstractmethod
    def get_historical_data(self, symbol: str, start_date: str, end_date: str,
                           bar_size: str = '1 day') -> pd.DataFrame:
        """
        Get historical price data.

        Args:
            symbol: Ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            bar_size: Bar size ('1 min', '5 mins', '1 hour', '1 day', etc.)

        Returns:
            DataFrame with columns: date, open, high, low, close, volume

        Raises:
            RuntimeError: If not connected or API call fails
        """
        pass

    def get_account_summary(self) -> Dict:
        """
        Get account summary.

        Returns:
            Dictionary with account details (balance, buying_power, unrealized_pnl, etc.)

        Raises:
            RuntimeError: If not connected
        """
        balance = self.get_account_balance()
        positions = self.get_positions()

        total_value = balance
        unrealized_pnl = 0.0

        if not positions.empty and 'unrealized_pnl' in positions.columns:
            unrealized_pnl = positions['unrealized_pnl'].sum()

        return {
            'balance': balance,
            'total_value': total_value,
            'unrealized_pnl': unrealized_pnl,
            'num_positions': len(positions)
        }
