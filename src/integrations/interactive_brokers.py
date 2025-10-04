"""
Interactive Brokers API integration using ib_insync.

This module provides a production-ready implementation of the BrokerInterface
for Interactive Brokers, supporting both paper and live trading.

Requirements:
    pip install ib_insync

Usage:
    from src.integrations import InteractiveBrokersAPI, BrokerConfig

    # Create configuration
    config = BrokerConfig(ib_paper_trading=True, ib_port=7497)

    # Initialize and connect
    broker = InteractiveBrokersAPI(config)
    broker.connect()

    # Get market data
    data = broker.get_market_data(['AAPL', 'MSFT'])

    # Place order
    order_id = broker.place_order('AAPL', 100)  # Buy 100 shares

    # Disconnect
    broker.disconnect()
"""

from typing import List, Dict, Optional
import pandas as pd
import logging
from datetime import datetime, timedelta

try:
    from ib_insync import IB, Stock, MarketOrder, LimitOrder, Contract, util
    IB_INSYNC_AVAILABLE = True
except ImportError:
    IB_INSYNC_AVAILABLE = False
    logging.warning("ib_insync not installed. Install with: pip install ib_insync")

from .broker_interface import BrokerInterface
from .broker_config import BrokerConfig

logger = logging.getLogger(__name__)


class InteractiveBrokersAPI(BrokerInterface):
    """Interactive Brokers API implementation using ib_insync."""

    def __init__(self, config: Optional[BrokerConfig] = None):
        """
        Initialize Interactive Brokers API.

        Args:
            config: Broker configuration. If None, uses default config from environment.

        Raises:
            ImportError: If ib_insync is not installed
        """
        if not IB_INSYNC_AVAILABLE:
            raise ImportError(
                "ib_insync is required for Interactive Brokers integration. "
                "Install with: pip install ib_insync"
            )

        self.config = config or BrokerConfig.from_env()
        self.config.validate()

        self.ib = IB()
        self.connected = False

        logger.info(f"Initialized IB API: {self.config}")

    def connect(self) -> bool:
        """Connect to Interactive Brokers TWS/Gateway."""
        try:
            self.ib.connect(
                self.config.ib_host,
                self.config.ib_port,
                clientId=self.config.ib_client_id,
                readonly=False,
                timeout=20
            )
            self.connected = True

            mode = "Paper Trading" if self.config.ib_paper_trading else "Live Trading"
            logger.info(f"✅ Connected to IB {mode} at {self.config.ib_host}:{self.config.ib_port}")

            return True

        except Exception as e:
            logger.error(f"❌ Failed to connect to IB: {str(e)}")
            self.connected = False
            raise ConnectionError(f"Failed to connect to Interactive Brokers: {str(e)}")

    def disconnect(self) -> bool:
        """Disconnect from Interactive Brokers."""
        try:
            if self.connected:
                self.ib.disconnect()
                self.connected = False
                logger.info("Disconnected from IB")
            return True

        except Exception as e:
            logger.error(f"Failed to disconnect from IB: {str(e)}")
            return False

    def is_connected(self) -> bool:
        """Check if currently connected."""
        return self.connected and self.ib.isConnected()

    def _ensure_connected(self):
        """Ensure connection is active, raise RuntimeError if not."""
        if not self.is_connected():
            raise RuntimeError("Not connected to Interactive Brokers. Call connect() first.")

    def get_account_balance(self) -> float:
        """Get current account net liquidation value."""
        self._ensure_connected()

        try:
            account_values = self.ib.accountValues()

            for value in account_values:
                if value.tag == 'NetLiquidation' and value.currency == 'USD':
                    balance = float(value.value)
                    logger.info(f"Account balance: ${balance:,.2f}")
                    return balance

            logger.warning("NetLiquidation value not found, returning 0")
            return 0.0

        except Exception as e:
            logger.error(f"Failed to get account balance: {str(e)}")
            raise RuntimeError(f"Failed to get account balance: {str(e)}")

    def get_positions(self) -> pd.DataFrame:
        """Get current portfolio positions."""
        self._ensure_connected()

        try:
            positions = self.ib.positions()

            if not positions:
                logger.info("No open positions")
                return pd.DataFrame(columns=['symbol', 'quantity', 'avg_cost',
                                            'market_value', 'unrealized_pnl'])

            data = []
            for position in positions:
                market_price = position.avgCost  # Default to avg cost
                try:
                    # Try to get current market price
                    ticker = self.ib.reqTickers(position.contract)[0]
                    if ticker.marketPrice():
                        market_price = ticker.marketPrice()
                except:
                    pass

                data.append({
                    'symbol': position.contract.symbol,
                    'quantity': float(position.position),
                    'avg_cost': float(position.avgCost),
                    'market_value': float(position.position * market_price),
                    'unrealized_pnl': float(position.position * (market_price - position.avgCost))
                })

            df = pd.DataFrame(data)
            logger.info(f"Retrieved {len(df)} positions")
            return df

        except Exception as e:
            logger.error(f"Failed to get positions: {str(e)}")
            raise RuntimeError(f"Failed to get positions: {str(e)}")

    def get_market_data(self, symbols: List[str], data_type: str = 'snapshot') -> pd.DataFrame:
        """Get real-time market data for symbols."""
        self._ensure_connected()

        try:
            data = []

            for symbol in symbols:
                contract = Stock(symbol, 'SMART', 'USD')
                self.ib.qualifyContracts(contract)

                # Request market data
                ticker = self.ib.reqMktData(contract, '', False, False)
                self.ib.sleep(0.5)  # Wait for data to arrive

                data.append({
                    'symbol': symbol,
                    'bid': ticker.bid if ticker.bid else 0.0,
                    'ask': ticker.ask if ticker.ask else 0.0,
                    'last': ticker.last if ticker.last else 0.0,
                    'volume': ticker.volume if ticker.volume else 0,
                    'timestamp': datetime.now()
                })

                # Cancel market data subscription to avoid overload
                self.ib.cancelMktData(contract)

            df = pd.DataFrame(data)
            logger.info(f"Retrieved market data for {len(symbols)} symbols")
            return df

        except Exception as e:
            logger.error(f"Failed to get market data: {str(e)}")
            raise RuntimeError(f"Failed to get market data: {str(e)}")

    def place_order(self, symbol: str, quantity: int, order_type: str = 'MARKET',
                   limit_price: Optional[float] = None) -> Optional[str]:
        """Place an order."""
        self._ensure_connected()

        try:
            # Validate quantity
            if quantity == 0:
                raise ValueError("Order quantity cannot be zero")

            # Create contract
            contract = Stock(symbol, 'SMART', 'USD')
            self.ib.qualifyContracts(contract)

            # Create order
            action = 'BUY' if quantity > 0 else 'SELL'
            total_quantity = abs(quantity)

            if order_type == 'MARKET':
                order = MarketOrder(action, total_quantity)
            elif order_type == 'LIMIT':
                if limit_price is None:
                    raise ValueError("limit_price required for LIMIT orders")
                order = LimitOrder(action, total_quantity, limit_price)
            else:
                raise ValueError(f"Unsupported order type: {order_type}")

            # Place order
            trade = self.ib.placeOrder(contract, order)
            self.ib.sleep(1)  # Wait for order to be submitted

            order_id = str(trade.order.orderId)
            logger.info(f"✅ Placed {order_type} order: {symbol} {quantity} shares, Order ID: {order_id}")

            return order_id

        except Exception as e:
            logger.error(f"❌ Failed to place order for {symbol}: {str(e)}")
            raise RuntimeError(f"Failed to place order: {str(e)}")

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        self._ensure_connected()

        try:
            # Find the trade with this order ID
            trades = self.ib.openTrades()
            for trade in trades:
                if str(trade.order.orderId) == order_id:
                    self.ib.cancelOrder(trade.order)
                    logger.info(f"Cancelled order {order_id}")
                    return True

            logger.warning(f"Order {order_id} not found in open orders")
            return False

        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {str(e)}")
            raise RuntimeError(f"Failed to cancel order: {str(e)}")

    def get_order_status(self, order_id: str) -> Dict:
        """Get order status."""
        self._ensure_connected()

        try:
            # Search in all trades (open and closed)
            all_trades = self.ib.trades()

            for trade in all_trades:
                if str(trade.order.orderId) == order_id:
                    return {
                        'order_id': order_id,
                        'symbol': trade.contract.symbol,
                        'status': trade.orderStatus.status,
                        'filled_quantity': trade.orderStatus.filled,
                        'remaining_quantity': trade.orderStatus.remaining,
                        'avg_fill_price': trade.orderStatus.avgFillPrice,
                        'last_update': trade.orderStatus.lastFillTime
                    }

            logger.warning(f"Order {order_id} not found")
            return {'order_id': order_id, 'status': 'NOT_FOUND'}

        except Exception as e:
            logger.error(f"Failed to get order status for {order_id}: {str(e)}")
            raise RuntimeError(f"Failed to get order status: {str(e)}")

    def get_historical_data(self, symbol: str, start_date: str, end_date: str,
                           bar_size: str = '1 day') -> pd.DataFrame:
        """Get historical price data."""
        self._ensure_connected()

        try:
            contract = Stock(symbol, 'SMART', 'USD')
            self.ib.qualifyContracts(contract)

            # Calculate duration from dates
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            duration_days = (end - start).days

            # IB uses duration strings like '1 Y', '6 M', '30 D'
            if duration_days > 365:
                duration_str = f"{duration_days // 365} Y"
            elif duration_days > 30:
                duration_str = f"{duration_days // 30} M"
            else:
                duration_str = f"{duration_days} D"

            # Request historical data
            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime=end_date,
                durationStr=duration_str,
                barSizeSetting=bar_size,
                whatToShow='TRADES',
                useRTH=True,  # Regular trading hours only
                formatDate=1
            )

            if not bars:
                logger.warning(f"No historical data returned for {symbol}")
                return pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close', 'volume'])

            # Convert to DataFrame
            df = util.df(bars)
            df = df.rename(columns={'date': 'date', 'open': 'open', 'high': 'high',
                                   'low': 'low', 'close': 'close', 'volume': 'volume'})
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)

            logger.info(f"Retrieved {len(df)} bars of historical data for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Failed to get historical data for {symbol}: {str(e)}")
            raise RuntimeError(f"Failed to get historical data: {str(e)}")

    def get_account_summary(self) -> Dict:
        """Get comprehensive account summary."""
        self._ensure_connected()

        try:
            summary = super().get_account_summary()

            # Add IB-specific details
            account_values = self.ib.accountValues()

            for value in account_values:
                if value.tag == 'BuyingPower' and value.currency == 'USD':
                    summary['buying_power'] = float(value.value)
                elif value.tag == 'TotalCashValue' and value.currency == 'USD':
                    summary['cash'] = float(value.value)
                elif value.tag == 'GrossPositionValue' and value.currency == 'USD':
                    summary['gross_position_value'] = float(value.value)

            logger.info(f"Account summary: {summary}")
            return summary

        except Exception as e:
            logger.error(f"Failed to get account summary: {str(e)}")
            raise RuntimeError(f"Failed to get account summary: {str(e)}")

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()

    def __repr__(self) -> str:
        """String representation."""
        status = "Connected" if self.is_connected() else "Disconnected"
        mode = "Paper" if self.config.ib_paper_trading else "Live"
        return f"InteractiveBrokersAPI({mode}, {status})"
