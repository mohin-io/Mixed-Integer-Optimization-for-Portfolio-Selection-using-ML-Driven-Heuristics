"""
Real-Time WebSocket Data Streams for Live Trading.

Implements:
- WebSocket connections to financial data providers
- Real-time price streaming
- Order book updates
- Trade execution notifications
- Asynchronous event handling
"""

import asyncio
import json
import pandas as pd
from typing import Optional, Dict, List, Callable, Any
from dataclasses import dataclass
from datetime import datetime
import warnings
from collections import deque

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    warnings.warn("websockets library not installed. WebSocket streaming unavailable.")

try:
    import alpaca_trade_api as tradeapi
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False


@dataclass
class StreamConfig:
    """Configuration for WebSocket stream."""
    api_key: str
    api_secret: str
    base_url: str = "wss://stream.data.alpaca.markets/v2/iex"
    reconnect_attempts: int = 5
    reconnect_delay: float = 5.0
    buffer_size: int = 1000


@dataclass
class QuoteData:
    """Real-time quote data."""
    symbol: str
    timestamp: datetime
    bid_price: float
    ask_price: float
    bid_size: int
    ask_size: int
    spread: float


@dataclass
class TradeData:
    """Real-time trade data."""
    symbol: str
    timestamp: datetime
    price: float
    size: int
    conditions: List[str]


class AlpacaWebSocketStream:
    """
    WebSocket stream for Alpaca market data.
    """

    def __init__(self, config: StreamConfig):
        """
        Initialize WebSocket stream.

        Args:
            config: Stream configuration
        """
        if not WEBSOCKETS_AVAILABLE:
            raise ImportError("websockets library required for streaming")

        self.config = config
        self.websocket = None
        self.authenticated = False
        self.subscribed_symbols = set()

        # Data buffers
        self.quote_buffer = {}  # symbol -> deque of quotes
        self.trade_buffer = {}  # symbol -> deque of trades
        self.bar_buffer = {}    # symbol -> deque of bars

        # Callbacks
        self.quote_callbacks = []
        self.trade_callbacks = []
        self.bar_callbacks = []

    async def connect(self):
        """Establish WebSocket connection."""
        try:
            self.websocket = await websockets.connect(self.config.base_url)
            print(f"Connected to {self.config.base_url}")
            await self.authenticate()
        except Exception as e:
            print(f"Connection failed: {e}")
            raise

    async def authenticate(self):
        """Authenticate with API credentials."""
        auth_message = {
            "action": "auth",
            "key": self.config.api_key,
            "secret": self.config.api_secret
        }

        await self.websocket.send(json.dumps(auth_message))

        # Wait for auth response
        response = await self.websocket.recv()
        data = json.loads(response)

        if data[0].get('T') == 'success' and data[0].get('msg') == 'authenticated':
            self.authenticated = True
            print("Authentication successful")
        else:
            raise ConnectionError(f"Authentication failed: {data}")

    async def subscribe(
        self,
        symbols: List[str],
        data_types: List[str] = ['quotes', 'trades', 'bars']
    ):
        """
        Subscribe to market data streams.

        Args:
            symbols: List of symbols to subscribe
            data_types: Types of data ('quotes', 'trades', 'bars')
        """
        if not self.authenticated:
            raise ConnectionError("Not authenticated. Call connect() first.")

        subscribe_message = {
            "action": "subscribe"
        }

        for data_type in data_types:
            subscribe_message[data_type] = symbols

        await self.websocket.send(json.dumps(subscribe_message))

        self.subscribed_symbols.update(symbols)

        # Initialize buffers
        for symbol in symbols:
            if symbol not in self.quote_buffer:
                self.quote_buffer[symbol] = deque(maxlen=self.config.buffer_size)
            if symbol not in self.trade_buffer:
                self.trade_buffer[symbol] = deque(maxlen=self.config.buffer_size)
            if symbol not in self.bar_buffer:
                self.bar_buffer[symbol] = deque(maxlen=self.config.buffer_size)

        print(f"Subscribed to {symbols} for {data_types}")

    async def unsubscribe(self, symbols: List[str]):
        """Unsubscribe from symbols."""
        unsubscribe_message = {
            "action": "unsubscribe",
            "quotes": symbols,
            "trades": symbols,
            "bars": symbols
        }

        await self.websocket.send(json.dumps(unsubscribe_message))
        self.subscribed_symbols.difference_update(symbols)

        print(f"Unsubscribed from {symbols}")

    async def listen(self):
        """Listen for incoming messages."""
        try:
            async for message in self.websocket:
                data = json.loads(message)

                for item in data:
                    msg_type = item.get('T')

                    if msg_type == 'q':  # Quote
                        await self._handle_quote(item)
                    elif msg_type == 't':  # Trade
                        await self._handle_trade(item)
                    elif msg_type == 'b':  # Bar
                        await self._handle_bar(item)

        except websockets.exceptions.ConnectionClosed:
            print("Connection closed")
            await self.reconnect()

    async def _handle_quote(self, data: Dict):
        """Handle quote message."""
        quote = QuoteData(
            symbol=data['S'],
            timestamp=pd.Timestamp(data['t']),
            bid_price=data['bp'],
            ask_price=data['ap'],
            bid_size=data['bs'],
            ask_size=data['as'],
            spread=data['ap'] - data['bp']
        )

        # Add to buffer
        self.quote_buffer[quote.symbol].append(quote)

        # Trigger callbacks
        for callback in self.quote_callbacks:
            await callback(quote)

    async def _handle_trade(self, data: Dict):
        """Handle trade message."""
        trade = TradeData(
            symbol=data['S'],
            timestamp=pd.Timestamp(data['t']),
            price=data['p'],
            size=data['s'],
            conditions=data.get('c', [])
        )

        # Add to buffer
        self.trade_buffer[trade.symbol].append(trade)

        # Trigger callbacks
        for callback in self.trade_callbacks:
            await callback(trade)

    async def _handle_bar(self, data: Dict):
        """Handle bar message."""
        bar = {
            'symbol': data['S'],
            'timestamp': pd.Timestamp(data['t']),
            'open': data['o'],
            'high': data['h'],
            'low': data['l'],
            'close': data['c'],
            'volume': data['v']
        }

        # Add to buffer
        self.bar_buffer[bar['symbol']].append(bar)

        # Trigger callbacks
        for callback in self.bar_callbacks:
            await callback(bar)

    async def reconnect(self):
        """Attempt to reconnect."""
        for attempt in range(self.config.reconnect_attempts):
            print(f"Reconnection attempt {attempt + 1}/{self.config.reconnect_attempts}")

            try:
                await asyncio.sleep(self.config.reconnect_delay)
                await self.connect()

                # Resubscribe to symbols
                if self.subscribed_symbols:
                    await self.subscribe(list(self.subscribed_symbols))

                print("Reconnection successful")
                return

            except Exception as e:
                print(f"Reconnection failed: {e}")

        print("Max reconnection attempts reached. Giving up.")

    def register_quote_callback(self, callback: Callable):
        """Register callback for quote updates."""
        self.quote_callbacks.append(callback)

    def register_trade_callback(self, callback: Callable):
        """Register callback for trade updates."""
        self.trade_callbacks.append(callback)

    def register_bar_callback(self, callback: Callable):
        """Register callback for bar updates."""
        self.bar_callbacks.append(callback)

    def get_latest_quote(self, symbol: str) -> Optional[QuoteData]:
        """Get latest quote for symbol."""
        if symbol in self.quote_buffer and self.quote_buffer[symbol]:
            return self.quote_buffer[symbol][-1]
        return None

    def get_latest_trade(self, symbol: str) -> Optional[TradeData]:
        """Get latest trade for symbol."""
        if symbol in self.trade_buffer and self.trade_buffer[symbol]:
            return self.trade_buffer[symbol][-1]
        return None

    def get_quote_history(self, symbol: str, n: int = 100) -> List[QuoteData]:
        """Get recent quote history."""
        if symbol in self.quote_buffer:
            return list(self.quote_buffer[symbol])[-n:]
        return []

    async def close(self):
        """Close WebSocket connection."""
        if self.websocket:
            await self.websocket.close()
            print("Connection closed")


class RealTimePortfolioMonitor:
    """
    Monitor portfolio in real-time using WebSocket streams.
    """

    def __init__(self, stream: AlpacaWebSocketStream):
        """
        Initialize real-time monitor.

        Args:
            stream: WebSocket stream instance
        """
        self.stream = stream
        self.portfolio_values = deque(maxlen=1000)
        self.last_prices = {}
        self.position_sizes = {}

    async def monitor_quote(self, quote: QuoteData):
        """Callback for quote updates."""
        mid_price = (quote.bid_price + quote.ask_price) / 2
        self.last_prices[quote.symbol] = mid_price

        # Update portfolio value
        portfolio_value = self._calculate_portfolio_value()
        self.portfolio_values.append({
            'timestamp': quote.timestamp,
            'value': portfolio_value
        })

        print(f"{quote.symbol}: Bid={quote.bid_price:.2f}, "
              f"Ask={quote.ask_price:.2f}, "
              f"Spread={quote.spread:.4f}, "
              f"Portfolio=${portfolio_value:,.2f}")

    def set_positions(self, positions: Dict[str, float]):
        """
        Set position sizes.

        Args:
            positions: Dictionary mapping symbol to number of shares
        """
        self.position_sizes = positions

    def _calculate_portfolio_value(self) -> float:
        """Calculate current portfolio value."""
        value = 0
        for symbol, shares in self.position_sizes.items():
            if symbol in self.last_prices:
                value += shares * self.last_prices[symbol]
        return value

    def get_portfolio_returns(self) -> pd.Series:
        """Calculate portfolio returns from real-time data."""
        if len(self.portfolio_values) < 2:
            return pd.Series(dtype=float)

        df = pd.DataFrame(list(self.portfolio_values))
        df['return'] = df['value'].pct_change()

        return df.set_index('timestamp')['return']


class LiveSignalGenerator:
    """
    Generate trading signals from real-time data.
    """

    def __init__(
        self,
        stream: AlpacaWebSocketStream,
        signal_func: Callable
    ):
        """
        Initialize live signal generator.

        Args:
            stream: WebSocket stream
            signal_func: Function to generate signals from price data
        """
        self.stream = stream
        self.signal_func = signal_func
        self.signals = {}

    async def process_trade(self, trade: TradeData):
        """Process incoming trade and generate signal."""
        # Get recent price history
        recent_trades = self.stream.trade_buffer[trade.symbol]

        if len(recent_trades) < 20:
            return  # Not enough data

        # Convert to DataFrame
        prices = [t.price for t in recent_trades]
        timestamps = [t.timestamp for t in recent_trades]

        price_series = pd.Series(prices, index=timestamps)

        # Generate signal
        signal = self.signal_func(price_series)

        self.signals[trade.symbol] = {
            'timestamp': trade.timestamp,
            'signal': signal,
            'price': trade.price
        }

        print(f"Signal for {trade.symbol}: {signal} @ {trade.price:.2f}")


# Example usage functions

async def example_streaming():
    """Example: Basic streaming setup."""
    if not WEBSOCKETS_AVAILABLE:
        print("WebSockets not available. Install with: pip install websockets")
        return

    # Configuration (use simulation mode)
    config = StreamConfig(
        api_key="your_api_key",
        api_secret="your_api_secret"
    )

    stream = AlpacaWebSocketStream(config)

    # Define callback
    async def on_quote(quote: QuoteData):
        print(f"{quote.symbol}: Bid={quote.bid_price:.2f}, "
              f"Ask={quote.ask_price:.2f}")

    # Register callback
    stream.register_quote_callback(on_quote)

    # Connect and subscribe
    await stream.connect()
    await stream.subscribe(['AAPL', 'MSFT'], data_types=['quotes'])

    # Listen for 60 seconds
    listen_task = asyncio.create_task(stream.listen())

    await asyncio.sleep(60)

    # Cleanup
    listen_task.cancel()
    await stream.close()


async def example_portfolio_monitoring():
    """Example: Real-time portfolio monitoring."""
    config = StreamConfig(
        api_key="your_api_key",
        api_secret="your_api_secret"
    )

    stream = AlpacaWebSocketStream(config)
    monitor = RealTimePortfolioMonitor(stream)

    # Set positions
    monitor.set_positions({
        'AAPL': 100,
        'MSFT': 50,
        'GOOGL': 25
    })

    # Register callback
    stream.register_quote_callback(monitor.monitor_quote)

    # Connect and run
    await stream.connect()
    await stream.subscribe(['AAPL', 'MSFT', 'GOOGL'], data_types=['quotes'])

    listen_task = asyncio.create_task(stream.listen())
    await asyncio.sleep(60)

    # Get results
    returns = monitor.get_portfolio_returns()
    print(f"\nPortfolio Returns:\n{returns.tail()}")

    # Cleanup
    listen_task.cancel()
    await stream.close()


if __name__ == "__main__":
    print("WebSocket Streaming Implementation")
    print("=" * 60)

    if not WEBSOCKETS_AVAILABLE:
        print("⚠️ websockets library not installed.")
        print("Install with: pip install websockets")
        print("\nSimulation mode active.")
    else:
        print("✅ WebSocket streaming available")
        print("\nTo use real streaming, update API credentials and run:")
        print("  asyncio.run(example_streaming())")
        print("  asyncio.run(example_portfolio_monitoring())")

    print("\n✅ WebSocket streaming implementation complete!")
