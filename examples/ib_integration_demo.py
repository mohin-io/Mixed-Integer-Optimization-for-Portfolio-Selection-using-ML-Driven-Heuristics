"""
Interactive Brokers Integration Demo

This script demonstrates how to use the Interactive Brokers API integration
for real-time trading and market data.

Prerequisites:
1. Install Interactive Brokers TWS or IB Gateway
2. Enable API connections in TWS/Gateway settings
3. Install ib_insync: pip install ib_insync
4. Configure .env file with IB settings

Usage:
    python examples/ib_integration_demo.py
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.integrations import InteractiveBrokersAPI, BrokerConfig
import pandas as pd
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demo_connection():
    """Demo: Connect and disconnect from IB."""
    print("\n" + "="*80)
    print("DEMO 1: Connection and Disconnection")
    print("="*80)

    # Create configuration for paper trading
    config = BrokerConfig(
        ib_host='127.0.0.1',
        ib_port=7497,  # Paper trading port
        ib_paper_trading=True
    )

    # Initialize API
    broker = InteractiveBrokersAPI(config)

    # Connect
    print("\n📡 Connecting to Interactive Brokers...")
    try:
        broker.connect()
        print(f"✅ Connection Status: {broker.is_connected()}")

        # Disconnect
        print("\n📡 Disconnecting...")
        broker.disconnect()
        print(f"✅ Connection Status: {broker.is_connected()}")

    except Exception as e:
        print(f"❌ Error: {str(e)}")


def demo_account_info():
    """Demo: Get account information."""
    print("\n" + "="*80)
    print("DEMO 2: Account Information")
    print("="*80)

    config = BrokerConfig.from_env()

    with InteractiveBrokersAPI(config) as broker:
        try:
            # Get account balance
            print("\n💰 Account Balance:")
            balance = broker.get_account_balance()
            print(f"   Net Liquidation Value: ${balance:,.2f}")

            # Get account summary
            print("\n📊 Account Summary:")
            summary = broker.get_account_summary()
            for key, value in summary.items():
                if isinstance(value, float):
                    print(f"   {key}: ${value:,.2f}")
                else:
                    print(f"   {key}: {value}")

        except Exception as e:
            print(f"❌ Error: {str(e)}")


def demo_market_data():
    """Demo: Get real-time market data."""
    print("\n" + "="*80)
    print("DEMO 3: Market Data")
    print("="*80)

    config = BrokerConfig.from_env()
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

    with InteractiveBrokersAPI(config) as broker:
        try:
            print(f"\n📈 Getting market data for: {', '.join(symbols)}")
            data = broker.get_market_data(symbols)

            print("\n📊 Real-Time Quotes:")
            print(data.to_string(index=False))

        except Exception as e:
            print(f"❌ Error: {str(e)}")


def demo_positions():
    """Demo: Get current positions."""
    print("\n" + "="*80)
    print("DEMO 4: Current Positions")
    print("="*80)

    config = BrokerConfig.from_env()

    with InteractiveBrokersAPI(config) as broker:
        try:
            print("\n📁 Current Portfolio Positions:")
            positions = broker.get_positions()

            if positions.empty:
                print("   No open positions")
            else:
                print(positions.to_string(index=False))

                # Calculate totals
                total_value = positions['market_value'].sum()
                total_pnl = positions['unrealized_pnl'].sum()

                print(f"\n💼 Portfolio Summary:")
                print(f"   Total Market Value: ${total_value:,.2f}")
                print(f"   Total Unrealized P&L: ${total_pnl:,.2f}")

        except Exception as e:
            print(f"❌ Error: {str(e)}")


def demo_historical_data():
    """Demo: Get historical data."""
    print("\n" + "="*80)
    print("DEMO 5: Historical Data")
    print("="*80)

    config = BrokerConfig.from_env()
    symbol = 'AAPL'

    with InteractiveBrokersAPI(config) as broker:
        try:
            print(f"\n📊 Fetching historical data for {symbol}...")

            from datetime import datetime, timedelta
            end_date = datetime.now().strftime('%Y%m%d %H:%M:%S')
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')

            data = broker.get_historical_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                bar_size='1 day'
            )

            print(f"\n📈 Last 10 days of {symbol}:")
            print(data.tail(10).to_string())

            # Calculate statistics
            print(f"\n📊 Statistics (30 days):")
            print(f"   Average Close: ${data['close'].mean():.2f}")
            print(f"   High: ${data['high'].max():.2f}")
            print(f"   Low: ${data['low'].min():.2f}")
            print(f"   Average Volume: {data['volume'].mean():,.0f}")

        except Exception as e:
            print(f"❌ Error: {str(e)}")


def demo_paper_trading():
    """Demo: Place orders (paper trading only)."""
    print("\n" + "="*80)
    print("DEMO 6: Paper Trading - Place Orders")
    print("="*80)

    config = BrokerConfig.from_env()

    # Safety check - only allow paper trading for this demo
    if not config.ib_paper_trading:
        print("⚠️  This demo only works with paper trading enabled!")
        print("   Set IB_PAPER_TRADING=True in your .env file")
        return

    with InteractiveBrokersAPI(config) as broker:
        try:
            # Get current balance
            balance = broker.get_account_balance()
            print(f"\n💰 Current Balance: ${balance:,.2f}")

            # Place a small market order (10 shares of AAPL)
            print("\n📝 Placing market order: BUY 10 shares of AAPL")
            order_id = broker.place_order(
                symbol='AAPL',
                quantity=10,
                order_type='MARKET'
            )

            if order_id:
                print(f"✅ Order placed successfully! Order ID: {order_id}")

                # Wait a moment for order to process
                time.sleep(2)

                # Check order status
                print(f"\n📊 Checking order status...")
                status = broker.get_order_status(order_id)
                print(f"   Status: {status}")

                # Optional: Cancel the order if it's still open
                if status.get('status') in ['Submitted', 'PreSubmitted']:
                    print(f"\n❌ Cancelling order {order_id}...")
                    cancelled = broker.cancel_order(order_id)
                    if cancelled:
                        print(f"✅ Order cancelled successfully")

            else:
                print("❌ Order placement failed")

        except Exception as e:
            print(f"❌ Error: {str(e)}")


def demo_portfolio_optimization_with_ib():
    """Demo: Optimize portfolio and execute trades."""
    print("\n" + "="*80)
    print("DEMO 7: Portfolio Optimization with IB Execution")
    print("="*80)

    config = BrokerConfig.from_env()

    if not config.ib_paper_trading:
        print("⚠️  This demo only works with paper trading enabled!")
        return

    with InteractiveBrokersAPI(config) as broker:
        try:
            # Get account balance
            balance = broker.get_account_balance()
            print(f"\n💰 Portfolio Value: ${balance:,.2f}")

            # Define target portfolio (example weights)
            target_weights = {
                'AAPL': 0.30,
                'MSFT': 0.25,
                'GOOGL': 0.20,
                'AMZN': 0.15,
                'TSLA': 0.10
            }

            print("\n🎯 Target Portfolio Allocation:")
            for symbol, weight in target_weights.items():
                print(f"   {symbol}: {weight:.1%}")

            # Get current positions
            current_positions = broker.get_positions()
            print("\n📁 Current Positions:")
            if current_positions.empty:
                print("   No current positions - will build from scratch")
            else:
                print(current_positions.to_string(index=False))

            # Calculate required trades
            print("\n📝 Required Trades (based on ${:,.2f} portfolio):".format(balance))

            # Get current market prices
            symbols = list(target_weights.keys())
            market_data = broker.get_market_data(symbols)

            for symbol, target_weight in target_weights.items():
                target_value = balance * target_weight
                price_data = market_data[market_data['symbol'] == symbol]

                if not price_data.empty:
                    current_price = price_data.iloc[0]['last']
                    if current_price > 0:
                        target_shares = int(target_value / current_price)
                        print(f"   {symbol}: Buy {target_shares} shares @ ${current_price:.2f} "
                              f"= ${target_value:,.2f} ({target_weight:.1%})")
                    else:
                        print(f"   {symbol}: Unable to get price")

            print("\n⚠️  Note: Actual trade execution disabled in this demo")
            print("   To execute trades, uncomment the broker.place_order() calls")

        except Exception as e:
            print(f"❌ Error: {str(e)}")


def main():
    """Run all demos."""
    print("\n" + "="*80)
    print("  Interactive Brokers Integration Demo Suite")
    print("="*80)

    print("\n⚠️  IMPORTANT:")
    print("   1. Make sure TWS or IB Gateway is running")
    print("   2. Enable API connections in TWS settings")
    print("   3. Configure .env file with your IB settings")
    print("   4. Use paper trading account for testing")

    input("\nPress Enter to continue...")

    try:
        # Run demos
        demo_connection()
        input("\nPress Enter for next demo...")

        demo_account_info()
        input("\nPress Enter for next demo...")

        demo_market_data()
        input("\nPress Enter for next demo...")

        demo_positions()
        input("\nPress Enter for next demo...")

        demo_historical_data()
        input("\nPress Enter for next demo...")

        demo_paper_trading()
        input("\nPress Enter for next demo...")

        demo_portfolio_optimization_with_ib()

        print("\n" + "="*80)
        print("  Demo Complete!")
        print("="*80)

    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}", exc_info=True)


if __name__ == '__main__':
    main()
