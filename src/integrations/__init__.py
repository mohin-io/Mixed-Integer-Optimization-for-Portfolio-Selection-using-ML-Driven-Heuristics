"""
Broker integration modules for real-time trading.

This package provides interfaces and implementations for connecting to
various brokers for live trading, market data, and account management.

Supported Brokers:
- Interactive Brokers (via ib_insync)
- More coming soon...
"""

from .broker_interface import BrokerInterface
from .broker_config import BrokerConfig

__all__ = ['BrokerInterface', 'BrokerConfig']
