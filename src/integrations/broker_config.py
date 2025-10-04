"""
Configuration management for broker integrations.

Handles loading and validation of broker connection settings from
environment variables or configuration files.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict
from pathlib import Path


def load_env_file(env_file: str = '.env'):
    """Load environment variables from .env file."""
    env_path = Path(env_file)
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()


# Load .env file if it exists
load_env_file()


@dataclass
class BrokerConfig:
    """Configuration for broker connections."""

    # Interactive Brokers Settings
    ib_host: str = field(default_factory=lambda: os.getenv('IB_HOST', '127.0.0.1'))
    ib_port: int = field(default_factory=lambda: int(os.getenv('IB_PORT', '7497')))
    ib_client_id: int = field(default_factory=lambda: int(os.getenv('IB_CLIENT_ID', '1')))
    ib_paper_trading: bool = field(default_factory=lambda: os.getenv('IB_PAPER_TRADING', 'True').lower() == 'true')

    # Risk Management Settings
    max_position_size: float = field(default_factory=lambda: float(os.getenv('MAX_POSITION_SIZE', '0.1')))
    risk_limit: float = field(default_factory=lambda: float(os.getenv('RISK_LIMIT', '0.02')))
    max_daily_trades: int = field(default_factory=lambda: int(os.getenv('MAX_DAILY_TRADES', '100')))

    # Order Settings
    default_order_type: str = field(default_factory=lambda: os.getenv('DEFAULT_ORDER_TYPE', 'MARKET'))
    enable_short_selling: bool = field(default_factory=lambda: os.getenv('ENABLE_SHORT_SELLING', 'False').lower() == 'true')

    # Data Settings
    default_bar_size: str = field(default_factory=lambda: os.getenv('DEFAULT_BAR_SIZE', '1 day'))
    max_data_points: int = field(default_factory=lambda: int(os.getenv('MAX_DATA_POINTS', '10000')))

    @classmethod
    def from_env(cls, env_file: Optional[str] = None) -> 'BrokerConfig':
        """
        Create configuration from environment variables.

        Args:
            env_file: Optional path to .env file

        Returns:
            BrokerConfig instance

        Example:
            >>> config = BrokerConfig.from_env()
            >>> config = BrokerConfig.from_env('.env.production')
        """
        if env_file:
            load_env_file(env_file)

        return cls()

    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'BrokerConfig':
        """
        Create configuration from dictionary.

        Args:
            config_dict: Dictionary with configuration values

        Returns:
            BrokerConfig instance

        Example:
            >>> config = BrokerConfig.from_dict({
            ...     'ib_host': '127.0.0.1',
            ...     'ib_port': 7497,
            ...     'ib_paper_trading': True
            ... })
        """
        return cls(**config_dict)

    def validate(self) -> bool:
        """
        Validate configuration values.

        Returns:
            True if valid, raises ValueError otherwise

        Raises:
            ValueError: If any configuration value is invalid
        """
        # Validate IB settings
        if not (1024 <= self.ib_port <= 65535):
            raise ValueError(f"IB port must be between 1024 and 65535, got {self.ib_port}")

        if not (0 <= self.ib_client_id <= 999):
            raise ValueError(f"IB client ID must be between 0 and 999, got {self.ib_client_id}")

        # Validate risk management
        if not (0.0 < self.max_position_size <= 1.0):
            raise ValueError(f"Max position size must be between 0 and 1, got {self.max_position_size}")

        if not (0.0 < self.risk_limit <= 1.0):
            raise ValueError(f"Risk limit must be between 0 and 1, got {self.risk_limit}")

        if self.max_daily_trades <= 0:
            raise ValueError(f"Max daily trades must be positive, got {self.max_daily_trades}")

        # Validate order settings
        valid_order_types = ['MARKET', 'LIMIT', 'STOP', 'STOP_LIMIT']
        if self.default_order_type not in valid_order_types:
            raise ValueError(f"Invalid order type: {self.default_order_type}. Must be one of {valid_order_types}")

        return True

    def to_dict(self) -> Dict:
        """
        Convert configuration to dictionary.

        Returns:
            Dictionary representation of configuration
        """
        return {
            'ib_host': self.ib_host,
            'ib_port': self.ib_port,
            'ib_client_id': self.ib_client_id,
            'ib_paper_trading': self.ib_paper_trading,
            'max_position_size': self.max_position_size,
            'risk_limit': self.risk_limit,
            'max_daily_trades': self.max_daily_trades,
            'default_order_type': self.default_order_type,
            'enable_short_selling': self.enable_short_selling,
            'default_bar_size': self.default_bar_size,
            'max_data_points': self.max_data_points
        }

    def __repr__(self) -> str:
        """String representation of configuration."""
        mode = "Paper Trading" if self.ib_paper_trading else "Live Trading"
        return (
            f"BrokerConfig(\n"
            f"  IB: {self.ib_host}:{self.ib_port} ({mode})\n"
            f"  Risk: max_position={self.max_position_size:.1%}, "
            f"risk_limit={self.risk_limit:.1%}\n"
            f"  Orders: type={self.default_order_type}, "
            f"short_selling={self.enable_short_selling}\n"
            f")"
        )


# Default configuration instance
default_config = BrokerConfig.from_env()
