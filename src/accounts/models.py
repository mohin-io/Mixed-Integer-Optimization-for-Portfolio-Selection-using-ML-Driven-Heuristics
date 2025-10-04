"""
SQLAlchemy models for multi-account portfolio management.

This module defines the database models for users, accounts, portfolios,
positions, transactions, and performance tracking.

Models:
- User: User accounts with authentication
- Account: Trading accounts (IB, Alpaca, etc.)
- Portfolio: Investment portfolios with strategies
- Position: Current holdings
- Transaction: Trade history
- PerformanceHistory: Daily snapshots
- RebalanceHistory: Rebalancing records
- Watchlist: Saved ticker lists
- Alert: System alerts
- APIKey: Encrypted broker credentials
- AuditLog: Action audit trail
"""

from sqlalchemy import (
    Column, Integer, String, Numeric, Boolean, DateTime,
    ForeignKey, Text, Date, JSON, ARRAY, CheckConstraint, UniqueConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
from typing import List, Dict, Optional

Base = declarative_base()


class User(Base):
    """User model for authentication and account management."""

    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    username = Column(String(100), unique=True, nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    full_name = Column(String(200))
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    last_login = Column(DateTime)

    # Relationships
    accounts = relationship("Account", back_populates="user", cascade="all, delete-orphan")
    watchlists = relationship("Watchlist", back_populates="user", cascade="all, delete-orphan")
    api_keys = relationship("APIKey", back_populates="user", cascade="all, delete-orphan")

    # Constraints
    __table_args__ = (
        CheckConstraint('LENGTH(username) >= 3', name='username_length'),
    )

    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}', email='{self.email}')>"

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'full_name': self.full_name,
            'is_active': self.is_active,
            'is_admin': self.is_admin,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_login': self.last_login.isoformat() if self.last_login else None
        }


class Account(Base):
    """Trading account model (IB, Alpaca, paper trading, etc.)."""

    __tablename__ = 'accounts'

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    account_name = Column(String(100), nullable=False)
    broker = Column(String(50), nullable=False)
    account_number = Column(String(100))
    account_type = Column(String(50), default='margin')  # margin, cash, ira
    initial_balance = Column(Numeric(15, 2), nullable=False)
    current_balance = Column(Numeric(15, 2), nullable=False)
    currency = Column(String(3), default='USD')
    is_active = Column(Boolean, default=True)
    is_paper = Column(Boolean, default=True)  # paper vs live
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Relationships
    user = relationship("User", back_populates="accounts")
    portfolios = relationship("Portfolio", back_populates="account", cascade="all, delete-orphan")

    # Constraints
    __table_args__ = (
        CheckConstraint('initial_balance >= 0', name='positive_initial_balance'),
        CheckConstraint('current_balance >= 0', name='positive_current_balance'),
        UniqueConstraint('user_id', 'account_name', name='unique_account_per_user'),
    )

    def __repr__(self):
        return f"<Account(id={self.id}, name='{self.account_name}', broker='{self.broker}')>"

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'account_name': self.account_name,
            'broker': self.broker,
            'account_number': self.account_number,
            'account_type': self.account_type,
            'initial_balance': float(self.initial_balance),
            'current_balance': float(self.current_balance),
            'currency': self.currency,
            'is_active': self.is_active,
            'is_paper': self.is_paper,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class Portfolio(Base):
    """Portfolio model with optimization strategy."""

    __tablename__ = 'portfolios'

    id = Column(Integer, primary_key=True)
    account_id = Column(Integer, ForeignKey('accounts.id'), nullable=False)
    portfolio_name = Column(String(100), nullable=False)
    strategy = Column(String(50), nullable=False)  # Max Sharpe, Min Variance, etc.
    risk_tolerance = Column(Numeric(3, 2), default=0.50)
    max_position_size = Column(Numeric(3, 2), default=0.10)
    rebalance_frequency = Column(String(20), default='monthly')
    auto_rebalance = Column(Boolean, default=False)
    description = Column(Text)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Relationships
    account = relationship("Account", back_populates="portfolios")
    positions = relationship("Position", back_populates="portfolio", cascade="all, delete-orphan")
    transactions = relationship("Transaction", back_populates="portfolio", cascade="all, delete-orphan")
    performance_history = relationship("PerformanceHistory", back_populates="portfolio", cascade="all, delete-orphan")
    rebalance_history = relationship("RebalanceHistory", back_populates="portfolio", cascade="all, delete-orphan")
    alerts = relationship("Alert", back_populates="portfolio", cascade="all, delete-orphan")

    # Constraints
    __table_args__ = (
        CheckConstraint('risk_tolerance BETWEEN 0 AND 1', name='valid_risk_tolerance'),
        CheckConstraint('max_position_size BETWEEN 0 AND 1', name='valid_max_position'),
        UniqueConstraint('account_id', 'portfolio_name', name='unique_portfolio_per_account'),
    )

    def __repr__(self):
        return f"<Portfolio(id={self.id}, name='{self.portfolio_name}', strategy='{self.strategy}')>"

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'account_id': self.account_id,
            'portfolio_name': self.portfolio_name,
            'strategy': self.strategy,
            'risk_tolerance': float(self.risk_tolerance),
            'max_position_size': float(self.max_position_size),
            'rebalance_frequency': self.rebalance_frequency,
            'auto_rebalance': self.auto_rebalance,
            'description': self.description,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class Position(Base):
    """Current position holdings."""

    __tablename__ = 'positions'

    id = Column(Integer, primary_key=True)
    portfolio_id = Column(Integer, ForeignKey('portfolios.id'), nullable=False)
    symbol = Column(String(20), nullable=False)
    asset_class = Column(String(50), default='equity')
    quantity = Column(Numeric(15, 4), nullable=False)
    avg_cost = Column(Numeric(15, 4), nullable=False)
    current_price = Column(Numeric(15, 4))
    market_value = Column(Numeric(15, 2))
    unrealized_pnl = Column(Numeric(15, 2))
    realized_pnl = Column(Numeric(15, 2), default=0)
    weight = Column(Numeric(5, 4))  # Portfolio weight
    last_updated = Column(DateTime, default=func.now())

    # Relationships
    portfolio = relationship("Portfolio", back_populates="positions")

    # Constraints
    __table_args__ = (
        UniqueConstraint('portfolio_id', 'symbol', name='unique_position_per_portfolio'),
    )

    def __repr__(self):
        return f"<Position(symbol='{self.symbol}', quantity={self.quantity}, value=${self.market_value})>"

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'portfolio_id': self.portfolio_id,
            'symbol': self.symbol,
            'asset_class': self.asset_class,
            'quantity': float(self.quantity),
            'avg_cost': float(self.avg_cost),
            'current_price': float(self.current_price) if self.current_price else None,
            'market_value': float(self.market_value) if self.market_value else None,
            'unrealized_pnl': float(self.unrealized_pnl) if self.unrealized_pnl else None,
            'realized_pnl': float(self.realized_pnl),
            'weight': float(self.weight) if self.weight else None,
            'last_updated': self.last_updated.isoformat() if self.last_updated else None
        }


class Transaction(Base):
    """Transaction history."""

    __tablename__ = 'transactions'

    id = Column(Integer, primary_key=True)
    portfolio_id = Column(Integer, ForeignKey('portfolios.id'), nullable=False)
    symbol = Column(String(20), nullable=False)
    transaction_type = Column(String(10), nullable=False)  # BUY, SELL, DIVIDEND, FEE
    quantity = Column(Numeric(15, 4), nullable=False)
    price = Column(Numeric(15, 4), nullable=False)
    total_amount = Column(Numeric(15, 2), nullable=False)
    commission = Column(Numeric(10, 2), default=0)
    fees = Column(Numeric(10, 2), default=0)
    transaction_date = Column(DateTime, default=func.now())
    order_id = Column(String(100))
    notes = Column(Text)

    # Relationships
    portfolio = relationship("Portfolio", back_populates="transactions")

    # Constraints
    __table_args__ = (
        CheckConstraint("transaction_type IN ('BUY', 'SELL', 'DIVIDEND', 'FEE', 'SPLIT')", name='valid_transaction_type'),
    )

    def __repr__(self):
        return f"<Transaction({self.transaction_type} {self.quantity} {self.symbol} @ ${self.price})>"

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'portfolio_id': self.portfolio_id,
            'symbol': self.symbol,
            'transaction_type': self.transaction_type,
            'quantity': float(self.quantity),
            'price': float(self.price),
            'total_amount': float(self.total_amount),
            'commission': float(self.commission),
            'fees': float(self.fees),
            'transaction_date': self.transaction_date.isoformat() if self.transaction_date else None,
            'order_id': self.order_id,
            'notes': self.notes
        }


class PerformanceHistory(Base):
    """Daily performance snapshots."""

    __tablename__ = 'performance_history'

    id = Column(Integer, primary_key=True)
    portfolio_id = Column(Integer, ForeignKey('portfolios.id'), nullable=False)
    date = Column(Date, nullable=False)
    total_value = Column(Numeric(15, 2), nullable=False)
    cash_value = Column(Numeric(15, 2))
    positions_value = Column(Numeric(15, 2))
    daily_return = Column(Numeric(10, 6))
    cumulative_return = Column(Numeric(10, 6))
    sharpe_ratio = Column(Numeric(10, 4))
    sortino_ratio = Column(Numeric(10, 4))
    volatility = Column(Numeric(10, 6))
    max_drawdown = Column(Numeric(10, 6))
    num_positions = Column(Integer)

    # Relationships
    portfolio = relationship("Portfolio", back_populates="performance_history")

    # Constraints
    __table_args__ = (
        UniqueConstraint('portfolio_id', 'date', name='unique_portfolio_date'),
    )

    def __repr__(self):
        return f"<PerformanceHistory(date={self.date}, value=${self.total_value})>"

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'portfolio_id': self.portfolio_id,
            'date': self.date.isoformat() if self.date else None,
            'total_value': float(self.total_value),
            'cash_value': float(self.cash_value) if self.cash_value else None,
            'positions_value': float(self.positions_value) if self.positions_value else None,
            'daily_return': float(self.daily_return) if self.daily_return else None,
            'cumulative_return': float(self.cumulative_return) if self.cumulative_return else None,
            'sharpe_ratio': float(self.sharpe_ratio) if self.sharpe_ratio else None,
            'sortino_ratio': float(self.sortino_ratio) if self.sortino_ratio else None,
            'volatility': float(self.volatility) if self.volatility else None,
            'max_drawdown': float(self.max_drawdown) if self.max_drawdown else None,
            'num_positions': self.num_positions
        }


class RebalanceHistory(Base):
    """Portfolio rebalancing records."""

    __tablename__ = 'rebalance_history'

    id = Column(Integer, primary_key=True)
    portfolio_id = Column(Integer, ForeignKey('portfolios.id'), nullable=False)
    rebalance_date = Column(DateTime, default=func.now())
    strategy_used = Column(String(50))
    old_weights = Column(JSON)
    new_weights = Column(JSON)
    trades_executed = Column(Integer, default=0)
    total_cost = Column(Numeric(10, 2), default=0)
    success = Column(Boolean, default=True)
    notes = Column(Text)

    # Relationships
    portfolio = relationship("Portfolio", back_populates="rebalance_history")

    def __repr__(self):
        return f"<RebalanceHistory(date={self.rebalance_date}, trades={self.trades_executed})>"


class Watchlist(Base):
    """User watchlists."""

    __tablename__ = 'watchlists'

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    watchlist_name = Column(String(100), nullable=False)
    symbols = Column(ARRAY(String))
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Relationships
    user = relationship("User", back_populates="watchlists")

    # Constraints
    __table_args__ = (
        UniqueConstraint('user_id', 'watchlist_name', name='unique_watchlist_per_user'),
    )

    def __repr__(self):
        return f"<Watchlist(name='{self.watchlist_name}', symbols={len(self.symbols) if self.symbols else 0})>"


class Alert(Base):
    """System alerts."""

    __tablename__ = 'alerts'

    id = Column(Integer, primary_key=True)
    portfolio_id = Column(Integer, ForeignKey('portfolios.id'))
    alert_type = Column(String(50), nullable=False)
    severity = Column(String(20), nullable=False)  # info, warning, critical
    message = Column(Text, nullable=False)
    metrics = Column(JSON)
    triggered_at = Column(DateTime, default=func.now())
    acknowledged = Column(Boolean, default=False)
    acknowledged_at = Column(DateTime)
    acknowledged_by = Column(Integer, ForeignKey('users.id'))

    # Relationships
    portfolio = relationship("Portfolio", back_populates="alerts")

    def __repr__(self):
        return f"<Alert({self.severity}: {self.alert_type})>"


class APIKey(Base):
    """Encrypted broker API keys."""

    __tablename__ = 'api_keys'

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    broker = Column(String(50), nullable=False)
    key_name = Column(String(100), nullable=False)
    api_key_encrypted = Column(Text, nullable=False)
    api_secret_encrypted = Column(Text)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.now())
    last_used = Column(DateTime)

    # Relationships
    user = relationship("User", back_populates="api_keys")

    # Constraints
    __table_args__ = (
        UniqueConstraint('user_id', 'broker', 'key_name', name='unique_key_per_user'),
    )

    def __repr__(self):
        return f"<APIKey(broker='{self.broker}', name='{self.key_name}')>"


class AuditLog(Base):
    """Audit trail of user actions."""

    __tablename__ = 'audit_log'

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id', ondelete='SET NULL'))
    action = Column(String(100), nullable=False)
    entity_type = Column(String(50))
    entity_id = Column(Integer)
    details = Column(JSON)
    ip_address = Column(String(45))
    user_agent = Column(Text)
    created_at = Column(DateTime, default=func.now())

    def __repr__(self):
        return f"<AuditLog({self.action} by user {self.user_id})>"


# Export all models
__all__ = [
    'Base',
    'User',
    'Account',
    'Portfolio',
    'Position',
    'Transaction',
    'PerformanceHistory',
    'RebalanceHistory',
    'Watchlist',
    'Alert',
    'APIKey',
    'AuditLog'
]
