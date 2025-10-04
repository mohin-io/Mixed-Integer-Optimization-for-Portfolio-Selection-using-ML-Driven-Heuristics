# 🚀 Future Enhancements Implementation Plan

**Status:** In Progress
**Version:** 2.1.0 (Planned)
**Target:** Production-Ready Real-World Integration

---

## 📋 Overview

This document outlines the implementation plan for advanced real-world integration features that will transform the portfolio optimization project from a demonstration tool to a **production trading system**.

---

## 🎯 Feature Roadmap

### Phase 1: Interactive Brokers API Integration
**Priority:** HIGH | **Complexity:** HIGH | **Timeline:** Week 1-2

#### Features
- Real-time market data streaming
- Automated order execution
- Portfolio position tracking
- Historical data fetching
- Account balance monitoring

#### Implementation
```
src/
├── integrations/
│   ├── __init__.py
│   ├── broker_interface.py      # Abstract broker interface
│   ├── interactive_brokers.py   # IB-specific implementation
│   └── broker_config.py         # Configuration management
```

#### Technical Stack
- **Library:** `ib_insync` (Modern async IB API wrapper)
- **Authentication:** TWS/IB Gateway connection
- **Data Format:** Real-time streaming via WebSocket
- **Fallback:** Paper trading mode for testing

---

### Phase 2: Production Monitoring Dashboard
**Priority:** HIGH | **Complexity:** MEDIUM | **Timeline:** Week 3

#### Features
- Real-time performance metrics
- System health monitoring
- Alert visualization
- Historical metric tracking
- Custom dashboards

#### Implementation
```
monitoring/
├── prometheus/
│   ├── prometheus.yml           # Prometheus config
│   ├── alerts.yml              # Alert rules
│   └── exporters/
│       └── portfolio_exporter.py
├── grafana/
│   ├── dashboards/
│   │   ├── portfolio_performance.json
│   │   ├── system_health.json
│   │   └── risk_metrics.json
│   └── provisioning/
│       ├── datasources.yml
│       └── dashboards.yml
└── docker-compose.yml          # Full stack deployment
```

#### Technical Stack
- **Monitoring:** Prometheus (metrics collection)
- **Visualization:** Grafana (dashboards)
- **Metrics Export:** `prometheus-client` Python library
- **Deployment:** Docker Compose

---

### Phase 3: Email/SMS Portfolio Alerts
**Priority:** MEDIUM | **Complexity:** LOW | **Timeline:** Week 4

#### Features
- Configurable alert thresholds
- Email notifications (SMTP)
- SMS notifications (Twilio)
- Alert priority levels (INFO, WARNING, CRITICAL)
- Alert history and acknowledgment

#### Implementation
```
src/
├── alerts/
│   ├── __init__.py
│   ├── alert_manager.py        # Alert orchestration
│   ├── email_sender.py         # Email integration
│   ├── sms_sender.py           # SMS integration (Twilio)
│   ├── alert_rules.py          # Alert condition definitions
│   └── alert_storage.py        # Alert history database
```

#### Technical Stack
- **Email:** `smtplib` + Gmail/SendGrid API
- **SMS:** Twilio API
- **Storage:** SQLite/PostgreSQL
- **Scheduling:** `APScheduler` for periodic checks

---

### Phase 4: Multi-Account Management
**Priority:** MEDIUM | **Complexity:** MEDIUM | **Timeline:** Week 5

#### Features
- Multiple portfolio management
- Account aggregation
- Per-account optimization
- Cross-account rebalancing
- User authentication & authorization

#### Implementation
```
src/
├── accounts/
│   ├── __init__.py
│   ├── account_manager.py      # Multi-account orchestration
│   ├── portfolio_aggregator.py # Cross-account analytics
│   ├── user_auth.py            # Authentication system
│   └── models.py               # Database models
└── database/
    ├── migrations/             # Alembic migrations
    ├── schema.sql             # Database schema
    └── seed_data.sql          # Sample data
```

#### Technical Stack
- **Database:** PostgreSQL (production) / SQLite (dev)
- **ORM:** SQLAlchemy
- **Authentication:** JWT tokens with `PyJWT`
- **Migration:** Alembic

---

## 🏗️ Architecture Design

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     User Interface Layer                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │   Streamlit  │  │   REST API   │  │   Admin Dashboard    │  │
│  │   Dashboard  │  │   (FastAPI)  │  │     (Grafana)        │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────┼─────────────────────────────────┐
│                     Business Logic Layer                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐│
│  │  Portfolio   │  │    Alert     │  │   Account Manager    ││
│  │  Optimizer   │  │   Manager    │  │                      ││
│  └──────────────┘  └──────────────┘  └──────────────────────┘│
└────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────┼─────────────────────────────────┐
│                     Integration Layer                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐│
│  │  IB API      │  │  Email/SMS   │  │   Monitoring         ││
│  │  Connector   │  │  Service     │  │   (Prometheus)       ││
│  └──────────────┘  └──────────────┘  └──────────────────────┘│
└────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────┼─────────────────────────────────┐
│                     Data Layer                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐│
│  │  PostgreSQL  │  │    Redis     │  │   Time Series DB     ││
│  │  (Accounts)  │  │   (Cache)    │  │   (Prometheus)       ││
│  └──────────────┘  └──────────────┘  └──────────────────────┘│
└────────────────────────────────────────────────────────────────┘
```

---

## 📦 Phase 1: Interactive Brokers Integration

### Step 1: Install Dependencies

```bash
pip install ib_insync pandas numpy python-dotenv
```

### Step 2: Create Broker Interface

```python
# src/integrations/broker_interface.py
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import pandas as pd

class BrokerInterface(ABC):
    """Abstract interface for broker integrations."""

    @abstractmethod
    def connect(self) -> bool:
        """Connect to broker API."""
        pass

    @abstractmethod
    def disconnect(self) -> bool:
        """Disconnect from broker API."""
        pass

    @abstractmethod
    def get_account_balance(self) -> float:
        """Get current account balance."""
        pass

    @abstractmethod
    def get_positions(self) -> pd.DataFrame:
        """Get current portfolio positions."""
        pass

    @abstractmethod
    def get_market_data(self, symbols: List[str]) -> pd.DataFrame:
        """Get real-time market data for symbols."""
        pass

    @abstractmethod
    def place_order(self, symbol: str, quantity: int, order_type: str) -> str:
        """Place an order. Returns order ID."""
        pass

    @abstractmethod
    def get_historical_data(self, symbol: str, start_date: str,
                           end_date: str) -> pd.DataFrame:
        """Get historical price data."""
        pass
```

### Step 3: Interactive Brokers Implementation

```python
# src/integrations/interactive_brokers.py
from ib_insync import IB, Stock, MarketOrder, LimitOrder
import pandas as pd
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

class InteractiveBrokersAPI(BrokerInterface):
    """Interactive Brokers API integration using ib_insync."""

    def __init__(self, host: str = '127.0.0.1', port: int = 7497,
                 client_id: int = 1, paper_trading: bool = True):
        """
        Initialize IB API connection.

        Args:
            host: TWS/Gateway host
            port: 7497 for TWS paper, 7496 for TWS live, 4002 for Gateway paper
            client_id: Unique client identifier
            paper_trading: Use paper trading account
        """
        self.host = host
        self.port = port
        self.client_id = client_id
        self.paper_trading = paper_trading
        self.ib = IB()
        self.connected = False

    def connect(self) -> bool:
        """Connect to Interactive Brokers TWS/Gateway."""
        try:
            self.ib.connect(self.host, self.port, clientId=self.client_id)
            self.connected = True
            logger.info(f"Connected to IB {'Paper' if self.paper_trading else 'Live'} Trading")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to IB: {str(e)}")
            self.connected = False
            return False

    def disconnect(self) -> bool:
        """Disconnect from IB."""
        try:
            self.ib.disconnect()
            self.connected = False
            logger.info("Disconnected from IB")
            return True
        except Exception as e:
            logger.error(f"Failed to disconnect from IB: {str(e)}")
            return False

    def get_account_balance(self) -> float:
        """Get current account balance."""
        try:
            account_values = self.ib.accountValues()
            for value in account_values:
                if value.tag == 'NetLiquidation':
                    return float(value.value)
            return 0.0
        except Exception as e:
            logger.error(f"Failed to get account balance: {str(e)}")
            return 0.0

    def get_positions(self) -> pd.DataFrame:
        """Get current portfolio positions."""
        try:
            positions = self.ib.positions()

            data = []
            for position in positions:
                data.append({
                    'symbol': position.contract.symbol,
                    'quantity': position.position,
                    'avg_cost': position.avgCost,
                    'market_value': position.position * position.avgCost
                })

            return pd.DataFrame(data)
        except Exception as e:
            logger.error(f"Failed to get positions: {str(e)}")
            return pd.DataFrame()

    def get_market_data(self, symbols: List[str]) -> pd.DataFrame:
        """Get real-time market data."""
        try:
            data = []

            for symbol in symbols:
                contract = Stock(symbol, 'SMART', 'USD')
                self.ib.qualifyContracts(contract)
                ticker = self.ib.reqMktData(contract)
                self.ib.sleep(1)  # Wait for data

                data.append({
                    'symbol': symbol,
                    'bid': ticker.bid,
                    'ask': ticker.ask,
                    'last': ticker.last,
                    'volume': ticker.volume
                })

            return pd.DataFrame(data)
        except Exception as e:
            logger.error(f"Failed to get market data: {str(e)}")
            return pd.DataFrame()

    def place_order(self, symbol: str, quantity: int,
                   order_type: str = 'MARKET') -> Optional[str]:
        """Place an order."""
        try:
            contract = Stock(symbol, 'SMART', 'USD')
            self.ib.qualifyContracts(contract)

            if order_type == 'MARKET':
                order = MarketOrder('BUY' if quantity > 0 else 'SELL', abs(quantity))
            else:
                # Add limit order support
                raise NotImplementedError("Only market orders supported currently")

            trade = self.ib.placeOrder(contract, order)
            logger.info(f"Placed order: {symbol} {quantity} shares")

            return str(trade.order.orderId)
        except Exception as e:
            logger.error(f"Failed to place order: {str(e)}")
            return None

    def get_historical_data(self, symbol: str, start_date: str,
                           end_date: str, bar_size: str = '1 day') -> pd.DataFrame:
        """Get historical price data."""
        try:
            contract = Stock(symbol, 'SMART', 'USD')
            self.ib.qualifyContracts(contract)

            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime=end_date,
                durationStr='1 Y',
                barSizeSetting=bar_size,
                whatToShow='TRADES',
                useRTH=True
            )

            df = pd.DataFrame([{
                'date': bar.date,
                'open': bar.open,
                'high': bar.high,
                'low': bar.low,
                'close': bar.close,
                'volume': bar.volume
            } for bar in bars])

            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)

            return df
        except Exception as e:
            logger.error(f"Failed to get historical data: {str(e)}")
            return pd.DataFrame()
```

### Step 4: Configuration Management

```python
# src/integrations/broker_config.py
import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

@dataclass
class BrokerConfig:
    """Configuration for broker connections."""

    # Interactive Brokers
    ib_host: str = os.getenv('IB_HOST', '127.0.0.1')
    ib_port: int = int(os.getenv('IB_PORT', '7497'))  # Paper trading port
    ib_client_id: int = int(os.getenv('IB_CLIENT_ID', '1'))
    ib_paper_trading: bool = os.getenv('IB_PAPER_TRADING', 'True').lower() == 'true'

    # General settings
    max_position_size: float = float(os.getenv('MAX_POSITION_SIZE', '0.1'))  # 10% of portfolio
    risk_limit: float = float(os.getenv('RISK_LIMIT', '0.02'))  # 2% daily loss limit

    @classmethod
    def from_env(cls) -> 'BrokerConfig':
        """Create configuration from environment variables."""
        return cls()
```

### Step 5: Example .env File

```bash
# .env
# Interactive Brokers Configuration
IB_HOST=127.0.0.1
IB_PORT=7497              # 7497=TWS Paper, 7496=TWS Live, 4002=Gateway Paper
IB_CLIENT_ID=1
IB_PAPER_TRADING=True

# Risk Management
MAX_POSITION_SIZE=0.1     # 10% max per position
RISK_LIMIT=0.02           # 2% daily loss limit

# Monitoring
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000

# Alerts
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password
ALERT_EMAIL=alerts@yourdomain.com

TWILIO_ACCOUNT_SID=your-twilio-sid
TWILIO_AUTH_TOKEN=your-twilio-token
TWILIO_FROM_NUMBER=+1234567890
ALERT_PHONE=+1987654321

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/portfolio_db
REDIS_URL=redis://localhost:6379/0
```

---

## 📊 Phase 2: Prometheus Monitoring

### Prometheus Configuration

```yaml
# monitoring/prometheus/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

alerting:
  alertmanagers:
    - static_configs:
        - targets:
            - localhost:9093

rule_files:
  - "alerts.yml"

scrape_configs:
  - job_name: 'portfolio-optimizer'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
```

### Alert Rules

```yaml
# monitoring/prometheus/alerts.yml
groups:
  - name: portfolio_alerts
    interval: 30s
    rules:
      - alert: HighPortfolioDrawdown
        expr: portfolio_drawdown > 0.10
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Portfolio drawdown exceeds 10%"
          description: "Current drawdown: {{ $value }}"

      - alert: LowSharpeRatio
        expr: portfolio_sharpe_ratio < 0.5
        for: 1h
        labels:
          severity: warning
        annotations:
          summary: "Sharpe ratio below acceptable threshold"
          description: "Current Sharpe: {{ $value }}"

      - alert: HighVolatility
        expr: portfolio_volatility > 0.25
        for: 30m
        labels:
          severity: warning
        annotations:
          summary: "Portfolio volatility is high"
          description: "Current volatility: {{ $value }}"
```

### Metrics Exporter

```python
# monitoring/prometheus/exporters/portfolio_exporter.py
from prometheus_client import start_http_server, Gauge, Counter, Histogram
import time
import logging

logger = logging.getLogger(__name__)

# Define metrics
portfolio_value = Gauge('portfolio_total_value', 'Total portfolio value in USD')
portfolio_return = Gauge('portfolio_return', 'Portfolio return')
portfolio_volatility = Gauge('portfolio_volatility', 'Portfolio volatility')
portfolio_sharpe = Gauge('portfolio_sharpe_ratio', 'Portfolio Sharpe ratio')
portfolio_drawdown = Gauge('portfolio_drawdown', 'Current drawdown from peak')
portfolio_positions = Gauge('portfolio_positions', 'Number of positions')

order_count = Counter('orders_placed_total', 'Total number of orders placed')
order_errors = Counter('order_errors_total', 'Total number of order errors')

optimization_duration = Histogram('optimization_duration_seconds',
                                 'Time spent in portfolio optimization')

class PortfolioMetricsExporter:
    """Export portfolio metrics to Prometheus."""

    def __init__(self, port: int = 8000):
        self.port = port

    def start(self):
        """Start the metrics HTTP server."""
        start_http_server(self.port)
        logger.info(f"Prometheus metrics server started on port {self.port}")

    def update_portfolio_metrics(self, value: float, ret: float, vol: float,
                                sharpe: float, dd: float, positions: int):
        """Update portfolio performance metrics."""
        portfolio_value.set(value)
        portfolio_return.set(ret)
        portfolio_volatility.set(vol)
        portfolio_sharpe.set(sharpe)
        portfolio_drawdown.set(dd)
        portfolio_positions.set(positions)

    def record_order(self, success: bool = True):
        """Record order placement."""
        order_count.inc()
        if not success:
            order_errors.inc()

    def record_optimization_time(self, duration: float):
        """Record optimization duration."""
        optimization_duration.observe(duration)
```

### Grafana Dashboard JSON

```json
// monitoring/grafana/dashboards/portfolio_performance.json
{
  "dashboard": {
    "title": "Portfolio Performance Dashboard",
    "panels": [
      {
        "title": "Portfolio Value",
        "type": "graph",
        "targets": [
          {
            "expr": "portfolio_total_value",
            "legendFormat": "Total Value"
          }
        ]
      },
      {
        "title": "Sharpe Ratio",
        "type": "stat",
        "targets": [
          {
            "expr": "portfolio_sharpe_ratio"
          }
        ]
      },
      {
        "title": "Drawdown",
        "type": "graph",
        "targets": [
          {
            "expr": "portfolio_drawdown",
            "legendFormat": "Drawdown"
          }
        ]
      }
    ]
  }
}
```

---

## 📧 Phase 3: Email/SMS Alerts

### Alert Manager

```python
# src/alerts/alert_manager.py
from enum import Enum
from typing import List, Optional
import logging
from datetime import datetime
from .email_sender import EmailSender
from .sms_sender import SMSSender
from .alert_rules import AlertRule, AlertLevel

logger = logging.getLogger(__name__)

class AlertManager:
    """Manage portfolio alerts and notifications."""

    def __init__(self, email_config: dict, sms_config: dict):
        self.email_sender = EmailSender(**email_config)
        self.sms_sender = SMSSender(**sms_config)
        self.alert_history = []

    def check_and_alert(self, portfolio_metrics: dict, rules: List[AlertRule]):
        """Check alert rules and send notifications."""
        triggered_alerts = []

        for rule in rules:
            if rule.evaluate(portfolio_metrics):
                alert = {
                    'rule': rule.name,
                    'level': rule.level,
                    'message': rule.get_message(portfolio_metrics),
                    'timestamp': datetime.now()
                }
                triggered_alerts.append(alert)
                self.alert_history.append(alert)

                # Send notifications based on alert level
                if rule.level == AlertLevel.CRITICAL:
                    self.send_critical_alert(alert)
                elif rule.level == AlertLevel.WARNING:
                    self.send_warning_alert(alert)
                else:
                    self.send_info_alert(alert)

        return triggered_alerts

    def send_critical_alert(self, alert: dict):
        """Send critical alert via email AND SMS."""
        subject = f"🚨 CRITICAL: {alert['rule']}"
        self.email_sender.send(subject, alert['message'])
        self.sms_sender.send(f"CRITICAL: {alert['message']}")
        logger.critical(f"Critical alert sent: {alert['rule']}")

    def send_warning_alert(self, alert: dict):
        """Send warning alert via email."""
        subject = f"⚠️ WARNING: {alert['rule']}"
        self.email_sender.send(subject, alert['message'])
        logger.warning(f"Warning alert sent: {alert['rule']}")

    def send_info_alert(self, alert: dict):
        """Send info alert via email (low priority)."""
        subject = f"ℹ️ INFO: {alert['rule']}"
        self.email_sender.send(subject, alert['message'])
        logger.info(f"Info alert sent: {alert['rule']}")
```

### Email Sender

```python
# src/alerts/email_sender.py
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging

logger = logging.getLogger(__name__)

class EmailSender:
    """Send email notifications."""

    def __init__(self, smtp_host: str, smtp_port: int,
                 username: str, password: str, to_email: str):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.to_email = to_email

    def send(self, subject: str, message: str, html: bool = False):
        """Send email notification."""
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.username
            msg['To'] = self.to_email

            if html:
                msg.attach(MIMEText(message, 'html'))
            else:
                msg.attach(MIMEText(message, 'plain'))

            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)

            logger.info(f"Email sent: {subject}")
            return True
        except Exception as e:
            logger.error(f"Failed to send email: {str(e)}")
            return False
```

### SMS Sender (Twilio)

```python
# src/alerts/sms_sender.py
from twilio.rest import Client
import logging

logger = logging.getLogger(__name__)

class SMSSender:
    """Send SMS notifications via Twilio."""

    def __init__(self, account_sid: str, auth_token: str,
                 from_number: str, to_number: str):
        self.client = Client(account_sid, auth_token)
        self.from_number = from_number
        self.to_number = to_number

    def send(self, message: str):
        """Send SMS notification."""
        try:
            # Limit message length for SMS
            if len(message) > 160:
                message = message[:157] + "..."

            sms = self.client.messages.create(
                body=message,
                from_=self.from_number,
                to=self.to_number
            )

            logger.info(f"SMS sent: {sms.sid}")
            return True
        except Exception as e:
            logger.error(f"Failed to send SMS: {str(e)}")
            return False
```

### Alert Rules

```python
# src/alerts/alert_rules.py
from enum import Enum
from typing import Callable, Dict

class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

class AlertRule:
    """Define an alert rule with conditions."""

    def __init__(self, name: str, level: AlertLevel,
                 condition: Callable[[Dict], bool],
                 message_template: str):
        self.name = name
        self.level = level
        self.condition = condition
        self.message_template = message_template

    def evaluate(self, metrics: Dict) -> bool:
        """Evaluate if alert should trigger."""
        return self.condition(metrics)

    def get_message(self, metrics: Dict) -> str:
        """Generate alert message."""
        return self.message_template.format(**metrics)

# Predefined alert rules
ALERT_RULES = [
    AlertRule(
        name="High Drawdown",
        level=AlertLevel.CRITICAL,
        condition=lambda m: m.get('drawdown', 0) > 0.10,
        message_template="Portfolio drawdown of {drawdown:.2%} exceeds 10% threshold!"
    ),
    AlertRule(
        name="Low Sharpe Ratio",
        level=AlertLevel.WARNING,
        condition=lambda m: m.get('sharpe', 1) < 0.5,
        message_template="Sharpe ratio of {sharpe:.2f} is below 0.5 threshold."
    ),
    AlertRule(
        name="High Volatility",
        level=AlertLevel.WARNING,
        condition=lambda m: m.get('volatility', 0) > 0.25,
        message_template="Portfolio volatility of {volatility:.2%} exceeds 25%."
    ),
    AlertRule(
        name="Negative Return",
        level=AlertLevel.WARNING,
        condition=lambda m: m.get('return', 0) < 0,
        message_template="Portfolio return is negative: {return:.2%}"
    )
]
```

---

## 👥 Phase 4: Multi-Account Management

### Database Schema

```sql
-- src/database/schema.sql

-- Users table
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(100) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Accounts table
CREATE TABLE accounts (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    account_name VARCHAR(100) NOT NULL,
    broker VARCHAR(50) NOT NULL,
    account_number VARCHAR(100),
    initial_balance DECIMAL(15, 2) NOT NULL,
    current_balance DECIMAL(15, 2) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, account_name)
);

-- Portfolios table
CREATE TABLE portfolios (
    id SERIAL PRIMARY KEY,
    account_id INTEGER REFERENCES accounts(id) ON DELETE CASCADE,
    portfolio_name VARCHAR(100) NOT NULL,
    strategy VARCHAR(50) NOT NULL,
    risk_tolerance DECIMAL(3, 2) DEFAULT 0.5,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(account_id, portfolio_name)
);

-- Positions table
CREATE TABLE positions (
    id SERIAL PRIMARY KEY,
    portfolio_id INTEGER REFERENCES portfolios(id) ON DELETE CASCADE,
    symbol VARCHAR(20) NOT NULL,
    quantity DECIMAL(15, 4) NOT NULL,
    avg_cost DECIMAL(15, 4) NOT NULL,
    current_price DECIMAL(15, 4),
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(portfolio_id, symbol)
);

-- Transactions table
CREATE TABLE transactions (
    id SERIAL PRIMARY KEY,
    portfolio_id INTEGER REFERENCES portfolios(id) ON DELETE CASCADE,
    symbol VARCHAR(20) NOT NULL,
    transaction_type VARCHAR(10) NOT NULL, -- BUY, SELL
    quantity DECIMAL(15, 4) NOT NULL,
    price DECIMAL(15, 4) NOT NULL,
    commission DECIMAL(10, 2) DEFAULT 0,
    transaction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Performance history table
CREATE TABLE performance_history (
    id SERIAL PRIMARY KEY,
    portfolio_id INTEGER REFERENCES portfolios(id) ON DELETE CASCADE,
    date DATE NOT NULL,
    total_value DECIMAL(15, 2) NOT NULL,
    daily_return DECIMAL(10, 6),
    sharpe_ratio DECIMAL(10, 4),
    volatility DECIMAL(10, 6),
    max_drawdown DECIMAL(10, 6),
    UNIQUE(portfolio_id, date)
);

-- Indexes for performance
CREATE INDEX idx_accounts_user ON accounts(user_id);
CREATE INDEX idx_portfolios_account ON portfolios(account_id);
CREATE INDEX idx_positions_portfolio ON positions(portfolio_id);
CREATE INDEX idx_transactions_portfolio ON transactions(portfolio_id);
CREATE INDEX idx_performance_portfolio_date ON performance_history(portfolio_id, date);
```

### Account Manager

```python
# src/accounts/account_manager.py
from typing import List, Optional, Dict
import pandas as pd
from sqlalchemy.orm import Session
from .models import User, Account, Portfolio, Position
import logging

logger = logging.getLogger(__name__)

class AccountManager:
    """Manage multiple user accounts and portfolios."""

    def __init__(self, db_session: Session):
        self.db = db_session

    def create_user(self, username: str, email: str, password: str) -> User:
        """Create a new user."""
        from werkzeug.security import generate_password_hash

        user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password)
        )
        self.db.add(user)
        self.db.commit()
        logger.info(f"Created user: {username}")
        return user

    def create_account(self, user_id: int, account_name: str,
                      broker: str, initial_balance: float) -> Account:
        """Create a new trading account."""
        account = Account(
            user_id=user_id,
            account_name=account_name,
            broker=broker,
            initial_balance=initial_balance,
            current_balance=initial_balance
        )
        self.db.add(account)
        self.db.commit()
        logger.info(f"Created account: {account_name} for user {user_id}")
        return account

    def create_portfolio(self, account_id: int, portfolio_name: str,
                        strategy: str, risk_tolerance: float = 0.5) -> Portfolio:
        """Create a new portfolio within an account."""
        portfolio = Portfolio(
            account_id=account_id,
            portfolio_name=portfolio_name,
            strategy=strategy,
            risk_tolerance=risk_tolerance
        )
        self.db.add(portfolio)
        self.db.commit()
        logger.info(f"Created portfolio: {portfolio_name} for account {account_id}")
        return portfolio

    def get_user_accounts(self, user_id: int) -> List[Account]:
        """Get all accounts for a user."""
        return self.db.query(Account).filter(
            Account.user_id == user_id,
            Account.is_active == True
        ).all()

    def get_account_portfolios(self, account_id: int) -> List[Portfolio]:
        """Get all portfolios for an account."""
        return self.db.query(Portfolio).filter(
            Portfolio.account_id == account_id
        ).all()

    def get_portfolio_positions(self, portfolio_id: int) -> pd.DataFrame:
        """Get all positions in a portfolio."""
        positions = self.db.query(Position).filter(
            Position.portfolio_id == portfolio_id
        ).all()

        data = [{
            'symbol': p.symbol,
            'quantity': float(p.quantity),
            'avg_cost': float(p.avg_cost),
            'current_price': float(p.current_price) if p.current_price else 0,
            'market_value': float(p.quantity * (p.current_price or p.avg_cost))
        } for p in positions]

        return pd.DataFrame(data)

    def aggregate_portfolios(self, account_id: int) -> Dict:
        """Aggregate metrics across all portfolios in an account."""
        portfolios = self.get_account_portfolios(account_id)

        total_value = 0
        total_positions = 0

        for portfolio in portfolios:
            positions = self.get_portfolio_positions(portfolio.id)
            if not positions.empty:
                total_value += positions['market_value'].sum()
                total_positions += len(positions)

        return {
            'total_value': total_value,
            'num_portfolios': len(portfolios),
            'total_positions': total_positions
        }
```

### SQLAlchemy Models

```python
# src/accounts/models.py
from sqlalchemy import Column, Integer, String, Decimal, Boolean, DateTime, ForeignKey, Date
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    username = Column(String(100), unique=True, nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    accounts = relationship("Account", back_populates="user", cascade="all, delete-orphan")

class Account(Base):
    __tablename__ = 'accounts'

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    account_name = Column(String(100), nullable=False)
    broker = Column(String(50), nullable=False)
    account_number = Column(String(100))
    initial_balance = Column(Decimal(15, 2), nullable=False)
    current_balance = Column(Decimal(15, 2), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="accounts")
    portfolios = relationship("Portfolio", back_populates="account", cascade="all, delete-orphan")

class Portfolio(Base):
    __tablename__ = 'portfolios'

    id = Column(Integer, primary_key=True)
    account_id = Column(Integer, ForeignKey('accounts.id'), nullable=False)
    portfolio_name = Column(String(100), nullable=False)
    strategy = Column(String(50), nullable=False)
    risk_tolerance = Column(Decimal(3, 2), default=0.5)
    created_at = Column(DateTime, default=datetime.utcnow)

    account = relationship("Account", back_populates="portfolios")
    positions = relationship("Position", back_populates="portfolio", cascade="all, delete-orphan")

class Position(Base):
    __tablename__ = 'positions'

    id = Column(Integer, primary_key=True)
    portfolio_id = Column(Integer, ForeignKey('portfolios.id'), nullable=False)
    symbol = Column(String(20), nullable=False)
    quantity = Column(Decimal(15, 4), nullable=False)
    avg_cost = Column(Decimal(15, 4), nullable=False)
    current_price = Column(Decimal(15, 4))
    last_updated = Column(DateTime, default=datetime.utcnow)

    portfolio = relationship("Portfolio", back_populates="positions")
```

---

## 🚀 Deployment Plan

### Docker Compose for Full Stack

```yaml
# docker-compose.yml
version: '3.8'

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_USER: portfolio_user
      POSTGRES_PASSWORD: portfolio_pass
      POSTGRES_DB: portfolio_db
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./src/database/schema.sql:/docker-entrypoint-initdb.d/schema.sql

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
      - ./monitoring/prometheus/alerts.yml:/etc/prometheus/alerts.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./monitoring/grafana/datasources.yml:/etc/grafana/provisioning/datasources/datasources.yml
      - grafana_data:/var/lib/grafana
    depends_on:
      - prometheus

  streamlit:
    build: .
    ports:
      - "8501:8501"
    environment:
      - DATABASE_URL=postgresql://portfolio_user:portfolio_pass@postgres:5432/portfolio_db
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - postgres
      - redis
    volumes:
      - .:/app

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:
```

---

## 📝 Next Steps

1. **Create directory structure** for new modules
2. **Implement Interactive Brokers integration** (Phase 1)
3. **Set up monitoring stack** (Phase 2)
4. **Build alert system** (Phase 3)
5. **Implement multi-account management** (Phase 4)
6. **Add comprehensive tests** for all new features
7. **Update documentation** with setup guides
8. **Deploy to production** environment

---

**Status:** Planning Complete ✅
**Ready to Implement:** Yes
**Estimated Completion:** 5 weeks
