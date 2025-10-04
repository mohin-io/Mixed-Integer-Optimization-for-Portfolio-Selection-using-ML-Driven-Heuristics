"""
Production Monitoring with Prometheus Metrics.

Implements:
- Portfolio performance metrics
- Trading execution metrics
- System health metrics
- Alert triggers
- Grafana dashboard compatibility
"""

import time
from typing import Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np

try:
    from prometheus_client import Counter, Gauge, Histogram, Summary, Info, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    # Fallback: simple logging
    class MockMetric:
        def __init__(self, *args, **kwargs):
            pass
        def inc(self, *args, **kwargs):
            pass
        def dec(self, *args, **kwargs):
            pass
        def set(self, *args, **kwargs):
            pass
        def observe(self, *args, **kwargs):
            pass
        def info(self, *args, **kwargs):
            pass

    Counter = Gauge = Histogram = Summary = Info = MockMetric


# Portfolio Performance Metrics
portfolio_value = Gauge('portfolio_value_usd', 'Current portfolio value in USD')
portfolio_return_daily = Gauge('portfolio_return_daily_pct', 'Daily portfolio return percentage')
portfolio_return_total = Gauge('portfolio_return_total_pct', 'Total portfolio return percentage')
portfolio_volatility = Gauge('portfolio_volatility_annual', 'Annualized portfolio volatility')
portfolio_sharpe_ratio = Gauge('portfolio_sharpe_ratio', 'Portfolio Sharpe ratio')

# Position Metrics
num_positions = Gauge('portfolio_num_positions', 'Number of active positions')
position_concentration = Gauge('portfolio_concentration', 'Portfolio concentration (HHI)')
largest_position_weight = Gauge('portfolio_largest_position_pct', 'Largest position weight percentage')

# Trading Execution Metrics
orders_submitted = Counter('trading_orders_submitted_total', 'Total orders submitted', ['side', 'type'])
orders_filled = Counter('trading_orders_filled_total', 'Total orders filled', ['side', 'type'])
orders_failed = Counter('trading_orders_failed_total', 'Total orders failed', ['side', 'type', 'reason'])
order_execution_time = Histogram('trading_order_execution_seconds', 'Order execution time in seconds')

# Transaction Costs
transaction_costs_total = Counter('trading_costs_total_usd', 'Total transaction costs in USD')
transaction_costs_rate = Gauge('trading_costs_rate_bps', 'Transaction costs in basis points')
slippage_total = Counter('trading_slippage_total_bps', 'Total slippage in basis points')

# Risk Metrics
portfolio_var_95 = Gauge('portfolio_var_95_pct', '95% Value at Risk')
portfolio_cvar_95 = Gauge('portfolio_cvar_95_pct', '95% Conditional VaR')
portfolio_beta = Gauge('portfolio_beta', 'Portfolio beta to market')
portfolio_max_drawdown = Gauge('portfolio_max_drawdown_pct', 'Maximum drawdown percentage')

# Rebalancing Metrics
rebalances_executed = Counter('portfolio_rebalances_total', 'Total portfolio rebalances')
rebalance_turnover = Histogram('portfolio_rebalance_turnover_pct', 'Portfolio turnover percentage')

# ESG Metrics
portfolio_esg_score = Gauge('portfolio_esg_score', 'Portfolio ESG score', ['component'])
portfolio_carbon_footprint = Gauge('portfolio_carbon_footprint', 'Portfolio carbon footprint')

# System Health
api_requests_total = Counter('api_requests_total', 'Total API requests', ['endpoint', 'status'])
api_latency = Histogram('api_latency_seconds', 'API latency in seconds', ['endpoint'])
websocket_messages = Counter('websocket_messages_total', 'WebSocket messages received', ['message_type'])
websocket_reconnections = Counter('websocket_reconnections_total', 'WebSocket reconnection attempts')

# Model Performance
model_predictions = Counter('model_predictions_total', 'Model predictions made', ['model_type'])
model_accuracy = Gauge('model_accuracy', 'Model prediction accuracy', ['model_type'])
model_inference_time = Histogram('model_inference_seconds', 'Model inference time', ['model_type'])

# System Info
system_info = Info('system', 'System information')


class PortfolioMetricsCollector:
    """
    Collects and exports portfolio metrics to Prometheus.
    """

    def __init__(self, update_interval: int = 60):
        """
        Initialize metrics collector.

        Args:
            update_interval: Seconds between metric updates
        """
        self.update_interval = update_interval
        self.is_running = False

    def update_portfolio_metrics(
        self,
        current_value: float,
        previous_value: float,
        positions: Dict,
        returns: np.ndarray,
        risk_metrics: Dict
    ):
        """
        Update portfolio-related metrics.

        Args:
            current_value: Current portfolio value
            previous_value: Previous portfolio value
            positions: Dictionary of positions
            returns: Historical returns
            risk_metrics: Risk metrics dictionary
        """
        # Value and returns
        portfolio_value.set(current_value)

        daily_return = (current_value - previous_value) / previous_value * 100
        portfolio_return_daily.set(daily_return)

        # Positions
        num_positions.set(len(positions))

        if positions:
            weights = np.array([p['weight'] for p in positions.values()])
            concentration = np.sum(weights ** 2)  # Herfindahl index
            portfolio_concentration.set(concentration)
            largest_position_weight.set(np.max(weights) * 100)

        # Risk metrics
        if 'volatility' in risk_metrics:
            portfolio_volatility.set(risk_metrics['volatility'])

        if 'sharpe_ratio' in risk_metrics:
            portfolio_sharpe_ratio.set(risk_metrics['sharpe_ratio'])

        if 'var_95' in risk_metrics:
            portfolio_var_95.set(risk_metrics['var_95'] * 100)

        if 'cvar_95' in risk_metrics:
            portfolio_cvar_95.set(risk_metrics['cvar_95'] * 100)

        if 'max_drawdown' in risk_metrics:
            portfolio_max_drawdown.set(risk_metrics['max_drawdown'] * 100)

    def record_order(
        self,
        side: str,
        order_type: str,
        status: str,
        execution_time: Optional[float] = None
    ):
        """
        Record order execution metrics.

        Args:
            side: 'buy' or 'sell'
            order_type: 'market', 'limit', etc.
            status: 'submitted', 'filled', 'failed'
            execution_time: Time to execute in seconds
        """
        if status == 'submitted':
            orders_submitted.labels(side=side, type=order_type).inc()

        elif status == 'filled':
            orders_filled.labels(side=side, type=order_type).inc()

            if execution_time is not None:
                order_execution_time.observe(execution_time)

        elif status == 'failed':
            orders_failed.labels(side=side, type=order_type, reason='unknown').inc()

    def record_transaction_costs(
        self,
        cost_usd: float,
        portfolio_value: float,
        slippage_bps: float = 0
    ):
        """
        Record transaction costs.

        Args:
            cost_usd: Transaction cost in USD
            portfolio_value: Current portfolio value
            slippage_bps: Slippage in basis points
        """
        transaction_costs_total.inc(cost_usd)

        cost_bps = (cost_usd / portfolio_value) * 10000
        transaction_costs_rate.set(cost_bps)

        if slippage_bps > 0:
            slippage_total.inc(slippage_bps)

    def record_rebalance(self, turnover: float):
        """
        Record rebalancing event.

        Args:
            turnover: Portfolio turnover percentage
        """
        rebalances_executed.inc()
        rebalance_turnover.observe(turnover)

    def update_esg_metrics(
        self,
        total_score: float,
        environmental: float,
        social: float,
        governance: float,
        carbon_footprint: float
    ):
        """
        Update ESG metrics.

        Args:
            total_score: Total ESG score
            environmental: Environmental score
            social: Social score
            governance: Governance score
            carbon_footprint: Carbon footprint
        """
        portfolio_esg_score.labels(component='total').set(total_score)
        portfolio_esg_score.labels(component='environmental').set(environmental)
        portfolio_esg_score.labels(component='social').set(social)
        portfolio_esg_score.labels(component='governance').set(governance)

        portfolio_carbon_footprint.set(carbon_footprint)

    def record_api_call(
        self,
        endpoint: str,
        status: int,
        latency: float
    ):
        """
        Record API call metrics.

        Args:
            endpoint: API endpoint
            status: HTTP status code
            latency: Latency in seconds
        """
        api_requests_total.labels(endpoint=endpoint, status=str(status)).inc()
        api_latency.labels(endpoint=endpoint).observe(latency)

    def record_websocket_message(self, message_type: str):
        """Record WebSocket message."""
        websocket_messages.labels(message_type=message_type).inc()

    def record_model_prediction(
        self,
        model_type: str,
        inference_time: float,
        accuracy: Optional[float] = None
    ):
        """
        Record model prediction metrics.

        Args:
            model_type: Type of model
            inference_time: Inference time in seconds
            accuracy: Prediction accuracy (if available)
        """
        model_predictions.labels(model_type=model_type).inc()
        model_inference_time.labels(model_type=model_type).observe(inference_time)

        if accuracy is not None:
            model_accuracy.labels(model_type=model_type).set(accuracy)

    def set_system_info(self, info_dict: Dict):
        """
        Set system information.

        Args:
            info_dict: Dictionary with system info
        """
        system_info.info(info_dict)


class AlertManager:
    """
    Manages alerts based on metric thresholds.
    """

    def __init__(self):
        """Initialize alert manager."""
        self.alerts = []
        self.thresholds = {
            'max_drawdown': 0.10,  # 10%
            'sharpe_ratio_min': 0.5,
            'concentration_max': 0.5,
            'largest_position_max': 0.30,
            'var_95_max': 0.05
        }

    def check_alerts(
        self,
        portfolio_value: float,
        metrics: Dict
    ) -> List[str]:
        """
        Check for alert conditions.

        Args:
            portfolio_value: Current portfolio value
            metrics: Current metrics

        Returns:
            List of alert messages
        """
        alerts = []

        # Max drawdown alert
        if metrics.get('max_drawdown', 0) > self.thresholds['max_drawdown']:
            alerts.append(
                f"⚠️ Maximum drawdown exceeded: "
                f"{metrics['max_drawdown']*100:.1f}% > "
                f"{self.thresholds['max_drawdown']*100:.1f}%"
            )

        # Sharpe ratio alert
        if metrics.get('sharpe_ratio', float('inf')) < self.thresholds['sharpe_ratio_min']:
            alerts.append(
                f"⚠️ Low Sharpe ratio: "
                f"{metrics['sharpe_ratio']:.2f} < "
                f"{self.thresholds['sharpe_ratio_min']:.2f}"
            )

        # Concentration alert
        if metrics.get('concentration', 0) > self.thresholds['concentration_max']:
            alerts.append(
                f"⚠️ High portfolio concentration: "
                f"{metrics['concentration']:.2f} > "
                f"{self.thresholds['concentration_max']:.2f}"
            )

        return alerts


def start_metrics_server(port: int = 8000):
    """
    Start Prometheus metrics HTTP server.

    Args:
        port: Port to serve metrics on
    """
    if PROMETHEUS_AVAILABLE:
        start_http_server(port)
        print(f"Metrics server started on port {port}")
        print(f"Access metrics at: http://localhost:{port}/metrics")
    else:
        print("Prometheus client not available. Install with: pip install prometheus-client")


if __name__ == "__main__":
    print("Testing Prometheus Metrics...")

    if not PROMETHEUS_AVAILABLE:
        print("⚠️ prometheus-client not installed.")
        print("Install with: pip install prometheus-client")
        print("\nUsing mock metrics for demonstration.")

    # Initialize collector
    collector = PortfolioMetricsCollector()

    # Simulate metrics
    print("\n1. Portfolio Metrics")
    collector.update_portfolio_metrics(
        current_value=105000,
        previous_value=100000,
        positions={
            'AAPL': {'weight': 0.30},
            'MSFT': {'weight': 0.25},
            'GOOGL': {'weight': 0.25},
            'AMZN': {'weight': 0.20}
        },
        returns=np.random.randn(252) * 0.01,
        risk_metrics={
            'volatility': 0.15,
            'sharpe_ratio': 1.5,
            'var_95': 0.025,
            'cvar_95': 0.035,
            'max_drawdown': 0.08
        }
    )
    print("   ✅ Portfolio metrics updated")

    # Record orders
    print("\n2. Trading Metrics")
    collector.record_order('buy', 'market', 'filled', execution_time=0.5)
    collector.record_order('sell', 'limit', 'filled', execution_time=1.2)
    print("   ✅ Order metrics recorded")

    # Transaction costs
    print("\n3. Cost Metrics")
    collector.record_transaction_costs(
        cost_usd=50,
        portfolio_value=105000,
        slippage_bps=2.5
    )
    print("   ✅ Cost metrics recorded")

    # Rebalancing
    print("\n4. Rebalancing Metrics")
    collector.record_rebalance(turnover=15.5)
    print("   ✅ Rebalancing metrics recorded")

    # ESG
    print("\n5. ESG Metrics")
    collector.update_esg_metrics(
        total_score=75.5,
        environmental=80.0,
        social=72.0,
        governance=74.5,
        carbon_footprint=125.3
    )
    print("   ✅ ESG metrics recorded")

    # System info
    print("\n6. System Info")
    collector.set_system_info({
        'version': '2.0.0',
        'environment': 'production',
        'strategy': 'RL-Enhanced MVO'
    })
    print("   ✅ System info set")

    # Alerts
    print("\n7. Alert Manager")
    alert_mgr = AlertManager()
    alerts = alert_mgr.check_alerts(
        portfolio_value=105000,
        metrics={
            'max_drawdown': 0.12,
            'sharpe_ratio': 1.5,
            'concentration': 0.35
        }
    )

    if alerts:
        print("   Alerts triggered:")
        for alert in alerts:
            print(f"   {alert}")
    else:
        print("   ✅ No alerts")

    print("\n✅ Prometheus metrics implementation complete!")
    print("\nTo start metrics server:")
    print("  start_metrics_server(port=8000)")
    print("\nThen view metrics at: http://localhost:8000/metrics")
