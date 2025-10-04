"""
Pydantic models for API request/response validation.
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime


class OptimizationRequest(BaseModel):
    """Request model for portfolio optimization."""

    tickers: List[str] = Field(
        ...,
        description="List of ticker symbols",
        min_items=2,
        max_items=50,
        example=["AAPL", "GOOGL", "MSFT", "AMZN"]
    )

    strategy: str = Field(
        ...,
        description="Optimization strategy",
        example="Max Sharpe"
    )

    period: str = Field(
        default="1y",
        description="Historical data period",
        example="1y"
    )

    max_assets: Optional[int] = Field(
        default=None,
        description="Maximum number of assets (for Concentrated strategy)",
        ge=2,
        le=20
    )

    risk_aversion: float = Field(
        default=2.5,
        description="Risk aversion parameter",
        ge=0.1,
        le=10.0
    )

    @validator('tickers')
    def validate_tickers(cls, v):
        """Validate ticker symbols."""
        # Remove duplicates and uppercase
        tickers = list(set(t.strip().upper() for t in v))

        if len(tickers) < 2:
            raise ValueError("At least 2 unique tickers required")

        return tickers

    @validator('strategy')
    def validate_strategy(cls, v):
        """Validate strategy name."""
        valid_strategies = [
            'Equal Weight',
            'Max Sharpe',
            'Min Variance',
            'Concentrated',
            'Risk Parity'
        ]

        if v not in valid_strategies:
            raise ValueError(
                f"Invalid strategy. Must be one of: {', '.join(valid_strategies)}"
            )

        return v

    @validator('period')
    def validate_period(cls, v):
        """Validate period format."""
        valid_periods = ['1mo', '3mo', '6mo', '1y', '2y', '5y', 'max']

        if v not in valid_periods:
            raise ValueError(
                f"Invalid period. Must be one of: {', '.join(valid_periods)}"
            )

        return v

    class Config:
        schema_extra = {
            "example": {
                "tickers": ["AAPL", "GOOGL", "MSFT", "AMZN", "NVDA"],
                "strategy": "Max Sharpe",
                "period": "1y",
                "risk_aversion": 2.5
            }
        }


class PortfolioMetrics(BaseModel):
    """Portfolio performance metrics."""

    expected_return: float = Field(..., description="Annual expected return")
    volatility: float = Field(..., description="Annual volatility")
    sharpe_ratio: float = Field(..., description="Sharpe ratio")
    n_assets: int = Field(..., description="Number of assets in portfolio")
    max_drawdown: Optional[float] = Field(None, description="Maximum drawdown")
    var_95: Optional[float] = Field(None, description="Value at Risk (95%)")
    cvar_95: Optional[float] = Field(None, description="Conditional VaR (95%)")

    class Config:
        schema_extra = {
            "example": {
                "expected_return": 0.185,
                "volatility": 0.215,
                "sharpe_ratio": 0.86,
                "n_assets": 5,
                "max_drawdown": -0.18,
                "var_95": -0.025,
                "cvar_95": -0.032
            }
        }


class OptimizationResponse(BaseModel):
    """Response model for portfolio optimization."""

    success: bool = Field(..., description="Whether optimization succeeded")
    message: str = Field(..., description="Status message")
    strategy: str = Field(..., description="Strategy used")
    weights: Dict[str, float] = Field(..., description="Portfolio weights")
    metrics: PortfolioMetrics = Field(..., description="Portfolio metrics")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp of optimization"
    )
    execution_time: float = Field(..., description="Execution time in seconds")

    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Optimization completed successfully",
                "strategy": "Max Sharpe",
                "weights": {
                    "AAPL": 0.25,
                    "GOOGL": 0.30,
                    "MSFT": 0.20,
                    "AMZN": 0.15,
                    "NVDA": 0.10
                },
                "metrics": {
                    "expected_return": 0.185,
                    "volatility": 0.215,
                    "sharpe_ratio": 0.86,
                    "n_assets": 5
                },
                "timestamp": "2025-10-04T12:00:00",
                "execution_time": 1.23
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Current server time"
    )

    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "version": "2.0.0",
                "timestamp": "2025-10-04T12:00:00"
            }
        }


class BacktestRequest(BaseModel):
    """Request model for backtesting."""

    tickers: List[str] = Field(..., min_items=2, max_items=50)
    strategy: str
    period: str = Field(default="2y")
    rebalance_frequency: int = Field(
        default=21,
        description="Rebalancing frequency in days",
        ge=1,
        le=252
    )
    transaction_cost: float = Field(
        default=0.001,
        description="Transaction cost as decimal",
        ge=0.0,
        le=0.05
    )

    @validator('tickers')
    def validate_tickers(cls, v):
        return list(set(t.strip().upper() for t in v))

    @validator('strategy')
    def validate_strategy(cls, v):
        valid_strategies = ['Equal Weight', 'Max Sharpe', 'Min Variance', 'Concentrated', 'Risk Parity']
        if v not in valid_strategies:
            raise ValueError(f"Invalid strategy: {v}")
        return v


class BacktestResponse(BaseModel):
    """Response model for backtesting."""

    success: bool
    message: str
    strategy: str
    total_return: float = Field(..., description="Total return over period")
    annual_return: float = Field(..., description="Annualized return")
    annual_volatility: float = Field(..., description="Annualized volatility")
    sharpe_ratio: float
    max_drawdown: float
    n_rebalances: int = Field(..., description="Number of rebalancing events")
    total_transaction_cost: float = Field(..., description="Total transaction costs")
    equity_curve: Dict[str, float] = Field(..., description="Equity curve over time")
    timestamp: datetime = Field(default_factory=datetime.now)
    execution_time: float


class ErrorResponse(BaseModel):
    """Error response model."""

    success: bool = Field(default=False)
    error: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.now)

    class Config:
        schema_extra = {
            "example": {
                "success": False,
                "error": "Invalid ticker symbol",
                "details": {"invalid_tickers": ["INVALID"]},
                "timestamp": "2025-10-04T12:00:00"
            }
        }
