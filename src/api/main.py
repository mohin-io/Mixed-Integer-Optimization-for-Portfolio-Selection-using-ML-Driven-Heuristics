"""
FastAPI REST API for Mixed-Integer-Optimization-for-Portfolio-Selection-using-ML-Driven-Heuristics

Production-ready API with automatic documentation, validation, and error handling.

Run with:
    uvicorn src.api.main:app --reload --port 8000

API Documentation:
    http://localhost:8000/docs (Swagger UI)
    http://localhost:8000/redoc (ReDoc)
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
import numpy as np
import pandas as pd
from typing import Dict, Any
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.api.models import (
    OptimizationRequest,
    OptimizationResponse,
    PortfolioMetrics,
    HealthResponse,
    BacktestRequest,
    BacktestResponse,
    ErrorResponse
)
from src.utils import (
    setup_logger,
    validate_tickers,
    validate_strategy,
    DataValidationError,
    OptimizationError,
    APIError
)

# Initialize logger
logger = setup_logger(__name__, log_level=20)  # INFO level

# Create FastAPI app
app = FastAPI(
    title="Mixed-Integer-Optimization-for-Portfolio-Selection-using-ML-Driven-Heuristics API",
    description="Mixed-Integer Optimization for Portfolio Selection with ML-Driven Heuristics",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/api/openapi.json"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(DataValidationError)
async def validation_error_handler(request, exc: DataValidationError):
    """Handle validation errors."""
    logger.error(f"Validation error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=ErrorResponse(
            error=str(exc),
            details=exc.details
        ).dict()
    )


@app.exception_handler(OptimizationError)
async def optimization_error_handler(request, exc: OptimizationError):
    """Handle optimization errors."""
    logger.error(f"Optimization error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error=str(exc),
            details=exc.details
        ).dict()
    )


@app.exception_handler(APIError)
async def api_error_handler(request, exc: APIError):
    """Handle API errors."""
    logger.error(f"API error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=ErrorResponse(
            error=str(exc),
            details=exc.details
        ).dict()
    )


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Mixed-Integer-Optimization-for-Portfolio-Selection-using-ML-Driven-Heuristics API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    logger.info("Health check requested")
    return HealthResponse(
        status="healthy",
        version="2.0.0"
    )


@app.post("/api/v1/optimize", response_model=OptimizationResponse)
async def optimize_portfolio(request: OptimizationRequest):
    """
    Optimize portfolio using specified strategy.

    Args:
        request: Optimization request with tickers, strategy, and parameters

    Returns:
        Optimization response with weights and metrics

    Raises:
        HTTPException: If optimization fails
    """
    start_time = time.time()

    try:
        logger.info(f"Optimization requested: {request.strategy} for {len(request.tickers)} tickers")

        # Validate inputs
        tickers = validate_tickers(request.tickers, min_tickers=2, max_tickers=50)
        strategy = validate_strategy(request.strategy)

        # Fetch real data
        from src.data.real_data_loader import RealDataLoader

        loader = RealDataLoader()
        prices, returns, success = loader.fetch_data(tickers, period=request.period)

        if not success:
            raise APIError(
                message="Failed to fetch market data",
                details={'tickers': tickers, 'period': request.period}
            )

        # Perform optimization
        weights, metrics = _optimize(
            returns,
            strategy,
            request.max_assets,
            request.risk_aversion
        )

        execution_time = time.time() - start_time

        logger.info(f"Optimization completed in {execution_time:.2f}s")

        return OptimizationResponse(
            success=True,
            message="Optimization completed successfully",
            strategy=strategy,
            weights=weights,
            metrics=metrics,
            execution_time=execution_time
        )

    except (DataValidationError, OptimizationError, APIError):
        raise

    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}"
        )


@app.post("/api/v1/backtest", response_model=BacktestResponse)
async def backtest_strategy(request: BacktestRequest):
    """
    Backtest portfolio strategy with rebalancing.

    Args:
        request: Backtest request with parameters

    Returns:
        Backtest results with performance metrics
    """
    start_time = time.time()

    try:
        logger.info(f"Backtest requested: {request.strategy}")

        # Validate inputs
        tickers = validate_tickers(request.tickers)
        strategy = validate_strategy(request.strategy)

        # Fetch data
        from src.data.real_data_loader import RealDataLoader

        loader = RealDataLoader()
        prices, returns, success = loader.fetch_data(tickers, period=request.period)

        if not success:
            raise APIError(
                message="Failed to fetch market data",
                details={'tickers': tickers}
            )

        # Perform backtest
        results = _backtest(
            returns,
            strategy,
            request.rebalance_frequency,
            request.transaction_cost
        )

        execution_time = time.time() - start_time

        return BacktestResponse(
            success=True,
            message="Backtest completed successfully",
            strategy=strategy,
            execution_time=execution_time,
            **results
        )

    except Exception as e:
        logger.error(f"Backtest error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


def _optimize(
    returns: pd.DataFrame,
    strategy: str,
    max_assets: int = None,
    risk_aversion: float = 2.5
) -> tuple:
    """Internal optimization logic."""
    from src.visualization.dashboard import optimize_portfolio, evaluate_portfolio

    weights, annual_returns, cov_matrix = optimize_portfolio(
        returns, strategy, max_assets, risk_aversion
    )
    metrics_dict = evaluate_portfolio(weights, annual_returns, cov_matrix)

    # Convert weights to dict
    weights_dict = {k: float(v) for k, v in weights.items() if v > 1e-6}

    # Calculate additional metrics
    portfolio_returns = (returns * weights.values).sum(axis=1)
    cumulative = (1 + portfolio_returns).cumprod()
    max_drawdown = float((cumulative / cumulative.cummax() - 1).min())

    # VaR and CVaR
    var_95 = float(np.percentile(portfolio_returns, 5))
    cvar_95 = float(portfolio_returns[portfolio_returns <= var_95].mean())

    metrics = PortfolioMetrics(
        expected_return=float(metrics_dict['return']),
        volatility=float(metrics_dict['volatility']),
        sharpe_ratio=float(metrics_dict['sharpe']),
        n_assets=int(metrics_dict['n_assets']),
        max_drawdown=max_drawdown,
        var_95=var_95,
        cvar_95=cvar_95
    )

    return weights_dict, metrics


def _backtest(
    returns: pd.DataFrame,
    strategy: str,
    rebalance_freq: int = 21,
    transaction_cost: float = 0.001
) -> Dict[str, Any]:
    """Internal backtesting logic."""
    from src.visualization.dashboard import optimize_portfolio

    n_days = len(returns)
    lookback = 252

    portfolio_value = 1.0
    equity_curve = {str(returns.index[0].date()): portfolio_value}
    total_cost = 0.0
    n_rebalances = 0

    current_weights = pd.Series(0, index=returns.columns)

    for i in range(lookback, n_days):
        # Rebalance
        if (i - lookback) % rebalance_freq == 0:
            hist_returns = returns.iloc[max(0, i - lookback):i]

            try:
                new_weights, _, _ = optimize_portfolio(hist_returns, strategy)

                # Calculate turnover
                turnover = np.abs(new_weights.values - current_weights.values).sum()
                cost = turnover * transaction_cost

                total_cost += cost
                portfolio_value *= (1 - cost)
                current_weights = new_weights
                n_rebalances += 1

            except Exception as e:
                logger.warning(f"Rebalancing failed at step {i}: {e}")
                continue

        # Daily return
        if (current_weights > 0).any():
            daily_return = (returns.iloc[i] * current_weights.values).sum()
            portfolio_value *= (1 + daily_return)

        equity_curve[str(returns.index[i].date())] = float(portfolio_value)

    # Calculate metrics
    total_return = float(portfolio_value - 1)
    n_years = n_days / 252
    annual_return = float((1 + total_return) ** (1 / n_years) - 1) if n_years > 0 else 0

    portfolio_returns = pd.Series(equity_curve).pct_change().dropna()
    annual_volatility = float(portfolio_returns.std() * np.sqrt(252))
    sharpe_ratio = float(annual_return / annual_volatility) if annual_volatility > 0 else 0

    equity_series = pd.Series(equity_curve)
    max_drawdown = float((equity_series / equity_series.cummax() - 1).min())

    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'annual_volatility': annual_volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'n_rebalances': n_rebalances,
        'total_transaction_cost': float(total_cost),
        'equity_curve': equity_curve
    }


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting Mixed-Integer-Optimization-for-Portfolio-Selection-using-ML-Driven-Heuristics API")
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
