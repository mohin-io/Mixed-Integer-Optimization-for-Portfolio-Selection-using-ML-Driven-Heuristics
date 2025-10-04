"""
Custom Exceptions for Portfolio Optimization

Professional error handling with specific exception types.
"""

from typing import Optional, Dict, Any


class PortfolioException(Exception):
    """
    Base exception for all portfolio-related errors.

    All custom exceptions should inherit from this class.
    """

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None
    ):
        """
        Initialize portfolio exception.

        Args:
            message: Human-readable error message
            details: Additional context about the error
            original_exception: Original exception if this is a wrapper
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.original_exception = original_exception

    def __str__(self) -> str:
        """Return formatted error message."""
        error_msg = self.message

        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            error_msg += f" ({details_str})"

        if self.original_exception:
            error_msg += f" | Caused by: {str(self.original_exception)}"

        return error_msg


class DataValidationError(PortfolioException):
    """
    Raised when input data fails validation.

    Examples:
        - Missing required columns
        - Invalid data types
        - NaN/Inf values
        - Insufficient data points
    """
    pass


class OptimizationError(PortfolioException):
    """
    Raised when portfolio optimization fails.

    Examples:
        - Solver convergence failure
        - Infeasible constraints
        - Numerical instability
    """
    pass


class InvalidStrategyError(PortfolioException):
    """
    Raised when an invalid strategy is specified.

    Examples:
        - Unknown strategy name
        - Strategy not compatible with data
        - Missing required parameters
    """
    pass


class InsufficientDataError(DataValidationError):
    """
    Raised when there is not enough data for analysis.

    Examples:
        - Too few time periods
        - Too few assets
        - Missing price data
    """
    pass


class ConfigurationError(PortfolioException):
    """
    Raised when configuration is invalid.

    Examples:
        - Missing config file
        - Invalid parameter values
        - Incompatible settings
    """
    pass


class APIError(PortfolioException):
    """
    Raised when external API calls fail.

    Examples:
        - Yahoo Finance fetch failure
        - Rate limiting
        - Network errors
    """
    pass


class BacktestError(PortfolioException):
    """
    Raised when backtesting encounters an error.

    Examples:
        - Invalid rebalancing frequency
        - Lookback window too large
        - Transaction cost out of range
    """
    pass


# Example usage and error handling patterns
if __name__ == '__main__':
    # Example 1: Data validation
    try:
        # Simulated validation
        data_length = 10
        min_required = 50

        if data_length < min_required:
            raise InsufficientDataError(
                message="Not enough historical data",
                details={
                    'data_length': data_length,
                    'min_required': min_required,
                    'asset': 'AAPL'
                }
            )
    except InsufficientDataError as e:
        print(f"Error: {e}")

    # Example 2: Optimization error
    try:
        # Simulated optimization failure
        raise OptimizationError(
            message="Portfolio optimization failed to converge",
            details={
                'strategy': 'Max Sharpe',
                'iterations': 10000,
                'best_sharpe': 0.45
            }
        )
    except OptimizationError as e:
        print(f"Error: {e}")

    # Example 3: Invalid strategy
    try:
        strategy = "SuperStrategy"
        valid_strategies = ['Equal Weight', 'Max Sharpe', 'Min Variance']

        if strategy not in valid_strategies:
            raise InvalidStrategyError(
                message=f"Strategy '{strategy}' is not recognized",
                details={
                    'requested': strategy,
                    'available': valid_strategies
                }
            )
    except InvalidStrategyError as e:
        print(f"Error: {e}")

    # Example 4: Nested exception
    try:
        try:
            # Simulate API call
            raise ConnectionError("Network timeout")
        except ConnectionError as e:
            raise APIError(
                message="Failed to fetch market data",
                details={'ticker': 'AAPL', 'period': '1y'},
                original_exception=e
            )
    except APIError as e:
        print(f"Error: {e}")
