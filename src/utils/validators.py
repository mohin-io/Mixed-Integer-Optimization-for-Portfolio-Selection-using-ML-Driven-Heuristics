"""
Input Validation Functions for Portfolio Optimization

Provides comprehensive validation for returns data, weights, and strategy parameters.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Union
from .exceptions import DataValidationError, InvalidStrategyError, InsufficientDataError


def validate_returns(
    returns: pd.DataFrame,
    min_periods: int = 20,
    max_nan_ratio: float = 0.1
) -> pd.DataFrame:
    """
    Validate returns DataFrame for portfolio optimization.

    Args:
        returns: DataFrame of asset returns
        min_periods: Minimum number of time periods required
        max_nan_ratio: Maximum allowed ratio of NaN values

    Returns:
        Validated returns DataFrame

    Raises:
        DataValidationError: If validation fails
        InsufficientDataError: If not enough data points
    """
    # Check if DataFrame
    if not isinstance(returns, pd.DataFrame):
        raise DataValidationError(
            message="Returns must be a pandas DataFrame",
            details={'type_received': type(returns).__name__}
        )

    # Check for empty DataFrame
    if returns.empty:
        raise InsufficientDataError(
            message="Returns DataFrame is empty",
            details={'shape': returns.shape}
        )

    # Check minimum periods
    n_periods = len(returns)
    if n_periods < min_periods:
        raise InsufficientDataError(
            message="Insufficient time periods for analysis",
            details={
                'periods_available': n_periods,
                'periods_required': min_periods
            }
        )

    # Check for minimum number of assets
    n_assets = len(returns.columns)
    if n_assets < 2:
        raise InsufficientDataError(
            message="At least 2 assets required for portfolio optimization",
            details={'n_assets': n_assets}
        )

    # Check for NaN values
    nan_ratio = returns.isna().sum().sum() / (returns.shape[0] * returns.shape[1])
    if nan_ratio > max_nan_ratio:
        raise DataValidationError(
            message="Too many NaN values in returns data",
            details={
                'nan_ratio': nan_ratio,
                'max_allowed': max_nan_ratio,
                'nan_count': returns.isna().sum().sum()
            }
        )

    # Check for infinite values
    if np.isinf(returns.values).any():
        inf_count = np.isinf(returns.values).sum()
        raise DataValidationError(
            message="Returns contain infinite values",
            details={'inf_count': int(inf_count)}
        )

    # Check for numeric data
    if not all(returns.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
        non_numeric = returns.dtypes[~returns.dtypes.apply(
            lambda x: np.issubdtype(x, np.number)
        )].to_dict()
        raise DataValidationError(
            message="Returns contain non-numeric columns",
            details={'non_numeric_columns': non_numeric}
        )

    # Check for extreme values (potential data errors)
    extreme_threshold = 10.0  # 1000% return in single period
    extreme_values = (np.abs(returns) > extreme_threshold).sum().sum()
    if extreme_values > 0:
        raise DataValidationError(
            message="Returns contain extreme values (potential data errors)",
            details={
                'extreme_count': int(extreme_values),
                'threshold': extreme_threshold,
                'max_value': float(returns.abs().max().max())
            }
        )

    return returns


def validate_weights(
    weights: Union[pd.Series, np.ndarray],
    tolerance: float = 1e-6
) -> pd.Series:
    """
    Validate portfolio weights.

    Args:
        weights: Portfolio weights (Series or array)
        tolerance: Numerical tolerance for sum check

    Returns:
        Validated weights as Series

    Raises:
        DataValidationError: If validation fails
    """
    # Convert to Series if array
    if isinstance(weights, np.ndarray):
        weights = pd.Series(weights)

    # Check for negative weights
    if (weights < -tolerance).any():
        negative_assets = weights[weights < -tolerance].index.tolist()
        raise DataValidationError(
            message="Weights contain negative values (short-selling not allowed)",
            details={
                'negative_assets': negative_assets,
                'min_weight': float(weights.min())
            }
        )

    # Check sum to 1
    weight_sum = weights.sum()
    if not np.isclose(weight_sum, 1.0, atol=tolerance):
        raise DataValidationError(
            message="Weights do not sum to 1",
            details={
                'weight_sum': float(weight_sum),
                'deviation': float(abs(weight_sum - 1.0))
            }
        )

    # Check for NaN or infinite weights
    if weights.isna().any():
        raise DataValidationError(
            message="Weights contain NaN values",
            details={'nan_count': int(weights.isna().sum())}
        )

    if np.isinf(weights.values).any():
        raise DataValidationError(
            message="Weights contain infinite values",
            details={'inf_count': int(np.isinf(weights.values).sum())}
        )

    return weights


def validate_strategy(
    strategy: str,
    valid_strategies: Optional[List[str]] = None
) -> str:
    """
    Validate strategy name.

    Args:
        strategy: Strategy name
        valid_strategies: List of valid strategy names

    Returns:
        Validated strategy name

    Raises:
        InvalidStrategyError: If strategy is invalid
    """
    if valid_strategies is None:
        valid_strategies = [
            'Equal Weight',
            'Max Sharpe',
            'Min Variance',
            'Concentrated',
            'Risk Parity'
        ]

    if not isinstance(strategy, str):
        raise InvalidStrategyError(
            message="Strategy must be a string",
            details={'type_received': type(strategy).__name__}
        )

    if strategy not in valid_strategies:
        raise InvalidStrategyError(
            message=f"Strategy '{strategy}' is not recognized",
            details={
                'requested': strategy,
                'available': valid_strategies
            }
        )

    return strategy


def validate_covariance_matrix(
    cov_matrix: Union[pd.DataFrame, np.ndarray],
    tolerance: float = 1e-6
) -> pd.DataFrame:
    """
    Validate covariance matrix.

    Args:
        cov_matrix: Covariance matrix
        tolerance: Numerical tolerance for symmetry check

    Returns:
        Validated covariance matrix as DataFrame

    Raises:
        DataValidationError: If validation fails
    """
    # Convert to DataFrame if array
    if isinstance(cov_matrix, np.ndarray):
        cov_matrix = pd.DataFrame(cov_matrix)

    # Check square matrix
    if cov_matrix.shape[0] != cov_matrix.shape[1]:
        raise DataValidationError(
            message="Covariance matrix must be square",
            details={'shape': cov_matrix.shape}
        )

    # Check symmetry
    if not np.allclose(cov_matrix, cov_matrix.T, atol=tolerance):
        max_asymmetry = float(np.abs(cov_matrix - cov_matrix.T).max().max())
        raise DataValidationError(
            message="Covariance matrix is not symmetric",
            details={'max_asymmetry': max_asymmetry}
        )

    # Check positive semi-definite (all eigenvalues >= 0)
    eigenvalues = np.linalg.eigvalsh(cov_matrix.values)
    min_eigenvalue = float(eigenvalues.min())

    if min_eigenvalue < -tolerance:
        raise DataValidationError(
            message="Covariance matrix is not positive semi-definite",
            details={
                'min_eigenvalue': min_eigenvalue,
                'negative_eigenvalues': int((eigenvalues < -tolerance).sum())
            }
        )

    # Check for NaN or infinite values
    if cov_matrix.isna().any().any():
        raise DataValidationError(
            message="Covariance matrix contains NaN values",
            details={'nan_count': int(cov_matrix.isna().sum().sum())}
        )

    if np.isinf(cov_matrix.values).any():
        raise DataValidationError(
            message="Covariance matrix contains infinite values",
            details={'inf_count': int(np.isinf(cov_matrix.values).sum())}
        )

    return cov_matrix


def validate_tickers(
    tickers: Union[str, List[str]],
    min_tickers: int = 2,
    max_tickers: int = 100
) -> List[str]:
    """
    Validate ticker symbols.

    Args:
        tickers: Single ticker or list of tickers
        min_tickers: Minimum number of tickers required
        max_tickers: Maximum number of tickers allowed

    Returns:
        Validated list of ticker symbols

    Raises:
        DataValidationError: If validation fails
    """
    # Convert single ticker to list
    if isinstance(tickers, str):
        tickers = [tickers]

    # Check if list
    if not isinstance(tickers, list):
        raise DataValidationError(
            message="Tickers must be a string or list of strings",
            details={'type_received': type(tickers).__name__}
        )

    # Check all elements are strings
    if not all(isinstance(t, str) for t in tickers):
        non_string = [type(t).__name__ for t in tickers if not isinstance(t, str)]
        raise DataValidationError(
            message="All tickers must be strings",
            details={'non_string_types': non_string}
        )

    # Check for empty strings
    if any(not t.strip() for t in tickers):
        raise DataValidationError(
            message="Tickers cannot be empty strings",
            details={'tickers': tickers}
        )

    # Check minimum number
    if len(tickers) < min_tickers:
        raise DataValidationError(
            message="Too few tickers provided",
            details={
                'tickers_provided': len(tickers),
                'min_required': min_tickers
            }
        )

    # Check maximum number
    if len(tickers) > max_tickers:
        raise DataValidationError(
            message="Too many tickers provided",
            details={
                'tickers_provided': len(tickers),
                'max_allowed': max_tickers
            }
        )

    # Check for duplicates
    unique_tickers = set(tickers)
    if len(unique_tickers) < len(tickers):
        duplicates = [t for t in tickers if tickers.count(t) > 1]
        raise DataValidationError(
            message="Duplicate tickers found",
            details={'duplicates': list(set(duplicates))}
        )

    # Uppercase all tickers (standard format)
    tickers = [t.strip().upper() for t in tickers]

    return tickers


def validate_config(
    config: dict,
    required_keys: Optional[List[str]] = None
) -> dict:
    """
    Validate configuration dictionary.

    Args:
        config: Configuration dictionary
        required_keys: List of required keys

    Returns:
        Validated configuration

    Raises:
        DataValidationError: If validation fails
    """
    if not isinstance(config, dict):
        raise DataValidationError(
            message="Configuration must be a dictionary",
            details={'type_received': type(config).__name__}
        )

    if required_keys:
        missing_keys = [k for k in required_keys if k not in config]
        if missing_keys:
            raise DataValidationError(
                message="Missing required configuration keys",
                details={
                    'missing_keys': missing_keys,
                    'provided_keys': list(config.keys())
                }
            )

    return config


# Example usage
if __name__ == '__main__':
    # Example 1: Validate returns
    try:
        returns = pd.DataFrame(np.random.randn(100, 5) * 0.02)
        validated = validate_returns(returns)
        print("[OK] Returns validation passed")
    except DataValidationError as e:
        print(f"[FAIL] {e}")

    # Example 2: Validate weights
    try:
        weights = pd.Series([0.2, 0.3, 0.25, 0.15, 0.1])
        validated = validate_weights(weights)
        print("[OK] Weights validation passed")
    except DataValidationError as e:
        print(f"[FAIL] {e}")

    # Example 3: Validate strategy
    try:
        strategy = validate_strategy('Max Sharpe')
        print(f"[OK] Strategy '{strategy}' is valid")
    except InvalidStrategyError as e:
        print(f"[FAIL] {e}")

    # Example 4: Invalid strategy
    try:
        strategy = validate_strategy('SuperStrategy')
        print(f"[OK] Strategy '{strategy}' is valid")
    except InvalidStrategyError as e:
        print(f"[FAIL] {e}")

    # Example 5: Validate tickers
    try:
        tickers = validate_tickers(['AAPL', 'GOOGL', 'MSFT', 'AMZN'])
        print(f"[OK] Tickers validated: {tickers}")
    except DataValidationError as e:
        print(f"[FAIL] {e}")
