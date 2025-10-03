"""
Test that Streamlit app can be imported and initialized without errors.
"""

import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def test_dashboard_imports():
    """Test that all required imports work."""
    try:
        import streamlit as st
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        from datetime import datetime
        assert True
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")


def test_dashboard_module_loads():
    """Test that dashboard module can be imported."""
    try:
        from src.visualization import dashboard
        assert hasattr(dashboard, 'generate_synthetic_data')
        assert hasattr(dashboard, 'optimize_portfolio')
        assert hasattr(dashboard, 'evaluate_portfolio')
        assert hasattr(dashboard, 'main')
    except Exception as e:
        pytest.fail(f"Dashboard module load failed: {e}")


def test_dashboard_functions_callable():
    """Test that all main functions are callable."""
    from src.visualization.dashboard import (
        generate_synthetic_data,
        optimize_portfolio,
        evaluate_portfolio,
        main
    )

    assert callable(generate_synthetic_data)
    assert callable(optimize_portfolio)
    assert callable(evaluate_portfolio)
    assert callable(main)


def test_dashboard_functions_work():
    """Test that dashboard functions execute without errors."""
    from src.visualization.dashboard import (
        generate_synthetic_data,
        optimize_portfolio,
        evaluate_portfolio
    )

    # Test data generation
    prices, returns = generate_synthetic_data(5, 100, 42)
    assert prices is not None
    assert returns is not None

    # Test optimization
    weights, annual_returns, cov_matrix = optimize_portfolio(returns, 'Equal Weight')
    assert weights is not None
    assert annual_returns is not None
    assert cov_matrix is not None

    # Test evaluation
    metrics = evaluate_portfolio(weights, annual_returns, cov_matrix)
    assert metrics is not None
    assert 'return' in metrics
    assert 'volatility' in metrics
    assert 'sharpe' in metrics


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
