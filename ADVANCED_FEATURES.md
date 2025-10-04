# Advanced Portfolio Optimization Features

## 🎯 Overview

This document describes the advanced optimization features implemented in Phase 9 of the portfolio optimization project. These features extend the core MIO framework with state-of-the-art quantitative finance techniques.

---

## 📊 Implemented Features

### 1. **Fama-French 5-Factor Model**
**Location**: `src/forecasting/factor_models.py`

Implements the Fama-French 5-factor model for risk decomposition and return forecasting.

**Factors**:
- **Mkt-RF**: Market excess return (market factor)
- **SMB**: Small Minus Big (size factor)
- **HML**: High Minus Low (value factor)
- **RMW**: Robust Minus Weak (profitability factor)
- **CMA**: Conservative Minus Aggressive (investment factor)

**Key Features**:
- Factor data fetching from Kenneth French's library
- Factor loading estimation via time-series regression
- Covariance decomposition: `Cov(R) = B * Cov(F) * B' + Cov(ε)`
- Return forecasting using factor exposures
- Barra-style factor models with industry/style factors

**Usage Example**:
```python
from src.forecasting.factor_models import FamaFrenchFactors

ff_model = FamaFrenchFactors()
factors = ff_model.fetch_factor_data('2020-01-01', '2023-12-31')
result = ff_model.estimate_factor_loadings(asset_returns)

# Get factor loadings (betas)
print(result.factor_loadings)

# Compute total covariance
total_cov = ff_model.compute_total_covariance(result)

# Forecast returns
forecasted = ff_model.forecast_returns(result)
```

**Benefits**:
- Better risk attribution (systematic vs idiosyncratic)
- Improved covariance estimation
- Interpretation of portfolio exposures to common factors

---

### 2. **CVaR (Conditional Value-at-Risk) Optimization**
**Location**: `src/optimization/cvar_optimizer.py`

Portfolio optimization that minimizes tail risk (expected loss beyond VaR).

**Features**:
- Scenario-based CVaR optimization
- Efficient frontier computation
- Robust CVaR under parameter uncertainty
- Monte Carlo scenario generation

**Mathematical Formulation**:
```
minimize: CVaR_α(w) = VaR_α + (1/α) * E[max(0, -R'w - VaR_α)]

subject to: E[R'w] >= min_return
           Σw = 1
           w >= 0
```

**Usage Example**:
```python
from src.optimization.cvar_optimizer import CVaROptimizer, RobustCVaROptimizer

# Standard CVaR optimization
cvar_opt = CVaROptimizer(confidence_level=0.95, n_scenarios=1000)
result = cvar_opt.optimize(
    expected_returns,
    covariance,
    min_return=0.10
)

print(f"CVaR (95%): {result['cvar']:.4f}")
print(f"VaR (95%): {result['var']:.4f}")
print(f"Optimal Weights: {result['weights']}")

# Robust CVaR (accounts for parameter uncertainty)
robust_opt = RobustCVaROptimizer(
    confidence_level=0.95,
    robustness_param=0.1  # 10% uncertainty
)
robust_result = robust_opt.optimize_robust_cvar(
    expected_returns,
    covariance
)
```

**Benefits**:
- Better tail risk management than variance
- Coherent risk measure (subadditive)
- Aligns with regulatory requirements (Basel III)

---

### 3. **Black-Litterman Model**
**Location**: `src/forecasting/black_litterman.py`

Combines market equilibrium returns with investor views to generate superior forecasts.

**Features**:
- Market equilibrium return computation via reverse optimization
- Flexible view specification (absolute and relative)
- Posterior return and covariance estimation
- Integrated portfolio optimization

**Mathematical Formulation**:
```
Equilibrium: Π = δ * Σ * w_mkt

Posterior: E[R] = [(τΣ)^(-1) + P'Ω^(-1)P]^(-1) * [(τΣ)^(-1)Π + P'Ω^(-1)Q]

where:
- Π: implied equilibrium returns
- P: pick matrix (specifies which assets in each view)
- Q: view returns
- Ω: view uncertainty matrix
- τ: scaling parameter
```

**Usage Example**:
```python
from src.forecasting.black_litterman import (
    BlackLittermanModel,
    create_absolute_view,
    create_relative_view
)

bl_model = BlackLittermanModel(risk_aversion=2.5, tau=0.05)

# Define investor views
views = [
    create_absolute_view('AAPL', 0.15, confidence=0.8),  # AAPL will return 15%
    create_relative_view('TSLA', 'AMZN', 0.05, confidence=0.6)  # TSLA outperforms AMZN by 5%
]

# Run Black-Litterman
result = bl_model.run(covariance, views, market_weights)

print("Equilibrium Returns:", result['equilibrium_returns'])
print("Posterior Returns:", result['posterior_returns'])

# Optimize portfolio using posterior returns
from src.forecasting.black_litterman import BlackLittermanOptimizer

bl_optimizer = BlackLittermanOptimizer(bl_model)
optimal_weights = bl_optimizer.optimize(covariance, views, market_weights)
```

**Benefits**:
- Overcomes estimation error in expected returns
- Natural framework for incorporating views
- Avoids extreme portfolio positions

---

### 4. **Multi-Period Optimization**
**Location**: `src/optimization/multiperiod_optimizer.py`

Dynamic programming for sequential portfolio decisions across multiple periods.

**Features**:
- Deterministic multi-period optimization
- Stochastic dynamic programming with scenario trees
- Backward induction algorithm
- Threshold-based rebalancing policies
- Transaction cost modeling across periods

**Formulation**:
```
maximize: E[Σ_t γ^t U(W_t)]

subject to: W_{t+1} = W_t * (1 + R_t'w_t) - TC(w_{t-1}, w_t)
           w_t constraints (budget, bounds, cardinality)

where:
- W_t: wealth at time t
- γ: discount factor
- U: utility function
- TC: transaction costs
```

**Usage Example**:
```python
from src.optimization.multiperiod_optimizer import (
    MultiPeriodOptimizer,
    MultiPeriodConfig,
    ThresholdRebalancingPolicy
)

# Deterministic optimization
config = MultiPeriodConfig(
    n_periods=12,
    risk_aversion=2.5,
    transaction_cost=0.002
)

optimizer = MultiPeriodOptimizer(config)

result = optimizer.deterministic_multi_period(
    returns_path,  # Expected returns for each period
    cov_path,      # Covariance for each period
    initial_wealth=100.0
)

print(f"Final Wealth: ${result['final_wealth']:.2f}")
print(f"Total Return: {result['total_return']*100:.2f}%")

# Threshold rebalancing
rebalancer = ThresholdRebalancingPolicy(
    target_weights=np.array([0.4, 0.4, 0.2]),
    threshold=0.05,
    transaction_cost=0.001
)

if rebalancer.should_rebalance(current_weights):
    new_weights, cost = rebalancer.rebalance(current_weights, wealth=100)
```

**Benefits**:
- Accounts for transaction costs dynamically
- Optimal timing of rebalancing
- Considers long-term consequences of decisions

---

### 5. **Short-Selling & Leverage Constraints**
**Location**: `src/optimization/mio_optimizer.py` (extended)

Enhanced MIO optimizer supporting short positions and leverage.

**Features**:
- Separate long and short position variables
- Gross exposure constraints (long + short)
- Net exposure constraints (long - short)
- Binary indicators prevent simultaneous long/short

**Constraints**:
```
Net Exposure: Σw_long - Σw_short = target_exposure
Gross Exposure: Σw_long + Σw_short <= max_leverage
Cannot be both: y_long[i] + y_short[i] <= 1
```

**Usage Example**:
```python
from src.optimization.mio_optimizer import MIOOptimizer, OptimizationConfig

# Enable short-selling
config = OptimizationConfig(
    allow_short_selling=True,
    max_short_weight=0.20,     # Max 20% short per asset
    max_leverage=1.5,          # Max 150% gross exposure
    net_exposure=1.0,          # Maintain 100% net exposure
    max_assets=10
)

optimizer = MIOOptimizer(config)
weights = optimizer.optimize(expected_returns, covariance)

# Access detailed solution
print(f"Long positions: {optimizer.solution['long_weights']}")
print(f"Short positions: {optimizer.solution['short_weights']}")
print(f"Gross exposure: {optimizer.solution['gross_exposure']:.2%}")
print(f"Net exposure: {optimizer.solution['net_exposure']:.2%}")
```

**Benefits**:
- Relaxes long-only constraint for higher returns
- Market-neutral strategies possible
- Better hedging capabilities

---

### 6. **LSTM Neural Networks for Forecasting**
**Location**: `src/forecasting/lstm_forecast.py`

Deep learning models for time-series return prediction.

**Features**:
- Univariate and multivariate LSTM
- Bidirectional LSTM layers
- Attention-enhanced LSTM
- Ensemble LSTM forecasting
- Automatic sequence generation
- Fallback to EWMA when TensorFlow unavailable

**Architecture**:
- Input: Historical returns (lookback_window × n_assets)
- LSTM layers with dropout for regularization
- Attention mechanism for long-term dependencies
- Dense output layer for multi-asset prediction

**Usage Example**:
```python
from src.forecasting.lstm_forecast import LSTMForecaster, EnsembleLSTMForecaster

# Basic LSTM
lstm = LSTMForecaster(
    lookback_window=60,
    forecast_horizon=1,
    hidden_units=[64, 32],
    dropout_rate=0.2,
    epochs=100
)

history = lstm.fit(historical_returns, validation_split=0.2)
predictions = lstm.predict(recent_returns, n_steps=5)

print(f"5-step ahead predictions:\n{predictions}")

# Ensemble for robustness
ensemble = EnsembleLSTMForecaster(
    n_models=5,
    lookback_window=60,
    hidden_units=[32, 16]
)

ensemble.fit(historical_returns)
ensemble_pred = ensemble.predict(recent_returns, n_steps=5)
```

**Benefits**:
- Captures non-linear patterns
- Handles long-term dependencies
- Ensemble reduces overfitting

---

## 🧪 Testing

All features are tested in `tests/test_advanced_features.py`:

```python
# Run all advanced feature tests
pytest tests/test_advanced_features.py -v

# Run specific test class
pytest tests/test_advanced_features.py::TestCVaROptimization -v
```

**Test Coverage**:
- ✅ Fama-French factor estimation (4 tests)
- ✅ CVaR optimization (4 tests)
- ✅ Black-Litterman model (4 tests)
- ✅ Multi-period optimization (3 tests)
- ✅ Short-selling constraints (2 tests)

---

## 📦 Dependencies

New dependencies added to `requirements.txt`:

```
# For CVaR optimization
cvxpy>=1.3.0

# For factor models
pandas-datareader>=0.10.0

# For LSTM (optional)
tensorflow>=2.13.0
```

Install all dependencies:
```bash
pip install -r requirements.txt
```

---

## 📈 Performance Metrics

### Code Statistics
- **New Lines of Code**: ~3,000
- **New Modules**: 5
- **New Test Cases**: 17
- **Total Project LOC**: 12,000+

### Optimization Methods Now Available
1. Mean-Variance Optimization (MVO)
2. Mixed-Integer Optimization (MIO)
3. CVaR Optimization
4. Robust CVaR
5. Black-Litterman
6. Multi-Period DP
7. Genetic Algorithm
8. Simulated Annealing
9. Factor-Based Optimization
10. LSTM-Enhanced Optimization

---

## 🚀 Quick Start

### Example: Complete Advanced Workflow

```python
import numpy as np
import pandas as pd
from src.data.real_data_loader import RealAssetDataLoader
from src.forecasting.factor_models import FamaFrenchFactors
from src.forecasting.black_litterman import BlackLittermanModel, create_absolute_view
from src.optimization.cvar_optimizer import CVaROptimizer
from src.optimization.multiperiod_optimizer import MultiPeriodOptimizer

# 1. Load data
loader = RealAssetDataLoader()
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
prices = loader.fetch_prices(tickers, '2020-01-01', '2023-12-31')
returns = loader.compute_returns(prices)

# 2. Factor model for covariance
ff_model = FamaFrenchFactors()
factors = ff_model.fetch_factor_data('2020-01-01', '2023-12-31')
factor_result = ff_model.estimate_factor_loadings(returns)
covariance = ff_model.compute_total_covariance(factor_result)

# 3. Black-Litterman for expected returns
bl_model = BlackLittermanModel(risk_aversion=2.5)
views = [
    create_absolute_view('NVDA', 0.20, confidence=0.75),
    create_absolute_view('AAPL', 0.12, confidence=0.60)
]
market_weights = pd.Series([0.25, 0.20, 0.20, 0.20, 0.15], index=tickers)
bl_result = bl_model.run(covariance, views, market_weights)
expected_returns = bl_result['posterior_returns']

# 4. CVaR optimization
cvar_opt = CVaROptimizer(confidence_level=0.95)
cvar_result = cvar_opt.optimize(
    expected_returns.values,
    covariance.values,
    min_return=0.12
)

print("Optimal Portfolio:")
print(f"  CVaR (95%): {cvar_result['cvar']:.4f}")
print(f"  Expected Return: {cvar_result['expected_return']:.4f}")
print(f"  Weights: {cvar_result['weights']}")
```

---

## 📚 References

1. **Fama-French Model**: Fama, E. F., & French, K. R. (2015). "A five-factor asset pricing model"
2. **CVaR**: Rockafellar, R. T., & Uryasev, S. (2000). "Optimization of conditional value-at-risk"
3. **Black-Litterman**: Black, F., & Litterman, R. (1992). "Global Portfolio Optimization"
4. **Multi-Period**: Berkelaar, A., & Kouwenberg, R. (2000). "Dynamic asset allocation"
5. **LSTM**: Hochreiter, S., & Schmidhuber, J. (1997). "Long short-term memory"

---

## 🎯 Next Steps

Potential future enhancements:
- Reinforcement learning for adaptive rebalancing
- Transformer models for return prediction
- ESG integration with factor models
- Real-time portfolio monitoring
- Live broker integration

---

**Last Updated**: October 2025
**Version**: 1.1.0
**Status**: ✅ Production Ready
