# System Architecture

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     PORTFOLIO OPTIMIZATION SYSTEM                    │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────┐
│  DATA SOURCES   │
│                 │
│  • Yahoo Finance│
│  • Quandl       │
│  • Bloomberg    │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                                   │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐         ┌──────────────────────┐             │
│  │  AssetDataLoader │────────▶│  DataPreprocessor    │             │
│  │                  │         │                      │             │
│  │ • fetch_prices() │         │ • calculate_factors()│             │
│  │ • compute_returns│         │ • winsorize_returns()│             │
│  │ • handle_missing │         │ • rolling_windows()  │             │
│  └──────────────────┘         └──────────────────────┘             │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      FORECASTING LAYER                               │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌────────────────┐  ┌──────────────────┐        │
│  │   Returns    │  │   Volatility   │  │   Covariance     │        │
│  │  Forecaster  │  │   Forecaster   │  │    Estimator     │        │
│  │              │  │                │  │                  │        │
│  │ • ARIMA      │  │ • GARCH        │  │ • Ledoit-Wolf    │        │
│  │ • VAR        │  │ • EGARCH       │  │ • OAS            │        │
│  │ • ML Models  │  │ • EWMA         │  │ • Exponential    │        │
│  └──────────────┘  └────────────────┘  └──────────────────┘        │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   ML-DRIVEN HEURISTICS LAYER                         │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐         ┌──────────────────────┐             │
│  │ Asset Clustering │         │  Constraint          │             │
│  │                  │         │  Prediction          │             │
│  │ • K-Means        │         │                      │             │
│  │ • Hierarchical   │         │ • Random Forest      │             │
│  │ • Select Diverse │         │ • Binding Constraints│             │
│  └──────────────────┘         └──────────────────────┘             │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      OPTIMIZATION LAYER                              │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌────────────┐  ┌───────────────────┐           │
│  │     MIO      │  │  Genetic   │  │    Simulated      │           │
│  │  Optimizer   │  │ Algorithm  │  │    Annealing      │           │
│  │              │  │            │  │                   │           │
│  │ • PuLP/CVXPY │  │ • Evolution│  │ • Cooling Schedule│           │
│  │ • CBC Solver │  │ • Crossover│  │ • Energy Function │           │
│  │ • Constraints│  │ • Mutation │  │ • Metropolis      │           │
│  └──────────────┘  └────────────┘  └───────────────────┘           │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      BACKTESTING LAYER                               │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────┐           │
│  │         Rolling Window Backtest Engine                │           │
│  │                                                        │           │
│  │  For each time window:                                │           │
│  │    1. Train forecasters on historical data           │           │
│  │    2. Forecast returns, volatility, covariance       │           │
│  │    3. Optimize portfolio                             │           │
│  │    4. Execute trades (with costs)                    │           │
│  │    5. Record performance                             │           │
│  │                                                        │           │
│  │  Metrics: Sharpe, Sortino, Max DD, Turnover         │           │
│  └──────────────────────────────────────────────────────┘           │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    VISUALIZATION LAYER                               │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌────────────┐  ┌──────────────┐                │
│  │    Static    │  │ Interactive│  │   Dashboard  │                │
│  │    Plots     │  │   Charts   │  │              │                │
│  │              │  │            │  │              │                │
│  │ • Matplotlib │  │ • Plotly   │  │ • Streamlit  │                │
│  │ • Seaborn    │  │ • Bokeh    │  │ • Real-time  │                │
│  │ • PNG Export │  │ • HTML     │  │ • Interactive│                │
│  └──────────────┘  └────────────┘  └──────────────┘                │
└─────────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      DEPLOYMENT LAYER                                │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌────────────┐  ┌──────────────┐                │
│  │   FastAPI    │  │   Docker   │  │   Airflow    │                │
│  │     API      │  │ Container  │  │  Pipeline    │                │
│  │              │  │            │  │              │                │
│  │ • REST API   │  │ • Isolation│  │ • Scheduling │                │
│  │ • Auth       │  │ • Portable │  │ • Monitoring │                │
│  │ • Rate Limit │  │ • Scalable │  │ • Alerts     │                │
│  └──────────────┘  └────────────┘  └──────────────┘                │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

```
Raw Prices
    │
    ├─▶ Preprocessing
    │       │
    │       ├─▶ Returns Calculation
    │       ├─▶ Factor Computation
    │       └─▶ Outlier Handling
    │
    ▼
Returns Data
    │
    ├─▶ Forecasting
    │       │
    │       ├─▶ Expected Returns (ARIMA/VAR)
    │       ├─▶ Volatility (GARCH)
    │       └─▶ Covariance (Shrinkage)
    │
    ▼
Forecasted Inputs
    │
    ├─▶ ML Heuristics
    │       │
    │       ├─▶ Asset Clustering (K-Means)
    │       └─▶ Pre-selection (Top K from each cluster)
    │
    ▼
Reduced Asset Universe
    │
    ├─▶ Optimization
    │       │
    │       ├─▶ MIO Solver (Exact)
    │       ├─▶ Genetic Algorithm (Heuristic)
    │       └─▶ Simulated Annealing (Heuristic)
    │
    ▼
Optimal Weights
    │
    ├─▶ Backtesting
    │       │
    │       └─▶ Performance Metrics
    │
    ▼
Results & Visualizations
```

---

## Component Interactions

### 1. Data Acquisition Flow
```python
AssetDataLoader.fetch_prices()
    └─▶ Yahoo Finance API
         └─▶ Cache to data/raw/
              └─▶ Return DataFrame
```

### 2. Preprocessing Flow
```python
DataPreprocessor.calculate_factors()
    ├─▶ Momentum (6-month return)
    ├─▶ Volatility (annualized std)
    ├─▶ Size (market cap)
    └─▶ Beta (market sensitivity)
         └─▶ Standardize (mean=0, std=1)
```

### 3. Optimization Flow
```python
MIOOptimizer.optimize()
    ├─▶ Formulate MIP Problem
    │    ├─▶ Decision variables (weights, indicators)
    │    ├─▶ Objective (return - risk - costs)
    │    └─▶ Constraints (budget, cardinality, bounds)
    │
    ├─▶ Solve with PuLP/CBC
    │    └─▶ Branch & Bound algorithm
    │
    └─▶ Extract Solution
         └─▶ Normalize weights
```

### 4. Genetic Algorithm Flow
```python
GeneticOptimizer.optimize()
    ├─▶ Initialize population (random portfolios)
    │
    ├─▶ For each generation:
    │    ├─▶ Evaluate fitness (Sharpe ratio)
    │    ├─▶ Select parents (tournament)
    │    ├─▶ Crossover (blend weights)
    │    ├─▶ Mutate (perturb weights)
    │    └─▶ Repair (satisfy constraints)
    │
    └─▶ Return best solution
```

---

## Design Patterns Used

### 1. Strategy Pattern
Different optimization algorithms implement common interface:
```python
class Optimizer:
    def optimize(self, returns, covariance) -> weights
```

Implementations:
- `MIOOptimizer`
- `GeneticOptimizer`
- `SimulatedAnnealingOptimizer`
- `NaiveMeanVarianceOptimizer`

### 2. Factory Pattern
Forecaster selection based on method:
```python
class ReturnsForecast:
    def __init__(self, method='arima'):
        if method == 'arima':
            self.model = ARIMA(...)
        elif method == 'var':
            self.model = VAR(...)
```

### 3. Template Method Pattern
Backtesting follows fixed steps:
```python
class Backtester:
    def run_backtest(self):
        for window in rolling_windows:
            self._train_forecasters(window)
            self._forecast_inputs()
            self._optimize_portfolio()
            self._execute_trades()
            self._record_metrics()
```

### 4. Builder Pattern
Optimization configuration:
```python
config = OptimizationConfig()
    .set_risk_aversion(2.5)
    .set_max_assets(5)
    .set_transaction_costs(0.001, 0.0005)
```

---

## Scalability Considerations

### Current Capacity
- **Assets**: Up to 50 (MIO), 100+ (GA)
- **Time Series Length**: 5+ years of daily data
- **Optimization Time**: 2-30 seconds (depending on method)

### Optimization Strategies
1. **Clustering Pre-selection**: Reduces problem size by 50-70%
2. **Warm Starting**: Use previous solution as initial point
3. **Parallelization**: Genetic algorithm population evaluation
4. **Caching**: Store computed covariance matrices

### Future Improvements
- GPU acceleration for matrix operations
- Distributed computing for backtesting
- Incremental updates (don't recompute everything)
- Approximate solvers for very large problems

---

## Error Handling Strategy

```
┌─────────────────────────────────────────────┐
│            Error Handling Flow               │
├─────────────────────────────────────────────┤
│                                              │
│  Data Loading Error                         │
│    ├─▶ Check cache                          │
│    ├─▶ Retry with exponential backoff       │
│    └─▶ Fallback to alternative data source  │
│                                              │
│  Optimization Failure                       │
│    ├─▶ Log solver status                    │
│    ├─▶ Relax constraints                    │
│    └─▶ Fallback to heuristic method         │
│                                              │
│  Singular Covariance Matrix                 │
│    ├─▶ Apply shrinkage                      │
│    ├─▶ Add diagonal regularization          │
│    └─▶ Use diagonal approximation           │
│                                              │
│  Missing Data                                │
│    ├─▶ Forward fill                         │
│    ├─▶ Interpolation                        │
│    └─▶ Drop asset if >50% missing           │
│                                              │
└─────────────────────────────────────────────┘
```

---

## Performance Benchmarks

| Operation | Time (Avg) | Memory |
|-----------|------------|--------|
| Load 20 assets, 4 years | 3.2s | 50 MB |
| Compute returns & factors | 0.8s | 20 MB |
| GARCH estimation (10 assets) | 5.1s | 30 MB |
| MIO optimization (10 assets) | 2.3s | 40 MB |
| Genetic Algorithm (100 gen) | 12.8s | 60 MB |
| Generate 5 visualizations | 4.5s | 80 MB |
| **Total Pipeline** | **28.7s** | **280 MB** |

*Tested on: Intel i7-10700K, 16GB RAM, Windows 11*

---

## Security Considerations

### API Keys
- Store in `.env` file (not committed to Git)
- Use environment variables for production
- Rotate keys periodically

### Data Privacy
- No PII stored
- Market data only
- Comply with data provider ToS

### Deployment Security
- Rate limiting on API endpoints
- Input validation and sanitization
- HTTPS for production
- Authentication tokens for sensitive operations

---

## Monitoring & Logging

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/portfolio.log'),
        logging.StreamHandler()
    ]
)
```

### Key Metrics to Monitor
1. **Data Quality**: Missing data %, outliers detected
2. **Optimization**: Solve time, convergence status
3. **Performance**: Sharpe ratio, max drawdown
4. **System**: CPU usage, memory consumption

---

**Last Updated:** October 3, 2025
