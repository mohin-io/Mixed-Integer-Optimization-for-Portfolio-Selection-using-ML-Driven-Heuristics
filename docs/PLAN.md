# Mixed-Integer Optimization for Portfolio Selection
## Practical Portfolio Construction with Transaction Costs and Constraints using ML-Driven Heuristics

**Author:** Mohin Hasin
**GitHub:** [mohin-io](https://github.com/mohin-io)
**Email:** mohinhasin999@gmail.com
**Project Start Date:** October 2025

---

## 🎯 Project Overview

This project tackles **real-world portfolio optimization** by addressing limitations in classical mean-variance optimization:
- **Integer constraints**: Discrete trading units (can't buy fractional shares)
- **Transaction costs**: Fixed and proportional costs per trade
- **Cardinality constraints**: Limited number of assets to reduce monitoring overhead

### Why This Matters
Traditional portfolio optimization assumes frictionless markets and continuous asset allocation. In practice:
- Small funds face minimum lot sizes
- Transaction costs make frequent rebalancing expensive
- Fund managers need concentrated portfolios for operational efficiency

### Innovation: ML-Driven Heuristics
We enhance mixed-integer optimization (MIO) with machine learning:
1. **Asset Pre-selection**: Clustering algorithms identify diverse asset subsets
2. **Constraint Prediction**: ML models predict which constraints will bind
3. **Guided Search**: Genetic algorithms and simulated annealing find near-optimal solutions faster

---

## 📋 Step-by-Step Implementation Plan

### Phase 1: Foundation & Data Infrastructure (Days 1-2)

#### Step 1.1: Environment Setup
- [ ] Create virtual environment (`Python 3.10+`)
- [ ] Install core dependencies:
  - **Data**: `yfinance`, `pandas`, `numpy`
  - **Optimization**: `pyomo`, `pulp`, `cvxpy`
  - **ML**: `scikit-learn`, `xgboost`
  - **Forecasting**: `statsmodels`, `arch` (GARCH)
  - **Visualization**: `matplotlib`, `seaborn`, `plotly`
  - **Dashboard**: `streamlit`, `dash`
  - **API**: `fastapi`, `uvicorn`
  - **Testing**: `pytest`, `pytest-cov`

#### Step 1.2: Data Sourcing Module (`src/data/loader.py`)
```python
# Responsibilities:
# - Fetch historical price data (Yahoo Finance API)
# - Support multiple asset classes (stocks, ETFs, crypto)
# - Handle missing data, splits, dividends
# - Cache data locally for reproducibility
```

**Implementation:**
1. Create `AssetDataLoader` class with methods:
   - `fetch_prices(tickers, start_date, end_date)`
   - `compute_returns(prices, frequency='daily')`
   - `handle_missing_data(method='forward_fill')`
2. Store raw data in `data/raw/`
3. **Visual Output**: Price time series plot for sample portfolio

#### Step 1.3: Data Preprocessing (`src/data/preprocessing.py`)
```python
# Responsibilities:
# - Compute rolling windows for out-of-sample testing
# - Calculate risk factors (size, value, momentum)
# - Generate covariance matrices
# - Handle outliers and winsorization
```

**Implementation:**
1. Create `DataPreprocessor` class:
   - `compute_rolling_windows(window_size=252)`
   - `calculate_factors()`
   - `winsorize_returns(quantile=0.01)`
2. Store processed data in `data/processed/`
3. **Visual Output**: Correlation heatmap, factor exposures

---

### Phase 2: Forecasting Models (Days 3-4)

#### Step 2.1: Returns Forecasting (`src/forecasting/returns_forecast.py`)
**Models to Implement:**
- **ARIMA**: Autoregressive Integrated Moving Average
- **VAR**: Vector Autoregression (captures cross-asset dynamics)
- **ML Baseline**: Random Forest Regressor

**Implementation:**
1. Create `ReturnsForecast` class with methods:
   - `fit_arima(order=(1,0,1))`
   - `fit_var(lags=5)`
   - `fit_ml_model(features=['momentum', 'volatility'])`
2. Out-of-sample evaluation (train on 80%, test on 20%)
3. **Visual Output**: Predicted vs actual returns scatter plot

#### Step 2.2: Volatility Forecasting (`src/forecasting/volatility_forecast.py`)
**Models:**
- **GARCH(1,1)**: Generalized Autoregressive Conditional Heteroskedasticity
- **EGARCH**: Exponential GARCH (asymmetric shocks)

**Implementation:**
1. Create `VolatilityForecast` class:
   - `fit_garch(p=1, q=1)`
   - `fit_egarch(p=1, q=1)`
2. Forecast rolling volatility
3. **Visual Output**: Realized vs forecasted volatility time series

#### Step 2.3: Covariance Matrix Estimation (`src/forecasting/covariance.py`)
**Methods:**
- **Sample Covariance**: Baseline
- **Ledoit-Wolf Shrinkage**: Reduces estimation error
- **Factor Models**: Fama-French 3-factor

**Implementation:**
1. Create `CovarianceEstimator` class
2. Compare condition numbers across methods
3. **Visual Output**: Eigenvalue distribution plot

---

### Phase 3: Mixed-Integer Optimization Core (Days 5-6)

#### Step 3.1: Problem Formulation (`src/optimization/mio_optimizer.py`)

**Mathematical Model:**
```
maximize:   μᵀw - λ·(wᵀΣw) - transaction_costs(w, w_prev)

subject to:
    1. Σwᵢ = 1                    (budget constraint)
    2. wᵢ ∈ {0, l, 2l, ..., u}    (integer lots)
    3. Σyᵢ ≤ k                     (cardinality: max k assets)
    4. yᵢ ∈ {0,1}, wᵢ ≤ yᵢ         (binary indicators)
    5. wᵢ ≥ 0                      (long-only)

where:
    transaction_costs = Σ(fixed_cost·yᵢ + proportional_cost·|wᵢ - w_prev,ᵢ|)
```

**Implementation:**
1. Create `MIOOptimizer` class using `pyomo` or `pulp`
2. Support solvers: CPLEX (if licensed), CBC, GLPK
3. Parameters:
   - `risk_aversion` (λ)
   - `max_assets` (k)
   - `lot_size` (l)
   - `fixed_cost`, `proportional_cost`
4. **Visual Output**: Efficient frontier with transaction costs

#### Step 3.2: Solver Integration
1. Implement solver wrappers for multiple backends
2. Timeout handling for large problems
3. Log solver statistics (iterations, gap, runtime)

---

### Phase 4: ML-Driven Heuristics (Days 7-9) 🌟

#### Step 4.1: Asset Clustering (`src/heuristics/clustering.py`)

**Purpose**: Pre-select diverse assets to reduce problem size

**Algorithms:**
- **K-Means**: Group assets by return correlation
- **Hierarchical Clustering**: Dendrogram-based selection

**Implementation:**
1. Create `AssetClusterer` class:
   - `fit_kmeans(n_clusters=10)`
   - `fit_hierarchical(linkage='ward')`
   - `select_representatives(n_per_cluster=2)`
2. Feature engineering: Use correlation matrix + volatility
3. **Visual Output**: Dendrogram, cluster scatter plot (t-SNE)

#### Step 4.2: Constraint Prediction (`src/heuristics/constraint_predictor.py`)

**Purpose**: Predict which constraints will bind to prune search space

**ML Model:**
- **Random Forest Classifier**: Predicts if cardinality/cost constraints are active
- **Features**: Market volatility, portfolio turnover, asset dispersion

**Implementation:**
1. Train on historical optimization solutions
2. Use predictions to initialize heuristic search
3. **Visual Output**: Feature importance plot

#### Step 4.3: Genetic Algorithm (`src/heuristics/genetic_algorithm.py`)

**Purpose**: Combinatorial search for asset selection

**Algorithm:**
```
1. Initialize population of random portfolios
2. Evaluate fitness (Sharpe ratio - costs)
3. Selection (tournament selection)
4. Crossover (blend weights from two parents)
5. Mutation (randomly adjust weights)
6. Repeat for N generations
```

**Implementation:**
1. Create `GeneticOptimizer` class:
   - `initialize_population(size=100)`
   - `evaluate_fitness()`
   - `evolve(generations=50)`
2. ML guidance: Use clustering to seed initial population
3. **Visual Output**: Fitness convergence plot

#### Step 4.4: Simulated Annealing (`src/heuristics/simulated_annealing.py`)

**Purpose**: Escape local optima in non-convex cost landscape

**Implementation:**
1. Create `SimulatedAnnealingOptimizer` class
2. Cooling schedule: Exponential decay
3. **Visual Output**: Energy landscape over iterations

---

### Phase 5: Backtesting Framework (Days 10-11)

#### Step 5.1: Backtesting Engine (`src/backtesting/engine.py`)

**Rolling Window Strategy:**
```
for each window t:
    1. Train forecasting models on data[t-train_window:t]
    2. Forecast returns, volatility for period t+1
    3. Solve optimization problem
    4. Execute trades (simulate slippage)
    5. Record performance metrics
```

**Implementation:**
1. Create `Backtester` class:
   - `run_backtest(start_date, end_date, rebalance_freq='monthly')`
   - `calculate_metrics()`: Sharpe, Sortino, max drawdown, turnover
2. Transaction cost accounting
3. **Visual Output**: Cumulative returns plot

#### Step 5.2: Benchmark Comparison (`src/backtesting/benchmarks.py`)

**Strategies to Compare:**
1. **Naïve Mean-Variance**: No transaction costs, fractional weights
2. **Exact MIO Solver**: CPLEX/Gurobi with strict optimality
3. **Equal-Weight Portfolio**: 1/N allocation
4. **ML-Guided Heuristics**: Our approach

**Metrics:**
- Risk-adjusted return (Sharpe ratio)
- Realized transaction costs
- CPU runtime
- Portfolio turnover

**Visual Output**: Side-by-side performance table + bar charts

---

### Phase 6: Visualization & Reporting (Days 12-13)

#### Step 6.1: Static Visualizations (`src/visualization/plots.py`)

**Required Plots:**
1. **Price & Returns**: Time series of asset prices and log-returns
2. **Correlation Matrix**: Heatmap of asset correlations
3. **Factor Exposures**: Bar chart of size/value/momentum loadings
4. **Forecasting Performance**: Predicted vs actual scatter plots
5. **Efficient Frontier**: Risk-return trade-off with transaction costs
6. **Portfolio Weights**: Stacked area chart over time
7. **Performance Metrics**: Cumulative returns, drawdowns
8. **Heuristic Convergence**: GA/SA fitness over iterations
9. **Runtime Comparison**: Bar chart of solver times
10. **Cost Analysis**: Transaction costs breakdown

**Implementation:**
1. Create reusable plotting functions
2. Save all figures to `outputs/figures/` with timestamps
3. Use consistent styling (seaborn 'whitegrid')

#### Step 6.2: Streamlit Dashboard (`src/visualization/dashboard.py`)

**Interactive Features:**
- **Sidebar**: Adjust risk aversion, max assets, transaction costs
- **Tab 1**: Data exploration (price charts, correlations)
- **Tab 2**: Forecasting results (predicted returns, volatility)
- **Tab 3**: Optimization results (portfolio weights, efficient frontier)
- **Tab 4**: Backtesting (cumulative returns, metrics table)
- **Tab 5**: Heuristic comparison (runtime, performance trade-offs)

**Implementation:**
```bash
streamlit run src/visualization/dashboard.py
```

---

### Phase 7: API & Deployment (Days 14-15)

#### Step 7.1: FastAPI Service (`src/api/main.py`)

**Endpoints:**
1. `POST /optimize`: Submit optimization request
   ```json
   {
     "tickers": ["AAPL", "GOOGL", "MSFT"],
     "risk_aversion": 2.5,
     "max_assets": 5,
     "method": "genetic_algorithm"
   }
   ```
2. `GET /portfolio/{job_id}`: Retrieve optimization results
3. `GET /backtest/{portfolio_id}`: Get backtest metrics

**Implementation:**
1. Async task queue (Celery or FastAPI background tasks)
2. Input validation with Pydantic models
3. Error handling and logging

#### Step 7.2: Docker Containerization

**Dockerfile:**
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**docker-compose.yml:**
```yaml
services:
  api:
    build: .
    ports:
      - "8000:8000"
  dashboard:
    build: .
    command: streamlit run src/visualization/dashboard.py
    ports:
      - "8501:8501"
```

---

### Phase 8: Testing & Documentation (Day 16)

#### Step 8.1: Unit Tests (`tests/`)
- `test_data_loader.py`: Verify data fetching
- `test_forecasting.py`: Check model predictions
- `test_optimization.py`: Validate constraint satisfaction
- `test_heuristics.py`: Ensure convergence

**Target Coverage**: >80%

#### Step 8.2: README.md Structure

```markdown
# Project Title
[Badge: Python Version] [Badge: License] [Badge: Tests Passing]

## 🚀 Quick Start
# Installation and 3-line demo

## 📊 Key Results
[Embed: Performance comparison chart]
[Embed: Portfolio weights visualization]

## 🏗️ Architecture
[Diagram: System components]

## 📈 Methodology
# Brief explanation of MIO + ML approach

## 🔧 Usage
# Code examples

## 📂 Project Structure
# Directory tree

## 🤝 Contributing
## 📄 License
```

---

## 📊 Expected Outputs & Deliverables

### Simulations (stored in `outputs/simulations/`)
1. **Baseline Run**: 10 assets, 3-year backtest
2. **Scalability Test**: 50+ assets
3. **Cost Sensitivity**: Vary transaction costs
4. **Constraint Analysis**: Cardinality limits (5, 10, 15 assets)

### Visuals (stored in `outputs/figures/`)
- 10+ high-quality plots (PNG, 300 DPI)
- Each with descriptive filename (e.g., `efficient_frontier_with_costs.png`)
- Captions in `outputs/figures/README.md`

### Documentation
- `docs/PLAN.md` (this file)
- `docs/METHODOLOGY.md`: Mathematical formulations
- `docs/RESULTS.md`: Simulation outcomes
- `README.md`: Project showcase

---

## 🔄 Git Workflow & Commits

### Commit Strategy (Atomic Commits)
1. **Initial setup**: "chore: initialize project structure and planning docs"
2. **Data module**: "feat: implement asset data loader with Yahoo Finance integration"
3. **Preprocessing**: "feat: add data preprocessing with factor computation"
4. **Forecasting (returns)**: "feat: implement ARIMA and VAR returns forecasting"
5. **Forecasting (volatility)**: "feat: add GARCH volatility forecasting"
6. **Covariance estimation**: "feat: implement Ledoit-Wolf covariance estimation"
7. **MIO core**: "feat: build mixed-integer optimization solver with transaction costs"
8. **Clustering**: "feat: add K-Means and hierarchical asset clustering"
9. **Constraint prediction**: "feat: implement ML-based constraint prediction"
10. **Genetic algorithm**: "feat: add genetic algorithm heuristic optimizer"
11. **Simulated annealing**: "feat: implement simulated annealing optimizer"
12. **Backtesting engine**: "feat: create rolling window backtesting framework"
13. **Benchmarks**: "feat: add benchmark strategies for comparison"
14. **Visualization**: "feat: generate static plots for all components"
15. **Dashboard**: "feat: build Streamlit interactive dashboard"
16. **API**: "feat: implement FastAPI optimization service"
17. **Docker**: "chore: add Docker containerization setup"
18. **Tests**: "test: add unit tests with >80% coverage"
19. **README**: "docs: create comprehensive README with visuals and quickstart"
20. **Final simulation**: "feat: run complete backtest and generate all outputs"

### Git Configuration
```bash
git config user.name "mohin-io"
git config user.email "mohinhasin999@gmail.com"
```

### Push Sequence
- Push after every 3-5 logical commits
- Ensure all tests pass before pushing
- Tag releases: `v1.0.0-alpha`, `v1.0.0-beta`, `v1.0.0`

---

## 🎯 Success Criteria

### Technical Goals
- [ ] MIO solver handles 50+ assets in <5 minutes
- [ ] Heuristics achieve 95%+ of exact solver performance
- [ ] ML guidance reduces search time by 30%+
- [ ] Backtested Sharpe ratio > 1.0 (after costs)

### Presentation Goals
- [ ] README has embedded visuals (no external links)
- [ ] All plots have clear titles, labels, legends
- [ ] Code is PEP8 compliant and documented
- [ ] Dashboard runs smoothly on Streamlit Cloud

### Recruiter Impact
- [ ] GitHub profile pinned repository
- [ ] LinkedIn post with key visualizations
- [ ] Portfolio page with live demo link

---

## 📚 References & Resources

### Academic Papers
1. Bertsimas & Shioda (2009): "Algorithm for cardinality-constrained quadratic optimization"
2. Woodside-Oriakhi et al. (2011): "Heuristic algorithms for portfolio optimization"
3. Ledoit & Wolf (2004): "Honey, I shrunk the sample covariance matrix"

### Libraries Documentation
- [Pyomo Optimization](https://pyomo.readthedocs.io/)
- [ARCH GARCH Models](https://arch.readthedocs.io/)
- [Streamlit Docs](https://docs.streamlit.io/)

### Inspirational Projects
- [QuantConnect](https://www.quantconnect.com/)
- [Zipline Backtester](https://github.com/quantopian/zipline)

---

## 🚀 Next Steps (After Completion)

1. **Advanced Extensions:**
   - Multi-period optimization (dynamic programming)
   - Reinforcement learning for adaptive rebalancing
   - Factor-based risk models (Barra)

2. **Real-World Deployment:**
   - Connect to brokerage APIs (Alpaca, Interactive Brokers)
   - Real-time data streams (WebSocket)
   - Production monitoring (Prometheus + Grafana)

3. **Research Publications:**
   - Blog post series on Medium
   - Conference submission (IEEE CIS, ICAIF)
   - Kaggle kernel/competition

---

**Last Updated:** October 3, 2025
**Status:** Planning Phase Complete ✅
**Next Phase:** Implementation Start (Data Infrastructure)
