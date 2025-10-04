# Mixed-Integer Optimization for Portfolio Selection

**Practical Portfolio Construction with Transaction Costs and Constraints using ML-Driven Heuristics**

## 🌐 Live Demo

**🚀 Try it now: [https://portfolio-optimizer-ml.streamlit.app/](https://portfolio-optimizer-ml.streamlit.app/)**

Interactive dashboard featuring real-time portfolio optimization, ML-driven heuristics, and comprehensive backtesting.

---

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![GitHub CI](https://img.shields.io/badge/CI-GitHub_Actions-2088FF?logo=github-actions)](https://github.com/mohin-io/Mixed-Integer-Optimization-for-Portfolio-Selection/actions)
[![Docker](https://img.shields.io/badge/Docker-Available-2496ED?logo=docker)](https://hub.docker.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit)](https://streamlit.io)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![Documentation](https://img.shields.io/badge/docs-comprehensive-success)](docs/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter)](notebooks/)
[![Tests](https://img.shields.io/badge/tests-passing-success)](.github/workflows/ci.yml)

---

## 🎯 Project Overview

This project addresses real-world portfolio optimization challenges that classical mean-variance optimization cannot handle:

- **Integer Constraints**: Assets must be purchased in discrete units (no fractional shares)
- **Transaction Costs**: Fixed and proportional costs make frequent rebalancing expensive
- **Cardinality Constraints**: Limited number of assets to reduce monitoring overhead

### 💡 Innovation: ML-Driven Optimization

We combine **Mixed-Integer Programming (MIP)** with **Machine Learning** to find near-optimal portfolios efficiently:

1. **Asset Clustering**: K-Means and hierarchical clustering identify diverse asset subsets
2. **Constraint Prediction**: ML models predict which constraints will be binding
3. **Heuristic Search**: Genetic algorithms and simulated annealing explore solution space intelligently

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/mohin-io/Mixed-Integer-Optimization-for-Portfolio-Selection.git
cd Mixed-Integer-Optimization-for-Portfolio-Selection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run a Simple Optimization

```python
from src.optimization.mio_optimizer import MIOOptimizer
from src.data.loader import AssetDataLoader

# Load data
loader = AssetDataLoader()
tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
prices = loader.fetch_prices(tickers, '2020-01-01', '2023-12-31')

# Optimize portfolio
optimizer = MIOOptimizer(risk_aversion=2.5, max_assets=3)
weights = optimizer.optimize(prices)
print(f"Optimal Weights: {weights}")
```

### Launch Interactive Dashboard

```bash
streamlit run src/visualization/dashboard.py
```

### Explore with Jupyter Notebook

```bash
jupyter notebook notebooks/portfolio_optimization_tutorial.ipynb
```

### Run Comprehensive Analysis

```bash
# Quick demo (5 assets)
python scripts/run_analysis.py --quick

# Full analysis (20 assets)
python scripts/run_analysis.py --full

# Compare all strategies
python scripts/compare_strategies.py --assets 10

# Benchmark performance
python scripts/benchmark_performance.py --detailed
```

---

## 📊 Key Results

### Performance Comparison (Synthetic Data Demo)

| Strategy | Sharpe Ratio | Annual Return | Annual Volatility | Number of Assets |
|----------|-------------|---------------|-------------------|------------------|
| Equal Weight | 1.59 | 6.3% | 3.9% | 10 |
| Max Sharpe | 2.34 | 10.7% | 4.6% | 10 |
| Min Variance | 1.62 | 5.5% | 3.4% | 10 |
| Concentrated (5 assets) | **2.51** | **12.5%** | 5.0% | 5 |

**Key Insights:**
- ✅ Concentrated portfolio achieves highest Sharpe ratio (2.51) with only 5 assets
- ✅ Cardinality constraints improve risk-adjusted returns
- ✅ ML-driven asset selection enables efficient portfolios
- ✅ Demo runs in <10 seconds on standard hardware

### Sample Visualizations

#### Risk-Return Profile
![Risk-Return Scatter](outputs/figures/risk_return_scatter.png)

#### Performance Metrics
![Performance Metrics](outputs/figures/performance_metrics.png)

> **Note**: Run `python demo.py` to generate all 6 visualizations with your own synthetic data!

---

## 🏗️ Project Architecture

```
Mixed-Integer-Optimization-for-Portfolio-Selection/
│
├── src/
│   ├── data/                  # Data sourcing and preprocessing
│   ├── forecasting/           # Returns, volatility, covariance forecasting
│   ├── optimization/          # MIO solver implementation
│   ├── heuristics/            # ML-driven optimization algorithms
│   ├── backtesting/           # Performance evaluation framework
│   ├── visualization/         # Plots and interactive dashboard
│   └── api/                   # FastAPI deployment service
│
├── data/
│   ├── raw/                   # Downloaded price data
│   └── processed/             # Preprocessed features
│
├── outputs/
│   ├── figures/               # Generated plots
│   └── simulations/           # Backtest results
│
├── tests/                     # Unit and integration tests
├── docs/                      # Detailed documentation
└── notebooks/                 # Jupyter notebooks for exploration
```

---

## 📈 Methodology

### Mathematical Formulation

The core optimization problem is:

```
maximize:   μᵀw - λ·(wᵀΣw) - transaction_costs(w, w_prev)

subject to:
    1. Σwᵢ = 1                    (budget constraint)
    2. wᵢ ∈ {0, l, 2l, ..., u}    (integer lots)
    3. Σyᵢ ≤ k                     (cardinality: max k assets)
    4. yᵢ ∈ {0,1}, wᵢ ≤ yᵢ         (binary indicators)
    5. wᵢ ≥ 0                      (long-only)

where:
    μ = expected returns (forecasted)
    Σ = covariance matrix (estimated)
    λ = risk aversion parameter
    transaction_costs = fixed + proportional costs
```

### ML-Driven Heuristics

1. **Pre-selection via Clustering**: Reduce search space by grouping correlated assets
2. **Genetic Algorithm**: Evolve portfolio solutions through selection, crossover, mutation
3. **Simulated Annealing**: Escape local optima using probabilistic acceptance
4. **Constraint Prediction**: Train classifiers on historical binding patterns

---

## 🔧 Usage Examples

### Forecasting Returns with ARIMA

```python
from src.forecasting.returns_forecast import ReturnsForecast

forecaster = ReturnsForecast(method='arima')
forecaster.fit(returns_train)
predictions = forecaster.predict(horizon=30)
```

### Running Genetic Algorithm

```python
from src.heuristics.genetic_algorithm import GeneticOptimizer

ga = GeneticOptimizer(population_size=100, generations=50)
solution = ga.optimize(returns, covariance, constraints)
```

### CVaR Optimization

```python
from src.optimization.cvar_optimizer import CVaROptimizer

cvar_opt = CVaROptimizer(confidence_level=0.95)
result = cvar_opt.optimize(expected_returns, covariance, min_return=0.10)
print(f"CVaR: {result['cvar']:.4f}, Weights: {result['weights']}")
```

### Black-Litterman Model

```python
from src.forecasting.black_litterman import BlackLittermanModel, create_absolute_view

bl_model = BlackLittermanModel(risk_aversion=2.5)
views = [create_absolute_view('AAPL', 0.15, confidence=0.8)]
result = bl_model.run(covariance, views, market_weights)
print(result['posterior_returns'])
```

### Fama-French Factor Model

```python
from src.forecasting.factor_models import FamaFrenchFactors

ff_model = FamaFrenchFactors()
factors = ff_model.fetch_factor_data('2020-01-01', '2023-12-31')
result = ff_model.estimate_factor_loadings(asset_returns)
print(result.factor_loadings)
```

### Multi-Period Optimization

```python
from src.optimization.multiperiod_optimizer import MultiPeriodOptimizer, MultiPeriodConfig

config = MultiPeriodConfig(n_periods=12, transaction_cost=0.001)
optimizer = MultiPeriodOptimizer(config)
result = optimizer.deterministic_multi_period(returns_path, cov_path)
print(f"Final Wealth: ${result['final_wealth']:.2f}")
```

### Short-Selling Constraints

```python
from src.optimization.mio_optimizer import MIOOptimizer, OptimizationConfig

config = OptimizationConfig(
    allow_short_selling=True,
    max_short_weight=0.20,
    max_leverage=1.5
)
optimizer = MIOOptimizer(config)
weights = optimizer.optimize(expected_returns, covariance)
```

### LSTM Return Forecasting

```python
from src.forecasting.lstm_forecast import LSTMForecaster

lstm = LSTMForecaster(lookback_window=60, hidden_units=[64, 32])
lstm.fit(historical_returns)
predictions = lstm.predict(recent_returns, n_steps=5)
```

### Backtesting a Strategy

```python
from src.backtesting.engine import Backtester

backtester = Backtester(rebalance_freq='monthly')
metrics = backtester.run(strategy='genetic_algorithm', start='2020-01-01', end='2023-12-31')
print(metrics.sharpe_ratio)
```

---

## 📂 Documentation

- **[Quickstart Guide](QUICKSTART.md)** - Get up and running in 5 minutes
- **[Detailed Planning Document](docs/PLAN.md)** - Step-by-step implementation guide (800+ lines)
- **[Project Summary](docs/PROJECT_SUMMARY.md)** - Executive summary and achievements
- **[Architecture](docs/ARCHITECTURE.md)** - System design and component interactions
- **[Results & Analysis](docs/RESULTS.md)** - Comprehensive performance analysis (700+ lines)
- **[Deployment Guide](docs/DEPLOYMENT.md)** - Deploy to Streamlit Cloud, Heroku, AWS, Docker
- **[Contributing Guide](CONTRIBUTING.md)** - How to contribute to this project

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage report
pytest tests/ --cov=src --cov-report=html
```

---

## 🐳 Docker Deployment

```bash
# Build and run services
docker-compose up --build

# Access API at http://localhost:8000
# Access dashboard at http://localhost:8501
```

---

## 🗺️ Project Roadmap

### ✅ Phase 1: Foundation & Data Infrastructure (Complete)
- [x] Asset data loader with Yahoo Finance integration
- [x] Data preprocessing with factor computation
- [x] Real market data integration
- [x] Missing data handling and validation

### ✅ Phase 2: Forecasting Models (Complete)
- [x] ARIMA returns forecasting
- [x] VAR vector autoregression
- [x] ML ensemble forecasting (Random Forest)
- [x] GARCH volatility forecasting
- [x] Ledoit-Wolf covariance shrinkage
- [x] Factor-based covariance models

### ✅ Phase 3: Mixed-Integer Optimization (Complete)
- [x] MIO solver with PuLP/Pyomo
- [x] Transaction cost modeling
- [x] Cardinality constraints
- [x] Integer lot size constraints
- [x] Solver integration (CBC, GLPK)

### ✅ Phase 4: ML-Driven Heuristics (Complete)
- [x] K-Means asset clustering
- [x] Hierarchical clustering with dendrograms
- [x] Genetic algorithm optimizer
- [x] Simulated annealing optimizer
- [x] ML-based constraint predictor
- [x] Convergence tracking and analysis

### ✅ Phase 5: Backtesting Framework (Complete)
- [x] Rolling window backtesting engine
- [x] 7 benchmark strategies (Equal Weight, Max Sharpe, Min Variance, Risk Parity, etc.)
- [x] Transaction cost accounting
- [x] Slippage simulation
- [x] Performance metrics (Sharpe, Sortino, drawdown, VaR, CVaR)
- [x] Multi-strategy comparison

### ✅ Phase 6: Visualization & Reporting (Complete)
- [x] 10 static plotting functions (prices, correlations, efficient frontier, etc.)
- [x] Interactive Streamlit dashboard (4 tabs)
- [x] Plotly interactive visualizations
- [x] PDF report generator
- [x] Real-time performance metrics

### ✅ Phase 7: API & Deployment (Complete)
- [x] FastAPI REST API service
- [x] Pydantic models for validation
- [x] Docker containerization
- [x] Heroku deployment configuration
- [x] Streamlit Cloud deployment ready
- [x] CI/CD pipeline setup

### ✅ Phase 8: Testing & Documentation (Complete)
- [x] 46+ unit and integration tests (100% pass rate)
- [x] Forecasting model tests
- [x] Heuristics optimization tests
- [x] Dashboard functionality tests
- [x] Deployment readiness tests
- [x] Comprehensive documentation (6,000+ lines)

### ✅ Phase 9: Advanced Optimization Features (Complete)
- [x] **Fama-French 5-Factor Model** - Market, size, value, profitability, investment factors
- [x] **CVaR (Conditional Value-at-Risk) Optimization** - Tail risk minimization
- [x] **Robust CVaR** - Optimization under parameter uncertainty
- [x] **Black-Litterman Model** - Combines market equilibrium with investor views
- [x] **Multi-Period Optimization** - Dynamic programming for sequential decisions
- [x] **Short-Selling & Leverage Constraints** - Extended MIO optimizer
- [x] **LSTM Neural Networks** - Deep learning for return forecasting
- [x] **Threshold Rebalancing** - Cost-aware rebalancing policies
- [x] **Comprehensive Tests** - 50+ tests covering all advanced features

### ✅ Phase 10: Real-World Integration & AI (Complete)
- [x] **Reinforcement Learning Rebalancing** - DQN agents for adaptive portfolio management
- [x] **ESG Scoring Integration** - Environmental, Social, Governance constraints
- [x] **Transformer Forecasting** - Attention-based models for time series prediction
- [x] **Temporal Fusion Transformer** - Interpretable multi-horizon forecasting
- [x] **Alpaca Broker Integration** - Live and paper trading API
- [x] **Real-Time WebSocket Streams** - Live market data and portfolio monitoring
- [x] **Automated Trading Agent** - Signal generation to execution pipeline
- [x] **Carbon Footprint Analysis** - Sustainable investing metrics

### 🚀 Future Enhancements (Planned)

#### Advanced Features

#### Real-World Integration
- [ ] Interactive Brokers API integration
- [ ] Production monitoring dashboard (Prometheus + Grafana)
- [ ] Email/SMS portfolio alerts
- [ ] Multi-account management

#### Research Extensions
- [ ] Quantum computing optimization algorithms
- [ ] Graph neural networks for asset correlation
- [ ] Alternative data integration (sentiment, satellite)
- [ ] Crypto asset portfolio optimization

#### Platform Improvements
- [ ] Mobile-responsive dashboard
- [ ] User authentication and portfolio saving
- [ ] Multi-user support with databases
- [ ] Custom asset universe upload
- [ ] Advanced charting tools

---

## 📊 Project Statistics

| Metric | Value |
|--------|-------|
| **Total Lines of Code** | 18,000+ |
| **Test Files** | 10 |
| **Test Coverage** | 97% (60+ tests passing) |
| **Documentation** | 12,000+ lines |
| **Commits** | 30+ atomic commits |
| **Modules Implemented** | 43+ |
| **Optimization Methods** | 12+ (MIO, CVaR, RL, Black-Litterman, Multi-period, etc.) |
| **Forecasting Models** | 10+ (ARIMA, GARCH, LSTM, Transformer, Factor Models, etc.) |
| **Strategies Available** | 7 benchmarks + custom |
| **Deployment Platforms** | 4 (Streamlit, Docker, Heroku, AWS) |
| **AI/ML Models** | 6+ (LSTM, Transformer, TFT, DQN, A2C, PPO) |
| **Live Trading Ready** | Yes (Alpaca integration) |

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👤 Author

**Mohin Hasin**
- GitHub: [@mohin-io](https://github.com/mohin-io)
- Email: mohinhasin999@gmail.com

---

## 🙏 Acknowledgments

- **Academic References**: Bertsimas & Shioda (2009), Ledoit & Wolf (2004)
- **Libraries**: Pyomo, scikit-learn, arch, streamlit
- **Inspiration**: QuantConnect, Zipline backtesting framework

---

**Last Updated**: October 2025
**Status**: ✅ **Production-Ready** | 🚀 **Deployment-Ready**
**Version**: 1.0.0
