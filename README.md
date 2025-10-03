# Mixed-Integer Optimization for Portfolio Selection

**Practical Portfolio Construction with Transaction Costs and Constraints using ML-Driven Heuristics**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

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
- Methodology (TBD) - Mathematical formulations and algorithms
- Results (TBD) - Simulation outcomes and analysis
- API Reference (TBD) - Endpoint documentation

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
**Status**: 🚧 Under Active Development
