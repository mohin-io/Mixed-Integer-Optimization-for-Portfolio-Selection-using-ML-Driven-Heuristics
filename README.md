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

---

## 📊 Key Results

> **Note**: Results and visuals will be added as the project progresses.

### Performance Comparison

| Strategy | Sharpe Ratio | Annual Return | Max Drawdown | Turnover | Avg Runtime |
|----------|-------------|---------------|--------------|----------|-------------|
| Naïve MVO | TBD | TBD | TBD | TBD | TBD |
| Exact MIO | TBD | TBD | TBD | TBD | TBD |
| Genetic Algorithm | TBD | TBD | TBD | TBD | TBD |
| Simulated Annealing | TBD | TBD | TBD | TBD | TBD |

### Sample Visualizations

*Visualizations will be embedded here once generated*

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

- [Detailed Planning Document](docs/PLAN.md) - Step-by-step implementation guide
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
