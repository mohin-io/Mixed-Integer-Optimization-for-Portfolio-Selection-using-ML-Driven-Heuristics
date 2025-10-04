# Quick Start Guide

## Installation

### Prerequisites
- Python 3.10 or higher
- Git

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/mohin-io/Mixed-Integer-Optimization-for-Portfolio-Selection.git
cd Mixed-Integer-Optimization-for-Portfolio-Selection-using-ML-Driven-Heuristics
```

2. **Create virtual environment**
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python -m venv venv
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## Running the Project

### Run the Complete Pipeline

```bash
python main.py
```

This will:
- Download historical price data for 20+ assets
- Perform data preprocessing and factor analysis
- Generate forecasts for returns and volatility
- Cluster assets and select diverse representatives
- Optimize portfolios using 4 different strategies:
  1. Equal Weight
  2. Naive Mean-Variance
  3. Mixed-Integer Optimization (MIO)
  4. Genetic Algorithm
- Compare performance metrics
- Generate visualizations in `outputs/figures/`
- Save results to `outputs/simulations/`

### Expected Output

```
================================================================================
MIXED-INTEGER PORTFOLIO OPTIMIZATION WITH ML-DRIVEN HEURISTICS
================================================================================

[STEP 1] Loading Data...
✓ Loaded 21 assets
✓ Date range: 2020-01-01 to 2023-12-31

[STEP 2] Data Preprocessing...
✓ Computed factor exposures
✓ Correlation matrix shape: (21, 21)

[STEP 3] Forecasting...
✓ Expected returns forecasted
✓ Volatility forecasted
✓ Covariance matrix estimated

[STEP 4] Asset Clustering...
✓ Selected 10 diverse assets

[STEP 5] Portfolio Optimization...
✓ Equal Weight: 10.00% per asset
✓ Naive MVO: Optimized
✓ MIO: Solved in 2.3 seconds
✓ Genetic Algorithm: Converged in 12.5 seconds

[STEP 6] Performance Comparison...
                     expected_return  volatility  sharpe_ratio  n_assets
Equal Weight                  0.156       0.245         0.637        10
Naive MVO                     0.189       0.223         0.848         8
MIO                           0.192       0.218         0.881         5
Genetic Algorithm             0.187       0.221         0.846         5

[STEP 7] Generating Visualizations...
✓ Saved: asset_prices.png
✓ Saved: correlation_matrix.png
✓ Saved: portfolio_weights.png
✓ Saved: performance_metrics.png
✓ Saved: risk_return_scatter.png

PIPELINE COMPLETED SUCCESSFULLY!
```

## Running with Docker

```bash
# Build and run
docker-compose up --build

# Results will be saved to outputs/ directory
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

## Exploring Individual Modules

### Data Loading
```python
from src.data.loader import AssetDataLoader

loader = AssetDataLoader()
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
prices = loader.fetch_prices(tickers, '2020-01-01', '2023-12-31')
returns = loader.compute_returns(prices)
```

### Portfolio Optimization
```python
from src.optimization.mio_optimizer import MIOOptimizer, OptimizationConfig

config = OptimizationConfig(
    risk_aversion=2.5,
    max_assets=5,
    min_weight=0.10,
    max_weight=0.40
)

optimizer = MIOOptimizer(config=config)
weights = optimizer.optimize(expected_returns, cov_matrix)
```

### Genetic Algorithm
```python
from src.heuristics.genetic_algorithm import GeneticOptimizer, GAConfig

config = GAConfig(population_size=100, generations=50, max_assets=5)
ga = GeneticOptimizer(config=config)
weights = ga.optimize(expected_returns, cov_matrix)
```

## Customization

### Change Asset Universe
Edit `main.py` line 51-62 to modify the list of tickers:

```python
tickers = [
    'YOUR', 'CUSTOM', 'TICKERS', 'HERE'
]
```

### Adjust Optimization Parameters
Edit `main.py` line 172-178:

```python
mio_config = OptimizationConfig(
    risk_aversion=3.0,        # Higher = more conservative
    max_assets=3,             # Fewer assets = more concentrated
    min_weight=0.15,          # Higher min weight
    max_weight=0.50,          # Allow larger positions
)
```

### Change Date Range
Edit `main.py` line 64-65:

```python
start_date = '2018-01-01'
end_date = '2024-12-31'
```

## Output Structure

```
outputs/
├── figures/                   # Visualizations
│   ├── asset_prices.png
│   ├── correlation_matrix.png
│   ├── portfolio_weights.png
│   ├── performance_metrics.png
│   └── risk_return_scatter.png
│
└── simulations/               # Numerical results
    └── strategy_comparison.csv
```

## Troubleshooting

### Data Download Issues
- Ensure you have internet connection
- Yahoo Finance may rate-limit requests. Wait a few minutes and retry.
- Use cached data: Set `use_cache=True` in `fetch_prices()`

### Optimization Solver Errors
- MIO requires a solver (CBC is included with PuLP)
- For better performance, install commercial solvers (CPLEX/Gurobi)
- Fallback: Genetic Algorithm doesn't require external solvers

### Memory Issues
- Reduce number of assets
- Decrease genetic algorithm population size
- Use smaller date ranges

## Next Steps

1. **Extend to Backtesting**: Implement rolling window backtesting (see [docs/PLAN.md](docs/PLAN.md))
2. **Add Simulated Annealing**: Implement SA heuristic optimizer
3. **Build Dashboard**: Create interactive Streamlit dashboard
4. **Deploy API**: Set up FastAPI service for portfolio optimization

## Support

- Documentation: [docs/PLAN.md](docs/PLAN.md)
- Issues: https://github.com/mohin-io/Mixed-Integer-Optimization-for-Portfolio-Selection/issues
- Email: mohinhasin999@gmail.com
