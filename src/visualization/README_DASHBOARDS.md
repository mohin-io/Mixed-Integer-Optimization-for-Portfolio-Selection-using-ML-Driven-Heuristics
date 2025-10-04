# Interactive Dashboards Suite 📊

Complete suite of interactive Streamlit dashboards for portfolio optimization analysis.

---

## 🎯 Available Dashboards

### 1. **Main Dashboard V2** (`dashboard_v2.py`)
**Complete portfolio optimization with dual data modes**

**Features:**
- ✅ Synthetic & Real Market Data
- ✅ 5 Optimization Strategies
- ✅ Interactive Parameter Controls
- ✅ 4 Visualization Tabs
- ✅ Performance Metrics

**Run:**
```bash
streamlit run src/visualization/dashboard_v2.py
```

**Best For:** General portfolio optimization and exploration

---

### 2. **Strategy Comparison Dashboard** (`interactive_comparison.py`)
**Compare all strategies side-by-side**

**Features:**
- ✅ Compare 5 Strategies Simultaneously
- ✅ Risk-Return Scatter Plots
- ✅ Weight Distribution Comparison
- ✅ Cumulative Performance Tracking
- ✅ Multi-Dimensional Radar Charts
- ✅ Detailed Individual Analysis

**Interactive Visualizations:**
- Risk-return positioning
- Stacked weight charts
- Performance comparison
- 5-dimensional radar
- Individual deep-dives

**Run:**
```bash
streamlit run src/visualization/interactive_comparison.py
```

**Best For:** Choosing the best strategy for your needs

---

### 3. **Backtesting Dashboard** (`interactive_backtest.py`)
**Test strategies with historical simulation**

**Features:**
- ✅ Rolling Window Backtesting
- ✅ Configurable Rebalancing
- ✅ Transaction Cost Modeling
- ✅ Performance Attribution
- ✅ Drawdown Analysis
- ✅ Monthly Returns Heatmap

**Analysis Tools:**
- Equity curves
- Underwater plots
- Returns distribution
- Rolling metrics (Sharpe, volatility)
- Trading statistics

**Configurable Parameters:**
- Rebalancing frequency (1-63 days)
- Transaction costs (0-1%)
- Multiple strategies
- Lookback periods

**Run:**
```bash
streamlit run src/visualization/interactive_backtest.py
```

**Best For:** Validating strategy performance over time

---

### 4. **Risk Analytics Dashboard** (`interactive_risk.py`)
**Comprehensive risk analysis and stress testing**

**Features:**
- ✅ Value at Risk (VaR) Analysis
- ✅ Conditional VaR (CVaR)
- ✅ Monte Carlo Simulation
- ✅ Risk Decomposition
- ✅ Stress Testing
- ✅ Advanced Risk Metrics

**Risk Metrics:**
- **VaR**: Historical & Parametric
- **CVaR**: Expected Shortfall
- **Downside Deviation**
- **Sortino Ratio**
- **Calmar Ratio**
- **Skewness & Kurtosis**

**Analysis Methods:**
- Monte Carlo (up to 5000 simulations)
- Stress scenarios
- Risk contribution analysis
- Q-Q plots for normality
- Rolling VaR

**Run:**
```bash
streamlit run src/visualization/interactive_risk.py
```

**Best For:** Understanding and managing portfolio risk

---

## 📋 Quick Comparison

| Dashboard | Primary Use | Complexity | Best For |
|-----------|-------------|------------|----------|
| **Dashboard V2** | Portfolio Optimization | Beginner | Getting started |
| **Strategy Comparison** | Compare Strategies | Intermediate | Strategy selection |
| **Backtesting** | Historical Testing | Advanced | Validation |
| **Risk Analytics** | Risk Management | Advanced | Risk assessment |

---

## 🚀 Quick Start Guide

### 1. Install Dependencies

```bash
pip install streamlit plotly pandas numpy scipy
```

### 2. Run Any Dashboard

```bash
# Main optimization dashboard
streamlit run src/visualization/dashboard_v2.py

# Or comparison dashboard
streamlit run src/visualization/interactive_comparison.py

# Or backtesting
streamlit run src/visualization/interactive_backtest.py

# Or risk analytics
streamlit run src/visualization/interactive_risk.py
```

### 3. Configure & Explore

All dashboards have:
- **Sidebar Controls** - Adjust parameters
- **Interactive Charts** - Hover, zoom, pan
- **Multiple Tabs** - Different views
- **Real-time Updates** - Instant results

---

## 📊 Dashboard Features Matrix

| Feature | Dashboard V2 | Comparison | Backtest | Risk |
|---------|-------------|------------|----------|------|
| Real Market Data | ✅ | ❌ | ❌ | ❌ |
| Multiple Strategies | ✅ | ✅ | ✅ | ✅ |
| Interactive Plots | ✅ | ✅ | ✅ | ✅ |
| Risk Metrics | Basic | Basic | Advanced | Expert |
| Historical Testing | ❌ | ❌ | ✅ | ❌ |
| Monte Carlo | ❌ | ❌ | ❌ | ✅ |
| Stress Testing | ❌ | ❌ | ❌ | ✅ |
| Transaction Costs | ❌ | ❌ | ✅ | ❌ |

---

## 🎨 Visualization Types

### Dashboard V2
- Bar charts (weights)
- Line charts (prices)
- Heatmaps (correlation)
- Performance curves

### Strategy Comparison
- Scatter plots (risk-return)
- Stacked bars (weights)
- Multi-line charts (performance)
- Radar charts (multi-dimensional)

### Backtesting
- Equity curves
- Underwater plots (drawdowns)
- Histograms (returns distribution)
- Rolling metrics
- Monthly heatmaps

### Risk Analytics
- VaR charts
- Monte Carlo fan charts
- Risk decomposition bars
- Stress test comparisons
- Q-Q plots

---

## 💡 Usage Tips

### For Beginners
1. Start with **Dashboard V2**
2. Try different strategies
3. Compare with Equal Weight
4. Look at Sharpe ratios

### For Intermediate Users
1. Use **Strategy Comparison**
2. Analyze risk-return profiles
3. Check weight distributions
4. Compare performance metrics

### For Advanced Users
1. Run **Backtesting** simulations
2. Test different rebalancing frequencies
3. Account for transaction costs
4. Analyze monthly returns

### For Risk Managers
1. Use **Risk Analytics**
2. Calculate VaR and CVaR
3. Run Monte Carlo simulations
4. Perform stress tests
5. Check diversification benefits

---

## 🔧 Customization

### Modify Parameters

All dashboards support:
```python
# In sidebar configuration
n_assets = st.sidebar.slider("Number of Assets", 5, 20, 10)
n_days = st.sidebar.slider("Number of Days", 250, 2000, 500)
seed = st.sidebar.number_input("Random Seed", 1, 1000, 42)
```

### Add Custom Strategies

Edit the strategy list in any dashboard:
```python
strategies = ['Equal Weight', 'Max Sharpe', 'Min Variance', 'Your Strategy']
```

### Customize Visualizations

Modify Plotly figures:
```python
fig.update_layout(
    title='Your Custom Title',
    height=600,  # Change height
    template='plotly_dark'  # Change theme
)
```

---

## 📚 Documentation

Each dashboard includes:
- **Sidebar Help** - Parameter descriptions
- **Info Boxes** - Metric explanations
- **Welcome Screen** - Feature overview
- **Tips** - Interpretation guidance

---

## 🐛 Troubleshooting

### Dashboard Won't Load
```bash
# Reinstall streamlit
pip install --upgrade streamlit

# Clear cache
streamlit cache clear
```

### Slow Performance
- Reduce number of Monte Carlo simulations
- Use fewer assets
- Shorter time periods
- Lower rebalancing frequency

### Plotly Charts Not Showing
```bash
pip install --upgrade plotly
```

---

## 🚀 Deployment

### Streamlit Cloud

1. **Push to GitHub**
2. **Visit** share.streamlit.io
3. **Select** your dashboard file
4. **Deploy!**

### Local Network

```bash
streamlit run dashboard.py --server.address 0.0.0.0 --server.port 8501
```

Access from other devices: `http://YOUR_IP:8501`

---

## 📈 Performance Optimization

### For Faster Loading

```python
# Use caching
@st.cache_data
def expensive_computation(data):
    # Your code here
    return result

# Reduce iterations
n_monte_carlo = 1000  # Instead of 5000

# Sample data
data = data.sample(frac=0.5)  # Use 50% of data
```

### For Better UX

- Add loading spinners
- Show progress bars
- Provide clear error messages
- Include help text

---

## 🔗 Integration

### Combine with Real Data

```python
from src.data.real_data_loader import RealDataLoader

loader = RealDataLoader()
prices, returns = loader.fetch_data(['AAPL', 'MSFT'], period='1y')
```

### Export Results

```python
# Save to CSV
results_df.to_csv('portfolio_results.csv')

# Save plots
fig.write_html('plot.html')
fig.write_image('plot.png')
```

---

## 🎯 Roadmap

### Planned Features

- [ ] Real-time data updates
- [ ] Multi-currency support
- [ ] Factor model integration
- [ ] Machine learning predictions
- [ ] Social sharing
- [ ] PDF report generation

---

## 📞 Support

**Issues?** Open a GitHub issue
**Questions?** Check the documentation
**Enhancements?** Submit a pull request

---

## 📄 License

MIT License - See main README

---

**Last Updated:** 2025-10-04
**Version:** 2.0.0
**Dashboards:** 4 interactive tools
**Total Features:** 50+ analysis capabilities

🎉 **Happy Analyzing!**
