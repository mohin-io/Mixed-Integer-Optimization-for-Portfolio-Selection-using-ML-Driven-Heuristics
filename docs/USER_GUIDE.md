# User Guide - Portfolio Optimization Dashboard

**Version:** 1.0.0
**Last Updated:** 2025-10-04

---

## 📚 Table of Contents

1. [Getting Started](#getting-started)
2. [Dashboard Overview](#dashboard-overview)
3. [Configuration Options](#configuration-options)
4. [Optimization Strategies](#optimization-strategies)
5. [Understanding the Results](#understanding-the-results)
6. [Visualizations](#visualizations)
7. [Tips & Best Practices](#tips--best-practices)
8. [Troubleshooting](#troubleshooting)
9. [FAQ](#faq)

---

## 🚀 Getting Started

### Accessing the Dashboard

**Online (Streamlit Cloud):**
```
Visit: https://share.streamlit.io/mohin-io/mixed-integer-optimization-for-portfolio-selection
```

**Local Installation:**
```bash
# Clone repository
git clone https://github.com/mohin-io/Mixed-Integer-Optimization-for-Portfolio-Selection.git
cd Mixed-Integer-Optimization-for-Portfolio-Selection

# Install dependencies
pip install -r requirements.txt

# Run dashboard
streamlit run src/visualization/dashboard.py
```

The dashboard will open in your browser at `http://localhost:8501`

---

## 📊 Dashboard Overview

The Portfolio Optimization Dashboard is an interactive tool for exploring different portfolio construction strategies using Mixed-Integer Optimization and ML-driven heuristics.

### Main Features

- **4 Optimization Strategies:** Equal Weight, Max Sharpe, Min Variance, Concentrated
- **Interactive Configuration:** Adjust parameters in real-time
- **Multiple Visualizations:** Weights, prices, correlations, performance
- **Instant Results:** Fast optimization with immediate feedback
- **Export Ready:** Results can be used for further analysis

---

## ⚙️ Configuration Options

### Sidebar Controls

Access all configuration options in the left sidebar:

#### 1. Data Parameters

**Number of Assets** (5-20)
- Controls how many assets to include in the portfolio
- More assets = more diversification but slower optimization
- **Recommended:** 10 assets for balanced performance

**Number of Days** (250-2000)
- Historical data length for analysis
- More days = more stable statistics but less responsive to recent trends
- **Recommended:** 500-1000 days for reliable estimates

**Random Seed** (1-1000)
- Controls data generation randomness
- Same seed = same data (reproducible results)
- Different seed = different market conditions
- **Tip:** Try multiple seeds to test strategy robustness

#### 2. Strategy Selection

Choose from 4 optimization strategies:

1. **Equal Weight:** Naive 1/N allocation
2. **Max Sharpe:** Maximize risk-adjusted returns
3. **Min Variance:** Minimize portfolio volatility
4. **Concentrated:** Focus on top performers (requires Max Assets parameter)

#### 3. Strategy-Specific Parameters

**Max Assets** (for Concentrated strategy only)
- Limits number of assets in the portfolio
- Range: 3 to number of total assets
- **Recommended:** 5-7 for concentrated portfolios

---

## 🎯 Optimization Strategies

### 1. Equal Weight Strategy

**What it does:**
- Allocates equal weight (1/N) to each asset
- Simplest possible strategy
- No optimization required

**When to use:**
- As a benchmark for comparison
- When you want maximum diversification
- When you don't trust optimization models

**Pros:**
- ✅ Simple and transparent
- ✅ Low turnover
- ✅ Robust to estimation errors

**Cons:**
- ❌ Ignores asset characteristics
- ❌ May include poor performers
- ❌ Not risk-optimized

**Example Results:**
```
Expected Return: 6.64%
Volatility: 3.92%
Sharpe Ratio: 1.695
Active Assets: 10
```

---

### 2. Max Sharpe Strategy

**What it does:**
- Maximizes the Sharpe ratio (return per unit of risk)
- Balances return and volatility optimally
- Uses random search optimization (10,000 iterations)

**When to use:**
- When you want the best risk-adjusted returns
- For long-term investment portfolios
- When both return and risk matter

**Pros:**
- ✅ Optimal risk-adjusted performance
- ✅ Considers both return and volatility
- ✅ Theory-backed approach

**Cons:**
- ❌ Sensitive to return estimates
- ❌ May concentrate in few assets
- ❌ Can have high turnover

**Example Results:**
```
Expected Return: 15.20%
Volatility: 4.37%
Sharpe Ratio: 3.482
Active Assets: 10
```

---

### 3. Min Variance Strategy

**What it does:**
- Minimizes portfolio volatility
- Focuses purely on risk reduction
- Ignores expected returns

**When to use:**
- When risk minimization is the primary goal
- For conservative investors
- In uncertain market conditions
- For defensive portfolios

**Pros:**
- ✅ Lowest possible volatility
- ✅ More stable estimates (uses covariance only)
- ✅ Good for risk-averse investors

**Cons:**
- ❌ May sacrifice returns
- ❌ Can be too conservative
- ❌ Ignores return potential

**Example Results:**
```
Expected Return: 7.44%
Volatility: 3.28%
Sharpe Ratio: 2.266
Active Assets: 10
```

---

### 4. Concentrated Strategy

**What it does:**
- Selects top N assets by Sharpe ratio
- Optimizes weights among selected assets
- Implements cardinality constraint

**When to use:**
- When you want focused exposure
- To reduce transaction costs
- For high-conviction portfolios
- When managing many positions is difficult

**Pros:**
- ✅ Focused on best performers
- ✅ Lower transaction costs
- ✅ Easier to manage
- ✅ Often higher returns

**Cons:**
- ❌ Higher concentration risk
- ❌ Less diversification
- ❌ More volatile

**Example Results:**
```
Expected Return: 18.02%
Volatility: 4.89%
Sharpe Ratio: 3.682
Active Assets: 5
```

---

## 📈 Understanding the Results

### Portfolio Metrics

After optimization, you'll see four key metrics:

#### Expected Annual Return
- Projected yearly return based on historical data
- **Higher is better** (but consider volatility)
- Annualized (multiplied by 252 trading days)
- Example: 15.20% means $100 → $115.20 in one year (expected)

#### Annual Volatility
- Standard deviation of returns (risk measure)
- **Lower is better** for risk-averse investors
- Annualized using √252 scaling
- Example: 4.37% means typical yearly fluctuation

#### Sharpe Ratio
- Return per unit of risk (Return / Volatility)
- **Higher is better**
- Measures risk-adjusted performance
- Rule of thumb:
  - < 1.0: Poor
  - 1.0-2.0: Good
  - 2.0-3.0: Very Good
  - > 3.0: Excellent

#### Number of Assets
- Count of assets with meaningful allocation (>0.01%)
- Shows portfolio concentration
- Fewer assets = more concentrated
- More assets = more diversified

---

## 📊 Visualizations

The dashboard provides 4 interactive visualization tabs:

### Tab 1: Portfolio Weights

**Bar Chart**
- Shows allocation to each asset
- Color-coded for easy reading
- Sorted by weight
- Percentages labeled on bars

**Data Table**
- Exact weight percentages
- Only shows active positions (>0.01%)
- Sorted by weight (descending)

**What to look for:**
- Concentration: Are weights balanced or focused?
- Diversification: How many assets have significant weights?
- Outliers: Any extremely large positions?

---

### Tab 2: Asset Prices

**Line Chart**
- Historical price evolution
- Shows top 5 assets
- Time series view

**What to look for:**
- Trends: Upward, downward, or sideways
- Volatility: How much do prices fluctuate?
- Correlation: Do assets move together?

---

### Tab 3: Correlation Matrix

**Heatmap**
- Shows pairwise correlations
- Color scale: Red (positive) to Blue (negative)
- Values from -1 to +1

**How to read:**
- **+1.0:** Perfect positive correlation (move together)
- **0.0:** No correlation (independent)
- **-1.0:** Perfect negative correlation (move opposite)

**What to look for:**
- Diversification: Lower correlations = better diversification
- Risk concentrations: High correlations = similar risk exposures
- Hedging opportunities: Negative correlations

---

### Tab 4: Portfolio Performance

**Cumulative Return Chart**
- Shows portfolio growth over time
- Compares to Equal Weight benchmark
- Starts at 1.0 (normalized)

**What to look for:**
- Outperformance: Is strategy above benchmark?
- Stability: Is growth smooth or erratic?
- Drawdowns: How much does it fall during bad periods?

---

## 💡 Tips & Best Practices

### For Beginners

1. **Start with Equal Weight**
   - Understand the benchmark first
   - Compare other strategies to this baseline

2. **Use Default Parameters**
   - 10 assets, 1000 days, seed 42
   - Good starting point for exploration

3. **Try All Strategies**
   - See how different approaches compare
   - Understand trade-offs between strategies

### For Advanced Users

1. **Test Multiple Seeds**
   - Run same strategy with different seeds
   - Check if results are stable
   - Robust strategies work across conditions

2. **Compare Sharpe Ratios**
   - Best overall performance indicator
   - Balances return and risk
   - Use for strategy selection

3. **Consider Your Goals**
   - Risk-averse → Min Variance
   - Return-focused → Max Sharpe
   - Simple → Equal Weight
   - Concentrated → Concentrated

4. **Monitor Concentration**
   - Too few assets = high risk
   - Too many assets = diversification benefits diminish
   - Sweet spot: 5-15 assets

### Parameter Selection Guide

| Goal | Assets | Days | Strategy |
|------|--------|------|----------|
| Maximum Diversification | 15-20 | 1000 | Equal Weight |
| Best Risk-Adjusted Return | 10-15 | 500-1000 | Max Sharpe |
| Minimum Risk | 10-15 | 1000 | Min Variance |
| High Conviction | 5-10 | 500 | Concentrated |

---

## 🔧 Troubleshooting

### Issue: Optimization taking too long

**Solution:**
- Reduce number of assets (try 10 or fewer)
- Use Equal Weight for instant results
- Refresh the page and try again

### Issue: Strange or unexpected results

**Solution:**
- Try a different random seed
- Check if you have realistic parameters
- Increase number of days for more stable estimates
- Use Equal Weight to verify data generation is working

### Issue: Concentrated strategy not working

**Solution:**
- Ensure Max Assets parameter is set
- Make sure Max Assets < Total Assets
- Try reducing Max Assets value

### Issue: All weights are equal

**Solution:**
- You may have selected Equal Weight strategy
- Try Max Sharpe or Min Variance instead
- Check that optimization completed successfully

### Issue: Dashboard not loading

**Solution:**
- Check internet connection
- Clear browser cache
- Try a different browser
- For local installation: verify all dependencies installed

---

## ❓ FAQ

### Q: What data does the dashboard use?

**A:** The dashboard currently uses **synthetically generated data** with realistic statistical properties. This ensures:
- Reproducibility with random seeds
- No data access issues
- Fast performance
- Educational focus

Future versions may include real market data integration.

---

### Q: How accurate are the optimization results?

**A:** The optimization algorithms are mathematically sound and use industry-standard approaches:
- Equal Weight: Exact
- Max Sharpe: Approximate (random search with 10,000 iterations)
- Min Variance: Approximate (random search with 10,000 iterations)
- Concentrated: Approximate (random search with 10,000 iterations)

Results are reliable for educational and comparative purposes.

---

### Q: Can I use this for real money?

**A:** This dashboard is designed for **educational and research purposes**. Before using any strategy with real money:
- ✅ Understand the strategy thoroughly
- ✅ Test with real historical data
- ✅ Consider transaction costs
- ✅ Account for taxes
- ✅ Consult a financial advisor
- ✅ Start with small amounts

**Remember:** Past performance does not guarantee future results.

---

### Q: What's the difference between Max Sharpe and Min Variance?

**A:**
- **Max Sharpe:** Maximizes return per unit of risk (balances both)
- **Min Variance:** Only minimizes risk (ignores returns)

Max Sharpe typically has:
- ✅ Higher returns
- ❌ Higher volatility
- ✅ Better Sharpe ratio

Min Variance typically has:
- ❌ Lower returns
- ✅ Lower volatility
- ❌ Lower Sharpe ratio

---

### Q: How often should I rebalance?

**A:** The dashboard shows static allocations. In practice:
- **Conservative:** Quarterly (4x per year)
- **Moderate:** Monthly (12x per year)
- **Aggressive:** Weekly or daily

More frequent rebalancing:
- ✅ Stays closer to target weights
- ❌ Higher transaction costs
- ❌ More tax events

---

### Q: Can I export the results?

**A:** Currently, results are displayed in the dashboard. To export:
1. Take screenshots of visualizations
2. Copy weights from the data table
3. Note metrics from the summary

Future versions may include CSV/Excel export functionality.

---

### Q: What's a good Sharpe ratio?

**A:** General guidelines:
- **< 0:** Strategy loses money
- **0-1:** Poor risk-adjusted performance
- **1-2:** Good performance
- **2-3:** Very good performance
- **> 3:** Excellent performance (rare!)

Note: These are annualized Sharpe ratios. Adjust expectations for different time horizons.

---

### Q: Why do results change with different seeds?

**A:** Different seeds generate different synthetic market conditions:
- Different asset returns
- Different correlations
- Different volatilities

This helps you:
- ✅ Test strategy robustness
- ✅ Understand sensitivity to market conditions
- ✅ Avoid overfitting to one scenario

---

### Q: What does "Active Assets" mean?

**A:** Number of assets with allocation > 0.01% (1 basis point).

Ignores tiny allocations that are:
- Too small to trade practically
- Result of numerical precision
- Not meaningful for portfolio construction

---

### Q: How can I learn more about portfolio optimization?

**A:** Resources in this repository:
- [README.md](../README.md) - Project overview
- [docs/ARCHITECTURE.md](ARCHITECTURE.md) - System design
- [docs/RESULTS.md](RESULTS.md) - Detailed performance analysis
- [docs/PLAN.md](PLAN.md) - Implementation guide

External resources:
- Markowitz Portfolio Theory
- Modern Portfolio Theory (MPT)
- Factor models and risk decomposition
- Convex optimization

---

## 📞 Support

### Getting Help

- **Issues:** [GitHub Issues](https://github.com/mohin-io/Mixed-Integer-Optimization-for-Portfolio-Selection/issues)
- **Documentation:** See `docs/` folder
- **Email:** mohinhasin999@gmail.com

### Contributing

Contributions welcome! See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

---

## 📄 License

MIT License - See [LICENSE](../LICENSE) file for details.

---

**Last Updated:** 2025-10-04
**Version:** 1.0.0
**Maintained by:** Mohin Hasin (mohin-io)
