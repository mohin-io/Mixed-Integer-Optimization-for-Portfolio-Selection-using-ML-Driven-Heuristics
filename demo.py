"""
Demo Script - Portfolio Optimization with Synthetic Data

Demonstrates the complete workflow without requiring external data downloads.
Perfect for quick testing and showcasing the system.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time

# Set random seed for reproducibility
np.random.seed(42)

# Create output directories
OUTPUT_DIR = Path('outputs')
FIGURES_DIR = OUTPUT_DIR / 'figures'
SIMULATIONS_DIR = OUTPUT_DIR / 'simulations'

for dir_path in [OUTPUT_DIR, FIGURES_DIR, SIMULATIONS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

sns.set_style('whitegrid')

print("=" * 80)
print("PORTFOLIO OPTIMIZATION DEMO - SYNTHETIC DATA")
print("=" * 80)
print()

# ============================================================================
# STEP 1: Generate Synthetic Data
# ============================================================================
print("[STEP 1] Generating Synthetic Market Data...")
print("-" * 80)

# Parameters
n_assets = 10
n_days = 1000
tickers = [f'ASSET_{i+1}' for i in range(n_assets)]

# Generate synthetic returns with realistic correlations
# Create a factor model structure
n_factors = 3
factor_loadings = np.random.randn(n_assets, n_factors) * 0.3
factor_returns = np.random.randn(n_days, n_factors) * 0.01

# Asset-specific returns
idiosyncratic_returns = np.random.randn(n_days, n_assets) * 0.005

# Combined returns
returns_matrix = factor_returns @ factor_loadings.T + idiosyncratic_returns

# Add drift (positive expected returns)
drift = np.random.uniform(0.0001, 0.0005, n_assets)
returns_matrix += drift

# Create DataFrame
dates = pd.date_range('2020-01-01', periods=n_days, freq='D')
returns = pd.DataFrame(returns_matrix, index=dates, columns=tickers)

# Generate prices from returns
prices = (1 + returns).cumprod() * 100

print(f"[OK] Generated {n_assets} synthetic assets")
print(f"[OK] Date range: {dates[0].date()} to {dates[-1].date()}")
print(f"[OK] Total observations: {n_days}")
print()

# ============================================================================
# STEP 2: Compute Statistics
# ============================================================================
print("[STEP 2] Computing Statistics...")
print("-" * 80)

# Annualized returns and volatility
annual_returns = returns.mean() * 252
annual_volatility = returns.std() * np.sqrt(252)
sharpe_ratios = annual_returns / annual_volatility

# Covariance matrix
cov_matrix = returns.cov() * 252

# Correlation matrix
corr_matrix = returns.corr()

print(f"[OK] Annual returns: Mean = {annual_returns.mean():.2%}")
print(f"[OK] Annual volatility: Mean = {annual_volatility.mean():.2%}")
print(f"[OK] Sharpe ratios: Mean = {sharpe_ratios.mean():.3f}")
print()

# ============================================================================
# STEP 3: Portfolio Optimization (Simplified)
# ============================================================================
print("[STEP 3] Portfolio Optimization...")
print("-" * 80)

results = {}

# 3.1: Equal Weight
print("\n3.1 Equal Weight Portfolio")
equal_weights = pd.Series(1.0 / n_assets, index=tickers)
results['Equal Weight'] = equal_weights
print(f"[OK] Each asset: {1.0/n_assets:.1%}")

# 3.2: Maximum Sharpe Ratio (Simplified Mean-Variance)
print("\n3.2 Maximum Sharpe Ratio Portfolio")

# Simple gradient-free optimization
best_sharpe = -np.inf
best_weights = None

for _ in range(1000):
    # Random weights
    w = np.random.dirichlet(np.ones(n_assets))

    # Calculate Sharpe ratio
    port_return = (w * annual_returns.values).sum()
    port_vol = np.sqrt(w @ cov_matrix.values @ w)
    sharpe = port_return / port_vol if port_vol > 0 else 0

    if sharpe > best_sharpe:
        best_sharpe = sharpe
        best_weights = w

max_sharpe_weights = pd.Series(best_weights, index=tickers)
results['Max Sharpe'] = max_sharpe_weights
print(f"[OK] Sharpe ratio: {best_sharpe:.3f}")
print(f"[OK] Number of significant positions: {(max_sharpe_weights > 0.05).sum()}")

# 3.3: Minimum Variance
print("\n3.3 Minimum Variance Portfolio")

# Simple approach: find weights that minimize variance
min_var = np.inf
min_var_weights = None

for _ in range(1000):
    w = np.random.dirichlet(np.ones(n_assets))
    port_var = w @ cov_matrix.values @ w

    if port_var < min_var:
        min_var = port_var
        min_var_weights = w

min_var_weights = pd.Series(min_var_weights, index=tickers)
results['Min Variance'] = min_var_weights
print(f"[OK] Portfolio volatility: {np.sqrt(min_var):.2%}")

# 3.4: Concentrated Portfolio (Cardinality = 5)
print("\n3.4 Concentrated Portfolio (Max 5 Assets)")

# Select top 5 assets by Sharpe ratio
top_5 = sharpe_ratios.nlargest(5).index
concentrated_weights = pd.Series(0.0, index=tickers)

# Random search for best allocation among top 5
best_sharpe_concentrated = -np.inf
best_concentrated = None

for _ in range(1000):
    w = np.zeros(n_assets)
    top_5_indices = [tickers.index(t) for t in top_5]
    w[top_5_indices] = np.random.dirichlet(np.ones(5))

    port_return = (w * annual_returns.values).sum()
    port_vol = np.sqrt(w @ cov_matrix.values @ w)
    sharpe = port_return / port_vol if port_vol > 0 else 0

    if sharpe > best_sharpe_concentrated:
        best_sharpe_concentrated = sharpe
        best_concentrated = w

concentrated_weights = pd.Series(best_concentrated, index=tickers)
results['Concentrated'] = concentrated_weights
print(f"[OK] Number of assets: {(concentrated_weights > 1e-4).sum()}")
print(f"[OK] Sharpe ratio: {best_sharpe_concentrated:.3f}")
print()

# ============================================================================
# STEP 4: Performance Comparison
# ============================================================================
print("[STEP 4] Performance Comparison...")
print("-" * 80)

comparison_metrics = []

for strategy_name, weights in results.items():
    port_return = (weights * annual_returns).sum()
    port_vol = np.sqrt(weights.values @ cov_matrix.values @ weights.values)
    sharpe = port_return / port_vol if port_vol > 0 else 0
    n_assets_used = (weights > 1e-4).sum()

    comparison_metrics.append({
        'strategy': strategy_name,
        'expected_return': port_return,
        'volatility': port_vol,
        'sharpe_ratio': sharpe,
        'n_assets': n_assets_used
    })

comparison_df = pd.DataFrame(comparison_metrics).set_index('strategy')
print(comparison_df.to_string())
print()

# Save results
comparison_df.to_csv(SIMULATIONS_DIR / 'strategy_comparison.csv')
print(f"[OK] Results saved to {SIMULATIONS_DIR / 'strategy_comparison.csv'}")
print()

# ============================================================================
# STEP 5: Visualizations
# ============================================================================
print("[STEP 5] Generating Visualizations...")
print("-" * 80)

# 5.1: Price Evolution
fig, ax = plt.subplots(figsize=(12, 6))
for col in prices.columns[:5]:  # Plot first 5 for clarity
    ax.plot(prices.index, prices[col], label=col, linewidth=1.5, alpha=0.8)
ax.set_title('Synthetic Asset Prices Over Time', fontsize=14, fontweight='bold')
ax.set_xlabel('Date', fontsize=11)
ax.set_ylabel('Price (Base = 100)', fontsize=11)
ax.legend(loc='upper left', frameon=True, fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'asset_prices.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Saved: asset_prices.png")

# 5.2: Correlation Heatmap
fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(
    corr_matrix,
    mask=mask,
    cmap='coolwarm',
    center=0,
    annot=True,
    fmt='.2f',
    square=True,
    cbar_kws={'label': 'Correlation'},
    ax=ax,
    linewidths=0.5
)
ax.set_title('Asset Return Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Saved: correlation_matrix.png")

# 5.3: Portfolio Weights Comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, (strategy_name, weights) in enumerate(results.items()):
    active_weights = weights[weights > 1e-4].sort_values(ascending=True)

    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(active_weights)))
    axes[idx].barh(range(len(active_weights)), active_weights.values, color=colors)
    axes[idx].set_yticks(range(len(active_weights)))
    axes[idx].set_yticklabels(active_weights.index, fontsize=9)
    axes[idx].set_xlabel('Weight', fontsize=10)
    axes[idx].set_title(f'{strategy_name}', fontweight='bold', fontsize=11)
    axes[idx].grid(True, axis='x', alpha=0.3)

    # Add value labels
    for i, v in enumerate(active_weights.values):
        axes[idx].text(v + 0.01, i, f'{v:.1%}', va='center', fontsize=8)

plt.suptitle('Portfolio Weights Comparison', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'portfolio_weights.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Saved: portfolio_weights.png")

# 5.4: Performance Metrics
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

metrics_to_plot = ['expected_return', 'volatility', 'sharpe_ratio']
titles = ['Expected Annual Return', 'Annual Volatility', 'Sharpe Ratio']
colors = ['#2ecc71', '#e74c3c', '#3498db']

for idx, (metric, title, color) in enumerate(zip(metrics_to_plot, titles, colors)):
    values = comparison_df[metric]
    bars = axes[idx].bar(range(len(values)), values, color=color, alpha=0.7, edgecolor='black', linewidth=1.5)
    axes[idx].set_xticks(range(len(values)))
    axes[idx].set_xticklabels(values.index, rotation=45, ha='right', fontsize=9)
    axes[idx].set_title(title, fontweight='bold', fontsize=11)
    axes[idx].set_ylabel(title, fontsize=10)
    axes[idx].grid(True, axis='y', alpha=0.3, linestyle='--')

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, values)):
        height = bar.get_height()
        if metric in ['expected_return', 'volatility']:
            label = f'{val:.1%}'
        else:
            label = f'{val:.3f}'
        axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                      label, ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.suptitle('Strategy Performance Comparison', fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'performance_metrics.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Saved: performance_metrics.png")

# 5.5: Risk-Return Scatter
fig, ax = plt.subplots(figsize=(10, 7))

colors_scatter = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
markers = ['o', 's', '^', 'd']

for idx, strategy in enumerate(comparison_df.index):
    ret = comparison_df.loc[strategy, 'expected_return']
    vol = comparison_df.loc[strategy, 'volatility']
    sharpe = comparison_df.loc[strategy, 'sharpe_ratio']

    ax.scatter(vol, ret, s=300, alpha=0.7, label=strategy,
              color=colors_scatter[idx], marker=markers[idx],
              edgecolors='black', linewidths=2)

    # Add Sharpe ratio annotation
    ax.annotate(f'SR={sharpe:.3f}',
               xy=(vol, ret),
               xytext=(10, 10),
               textcoords='offset points',
               fontsize=8,
               bbox=dict(boxstyle='round,pad=0.3', facecolor=colors_scatter[idx], alpha=0.3))

ax.set_xlabel('Annual Volatility (Risk)', fontsize=12, fontweight='bold')
ax.set_ylabel('Expected Annual Return', fontsize=12, fontweight='bold')
ax.set_title('Risk-Return Profile of Portfolio Strategies', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', frameon=True, fontsize=10, shadow=True)
ax.grid(True, alpha=0.4, linestyle='--')

# Add reference line for Sharpe ratio contours
vol_range = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 100)
for sr in [0.5, 1.0, 1.5]:
    ax.plot(vol_range, sr * vol_range, 'k--', alpha=0.2, linewidth=0.8)
    ax.text(vol_range[-1], sr * vol_range[-1], f'SR={sr}',
           fontsize=7, alpha=0.5, rotation=45)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'risk_return_scatter.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Saved: risk_return_scatter.png")

# 5.6: Returns Distribution
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for idx, col in enumerate(tickers[:6]):
    asset_returns = returns[col] * 100  # Convert to percentage
    axes[idx].hist(asset_returns, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    axes[idx].axvline(asset_returns.mean(), color='red', linestyle='--',
                     linewidth=2, label=f'Mean={asset_returns.mean():.2f}%')
    axes[idx].set_title(col, fontweight='bold', fontsize=10)
    axes[idx].set_xlabel('Daily Return (%)', fontsize=9)
    axes[idx].set_ylabel('Frequency', fontsize=9)
    axes[idx].legend(fontsize=8)
    axes[idx].grid(True, alpha=0.3)

plt.suptitle('Distribution of Daily Returns', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'returns_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Saved: returns_distribution.png")

print()
print("=" * 80)
print("DEMO COMPLETED SUCCESSFULLY!")
print("=" * 80)
print(f"\nOutputs saved to:")
print(f"  - Figures: {FIGURES_DIR}")
print(f"  - Simulations: {SIMULATIONS_DIR}")
print(f"\nGenerated {len(list(FIGURES_DIR.glob('*.png')))} visualizations")
print()
