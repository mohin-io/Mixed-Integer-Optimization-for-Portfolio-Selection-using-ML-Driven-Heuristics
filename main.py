"""
Main Pipeline for Mixed-Integer Portfolio Optimization

Demonstrates the complete workflow:
1. Data loading and preprocessing
2. Forecasting (returns, volatility, covariance)
3. Portfolio optimization (MIO, GA, benchmarks)
4. Performance evaluation
5. Visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import time

# Import project modules
from src.data.loader import AssetDataLoader
from src.data.preprocessing import DataPreprocessor
from src.forecasting.returns_forecast import ReturnsForecast, EnsembleReturnsForecast
from src.forecasting.volatility_forecast import VolatilityForecast
from src.forecasting.covariance import CovarianceEstimator
from src.optimization.mio_optimizer import MIOOptimizer, NaiveMeanVarianceOptimizer, OptimizationConfig
from src.heuristics.clustering import AssetClusterer
from src.heuristics.genetic_algorithm import GeneticOptimizer, GAConfig

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

# Create output directories
OUTPUT_DIR = Path('outputs')
FIGURES_DIR = OUTPUT_DIR / 'figures'
SIMULATIONS_DIR = OUTPUT_DIR / 'simulations'

for dir_path in [OUTPUT_DIR, FIGURES_DIR, SIMULATIONS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


def main():
    """Run complete portfolio optimization pipeline."""

    print("=" * 80)
    print("MIXED-INTEGER PORTFOLIO OPTIMIZATION WITH ML-DRIVEN HEURISTICS")
    print("=" * 80)
    print()

    # -------------------------------------------------------------------------
    # STEP 1: DATA LOADING
    # -------------------------------------------------------------------------
    print("[STEP 1] Loading Data...")
    print("-" * 80)

    loader = AssetDataLoader()

    # Use a diverse set of assets across sectors
    tickers = [
        # Tech
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META',
        # Finance
        'JPM', 'V', 'MA', 'BAC',
        # Healthcare
        'JNJ', 'UNH', 'PFE',
        # Consumer
        'WMT', 'PG', 'KO', 'MCD',
        # Energy
        'XOM', 'CVX',
        # Industrial
        'CAT', 'BA'
    ]

    start_date = '2020-01-01'
    end_date = '2023-12-31'

    prices = loader.fetch_prices(tickers, start_date, end_date, use_cache=True)
    returns = loader.compute_returns(prices, method='log')

    print(f"✓ Loaded {len(prices.columns)} assets")
    print(f"✓ Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"✓ Total observations: {len(prices)}")
    print()

    # -------------------------------------------------------------------------
    # STEP 2: DATA PREPROCESSING
    # -------------------------------------------------------------------------
    print("[STEP 2] Data Preprocessing...")
    print("-" * 80)

    preprocessor = DataPreprocessor()

    # Factor analysis
    factors = preprocessor.calculate_factors(prices, returns)
    print("✓ Computed factor exposures (momentum, volatility, beta)")

    # Correlation matrix
    corr_matrix = preprocessor.compute_correlation_matrix(returns)
    print(f"✓ Correlation matrix shape: {corr_matrix.shape}")

    # Filter liquid assets
    liquid_prices = preprocessor.filter_assets_by_liquidity(prices, min_trading_days=900)
    liquid_returns = loader.compute_returns(liquid_prices)

    print(f"✓ Filtered to {len(liquid_prices.columns)} liquid assets")
    print()

    # -------------------------------------------------------------------------
    # STEP 3: FORECASTING
    # -------------------------------------------------------------------------
    print("[STEP 3] Forecasting Returns, Volatility, and Covariance...")
    print("-" * 80)

    # Returns forecasting
    returns_forecaster = ReturnsForecast(method='historical')
    returns_forecaster.fit(liquid_returns)
    expected_returns = returns_forecaster.predict()

    print(f"✓ Expected returns (annualized):")
    print(f"  Mean: {expected_returns.mean():.2%}")
    print(f"  Min: {expected_returns.min():.2%}, Max: {expected_returns.max():.2%}")

    # Volatility forecasting
    vol_forecaster = VolatilityForecast(method='historical')
    vol_forecaster.fit(liquid_returns)
    expected_volatility = vol_forecaster.predict()

    print(f"✓ Expected volatility (annualized):")
    print(f"  Mean: {expected_volatility.mean():.2%}")

    # Covariance estimation
    cov_estimator = CovarianceEstimator(method='ledoit_wolf')
    cov_matrix = cov_estimator.estimate(liquid_returns)

    print(f"✓ Covariance matrix estimated (Ledoit-Wolf shrinkage)")
    print(f"  Condition number: {cov_estimator.compute_condition_number():.2f}")
    print()

    # -------------------------------------------------------------------------
    # STEP 4: ASSET CLUSTERING (ML-Driven Pre-selection)
    # -------------------------------------------------------------------------
    print("[STEP 4] Asset Clustering for Pre-selection...")
    print("-" * 80)

    clusterer = AssetClusterer(method='kmeans')
    clusterer.fit(liquid_returns, n_clusters=5)

    cluster_summary = clusterer.get_cluster_summary(liquid_returns)
    print(cluster_summary.to_string(index=False))

    # Select diverse subset
    selected_assets = clusterer.select_representatives(
        liquid_returns,
        n_per_cluster=2,
        selection_criterion='sharpe'
    )

    print(f"\n✓ Selected {len(selected_assets)} diverse assets for optimization")
    print(f"  Assets: {', '.join(selected_assets)}")
    print()

    # Filter data to selected assets
    selected_returns = liquid_returns[selected_assets]
    selected_expected_returns = expected_returns[selected_assets]
    selected_cov_matrix = cov_matrix.loc[selected_assets, selected_assets]

    # -------------------------------------------------------------------------
    # STEP 5: PORTFOLIO OPTIMIZATION
    # -------------------------------------------------------------------------
    print("[STEP 5] Portfolio Optimization...")
    print("-" * 80)

    results = {}

    # 5.1: Equal Weight Portfolio
    print("\n5.1 Equal Weight Portfolio (Baseline)")
    n_selected = len(selected_assets)
    equal_weights = pd.Series(1.0 / n_selected, index=selected_assets)
    results['Equal Weight'] = equal_weights
    print(f"✓ Assigned equal weights: {1.0/n_selected:.2%} each")

    # 5.2: Naive Mean-Variance
    print("\n5.2 Naive Mean-Variance Optimization")
    naive_optimizer = NaiveMeanVarianceOptimizer(risk_aversion=2.5)
    naive_weights = naive_optimizer.optimize(selected_expected_returns, selected_cov_matrix)
    results['Naive MVO'] = naive_weights
    print(f"✓ Number of assets: {(naive_weights > 1e-4).sum()}")

    # 5.3: Mixed-Integer Optimization
    print("\n5.3 Mixed-Integer Optimization (MIO)")
    mio_config = OptimizationConfig(
        risk_aversion=2.5,
        max_assets=5,
        min_weight=0.10,
        max_weight=0.40,
        fixed_cost=0.0005,
        proportional_cost=0.001
    )

    mio_optimizer = MIOOptimizer(config=mio_config)
    start_time = time.time()
    mio_weights = mio_optimizer.optimize(selected_expected_returns, selected_cov_matrix)
    mio_time = time.time() - start_time

    results['MIO'] = mio_weights
    print(f"✓ Solve time: {mio_time:.2f} seconds")
    print(f"✓ Status: {mio_optimizer.solution['status']}")
    print(f"✓ Number of assets: {mio_optimizer.solution['n_assets']}")

    # 5.4: Genetic Algorithm
    print("\n5.4 Genetic Algorithm Optimization")
    ga_config = GAConfig(
        population_size=100,
        generations=30,
        max_assets=5,
        min_weight=0.10
    )

    ga_optimizer = GeneticOptimizer(config=ga_config)
    start_time = time.time()
    ga_weights = ga_optimizer.optimize(selected_expected_returns, selected_cov_matrix)
    ga_time = time.time() - start_time

    results['Genetic Algorithm'] = ga_weights
    print(f"✓ Solve time: {ga_time:.2f} seconds")
    print(f"✓ Best fitness (Sharpe): {ga_optimizer.best_solution['fitness']:.4f}")
    print()

    # -------------------------------------------------------------------------
    # STEP 6: PERFORMANCE COMPARISON
    # -------------------------------------------------------------------------
    print("[STEP 6] Performance Comparison...")
    print("-" * 80)

    comparison_metrics = []

    for strategy_name, weights in results.items():
        metrics = mio_optimizer.compute_portfolio_metrics(
            weights,
            selected_expected_returns,
            selected_cov_matrix
        )
        metrics['strategy'] = strategy_name
        comparison_metrics.append(metrics)

    comparison_df = pd.DataFrame(comparison_metrics).set_index('strategy')

    print(comparison_df.to_string())
    print()

    # Save results
    comparison_df.to_csv(SIMULATIONS_DIR / 'strategy_comparison.csv')
    print(f"✓ Results saved to {SIMULATIONS_DIR / 'strategy_comparison.csv'}")
    print()

    # -------------------------------------------------------------------------
    # STEP 7: VISUALIZATION
    # -------------------------------------------------------------------------
    print("[STEP 7] Generating Visualizations...")
    print("-" * 80)

    # 7.1: Price time series
    fig, ax = plt.subplots(figsize=(12, 6))
    normalized_prices = prices / prices.iloc[0] * 100
    for col in normalized_prices.columns[:5]:  # Plot first 5 for clarity
        ax.plot(normalized_prices.index, normalized_prices[col], label=col, linewidth=1.5)
    ax.set_title('Normalized Asset Prices (Base = 100)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Normalized Price')
    ax.legend(loc='best', frameon=True)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'asset_prices.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: asset_prices.png")

    # 7.2: Correlation heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        liquid_returns.corr(),
        cmap='coolwarm',
        center=0,
        annot=False,
        square=True,
        cbar_kws={'label': 'Correlation'},
        ax=ax
    )
    ax.set_title('Asset Return Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: correlation_matrix.png")

    # 7.3: Portfolio weights comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (strategy_name, weights) in enumerate(results.items()):
        active_weights = weights[weights > 1e-4].sort_values(ascending=True)
        axes[idx].barh(range(len(active_weights)), active_weights.values, color='steelblue')
        axes[idx].set_yticks(range(len(active_weights)))
        axes[idx].set_yticklabels(active_weights.index)
        axes[idx].set_xlabel('Weight')
        axes[idx].set_title(f'{strategy_name}', fontweight='bold')
        axes[idx].grid(True, axis='x', alpha=0.3)

        # Add value labels
        for i, v in enumerate(active_weights.values):
            axes[idx].text(v, i, f' {v:.1%}', va='center')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'portfolio_weights.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: portfolio_weights.png")

    # 7.4: Performance metrics comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics_to_plot = ['expected_return', 'volatility', 'sharpe_ratio']
    titles = ['Expected Return', 'Volatility', 'Sharpe Ratio']
    colors = ['green', 'red', 'blue']

    for idx, (metric, title, color) in enumerate(zip(metrics_to_plot, titles, colors)):
        comparison_df[metric].plot(kind='bar', ax=axes[idx], color=color, alpha=0.7)
        axes[idx].set_title(title, fontweight='bold')
        axes[idx].set_ylabel(title)
        axes[idx].set_xlabel('')
        axes[idx].grid(True, axis='y', alpha=0.3)
        axes[idx].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'performance_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: performance_metrics.png")

    # 7.5: Risk-Return scatter
    fig, ax = plt.subplots(figsize=(10, 6))

    for strategy in comparison_df.index:
        ret = comparison_df.loc[strategy, 'expected_return']
        vol = comparison_df.loc[strategy, 'volatility']
        ax.scatter(vol, ret, s=200, alpha=0.7, label=strategy)

    ax.set_xlabel('Volatility (Annual)', fontsize=12)
    ax.set_ylabel('Expected Return (Annual)', fontsize=12)
    ax.set_title('Risk-Return Profile', fontsize=14, fontweight='bold')
    ax.legend(loc='best', frameon=True)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'risk_return_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: risk_return_scatter.png")

    print()
    print("=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\nOutputs saved to:")
    print(f"  - Figures: {FIGURES_DIR}")
    print(f"  - Simulations: {SIMULATIONS_DIR}")
    print()


if __name__ == "__main__":
    main()
