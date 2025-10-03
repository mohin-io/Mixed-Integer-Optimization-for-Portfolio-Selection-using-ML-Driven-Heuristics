#!/usr/bin/env python
"""
Strategy Comparison Script

Compares multiple portfolio optimization strategies and generates
detailed comparison reports with visualizations.

Usage:
    python scripts/compare_strategies.py
    python scripts/compare_strategies.py --assets 20 --days 2000
    python scripts/compare_strategies.py --export-csv results.csv
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from datetime import datetime

sns.set_style('whitegrid')


class StrategyComparator:
    """Compare multiple portfolio optimization strategies."""

    def __init__(self, n_assets=10, n_days=1000, seed=42):
        """Initialize comparator with synthetic data."""
        np.random.seed(seed)
        self.n_assets = n_assets
        self.n_days = n_days
        self.tickers = [f'ASSET_{i+1}' for i in range(n_assets)]

        # Generate data
        self._generate_data()

    def _generate_data(self):
        """Generate synthetic market data."""
        # Factor model
        n_factors = 3
        factor_loadings = np.random.randn(self.n_assets, n_factors) * 0.3
        factor_returns = np.random.randn(self.n_days, n_factors) * 0.01
        idiosyncratic = np.random.randn(self.n_days, self.n_assets) * 0.005

        drift = np.random.uniform(0.0001, 0.0005, self.n_assets)
        returns_matrix = factor_returns @ factor_loadings.T + idiosyncratic + drift

        dates = pd.date_range('2020-01-01', periods=self.n_days, freq='D')
        self.returns = pd.DataFrame(returns_matrix, index=dates, columns=self.tickers)
        self.prices = (1 + self.returns).cumprod() * 100

        # Compute statistics
        self.annual_returns = self.returns.mean() * 252
        self.annual_volatility = self.returns.std() * np.sqrt(252)
        self.cov_matrix = self.returns.cov() * 252

    def optimize(self, strategy, **kwargs):
        """Optimize portfolio using specified strategy."""
        if strategy == 'equal_weight':
            return np.ones(self.n_assets) / self.n_assets

        elif strategy == 'inverse_volatility':
            inv_vol = 1.0 / self.annual_volatility.values
            return inv_vol / inv_vol.sum()

        elif strategy == 'max_sharpe':
            best_sharpe = -np.inf
            best_weights = None

            for _ in range(kwargs.get('iterations', 10000)):
                w = np.random.dirichlet(np.ones(self.n_assets))
                port_return = (w * self.annual_returns.values).sum()
                port_vol = np.sqrt(w @ self.cov_matrix.values @ w)
                sharpe = port_return / port_vol if port_vol > 0 else 0

                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_weights = w

            return best_weights

        elif strategy == 'min_variance':
            min_var = np.inf
            min_var_weights = None

            for _ in range(kwargs.get('iterations', 10000)):
                w = np.random.dirichlet(np.ones(self.n_assets))
                port_var = w @ self.cov_matrix.values @ w

                if port_var < min_var:
                    min_var = port_var
                    min_var_weights = w

            return min_var_weights

        elif strategy == 'risk_parity':
            # Simplified risk parity (equal risk contribution)
            inv_vol = 1.0 / np.sqrt(np.diag(self.cov_matrix.values))
            weights = inv_vol / inv_vol.sum()

            # Iterative adjustment
            for _ in range(10):
                marginal_contrib = (self.cov_matrix.values @ weights)
                risk_contrib = weights * marginal_contrib
                weights *= (risk_contrib.mean() / risk_contrib)
                weights /= weights.sum()

            return weights

        elif strategy == 'concentrated':
            max_assets = kwargs.get('max_assets', 5)
            sharpe_ratios = self.annual_returns / self.annual_volatility
            top_assets = sharpe_ratios.nlargest(max_assets).index

            best_sharpe = -np.inf
            best_weights = None

            for _ in range(kwargs.get('iterations', 10000)):
                w = np.zeros(self.n_assets)
                top_indices = [self.tickers.index(t) for t in top_assets]
                w[top_indices] = np.random.dirichlet(np.ones(max_assets))

                port_return = (w * self.annual_returns.values).sum()
                port_vol = np.sqrt(w @ self.cov_matrix.values @ w)
                sharpe = port_return / port_vol if port_vol > 0 else 0

                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_weights = w

            return best_weights

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def evaluate(self, weights):
        """Evaluate portfolio metrics."""
        port_return = (weights * self.annual_returns.values).sum()
        port_vol = np.sqrt(weights @ self.cov_matrix.values @ weights)
        sharpe = port_return / port_vol if port_vol > 0 else 0
        n_assets = (weights > 1e-4).sum()

        # Additional metrics
        diversification_ratio = (weights * self.annual_volatility.values).sum() / port_vol
        concentration = (weights ** 2).sum()  # Herfindahl index

        return {
            'return': port_return,
            'volatility': port_vol,
            'sharpe': sharpe,
            'n_assets': n_assets,
            'diversification_ratio': diversification_ratio,
            'concentration': concentration
        }

    def run_comparison(self, strategies):
        """Run comparison across all strategies."""
        results = {}
        weights_dict = {}

        for name, config in strategies.items():
            strategy_type = config.get('type', name.lower().replace(' ', '_'))
            kwargs = config.get('kwargs', {})

            print(f"Optimizing {name}...")
            weights = self.optimize(strategy_type, **kwargs)
            metrics = self.evaluate(weights)

            results[name] = metrics
            weights_dict[name] = pd.Series(weights, index=self.tickers)

        return pd.DataFrame(results).T, weights_dict

    def plot_comparison(self, results_df, weights_dict, save_path=None):
        """Create comprehensive comparison visualizations."""
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. Risk-Return Scatter
        ax1 = fig.add_subplot(gs[0, :2])
        colors = plt.cm.Set3(np.linspace(0, 1, len(results_df)))

        for idx, (strategy, row) in enumerate(results_df.iterrows()):
            ax1.scatter(row['volatility'], row['return'], s=300, alpha=0.7,
                       label=strategy, color=colors[idx], edgecolors='black', linewidth=2)
            ax1.annotate(f"SR={row['sharpe']:.2f}",
                        xy=(row['volatility'], row['return']),
                        xytext=(10, 10), textcoords='offset points', fontsize=8)

        ax1.set_xlabel('Annual Volatility', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Expected Annual Return', fontsize=11, fontweight='bold')
        ax1.set_title('Risk-Return Profile', fontsize=13, fontweight='bold')
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)

        # 2. Sharpe Ratio Comparison
        ax2 = fig.add_subplot(gs[0, 2])
        sharpe_sorted = results_df['sharpe'].sort_values(ascending=True)
        ax2.barh(range(len(sharpe_sorted)), sharpe_sorted.values, color='steelblue', edgecolor='black')
        ax2.set_yticks(range(len(sharpe_sorted)))
        ax2.set_yticklabels(sharpe_sorted.index, fontsize=9)
        ax2.set_xlabel('Sharpe Ratio', fontsize=10)
        ax2.set_title('Sharpe Ratio Ranking', fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')

        # 3. Portfolio Weights Heatmap
        ax3 = fig.add_subplot(gs[1, :])
        weights_df = pd.DataFrame(weights_dict).T
        sns.heatmap(weights_df, annot=False, cmap='YlOrRd', cbar_kws={'label': 'Weight'},
                   linewidths=0.5, ax=ax3)
        ax3.set_title('Portfolio Weights Distribution', fontsize=13, fontweight='bold')
        ax3.set_xlabel('Assets', fontsize=10)
        ax3.set_ylabel('Strategy', fontsize=10)

        # 4. Diversification Metrics
        ax4 = fig.add_subplot(gs[2, 0])
        results_df['n_assets'].plot(kind='bar', ax=ax4, color='green', alpha=0.7, edgecolor='black')
        ax4.set_title('Number of Assets', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Count', fontsize=10)
        ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha='right', fontsize=8)
        ax4.grid(True, alpha=0.3, axis='y')

        # 5. Concentration Index
        ax5 = fig.add_subplot(gs[2, 1])
        results_df['concentration'].plot(kind='bar', ax=ax5, color='orange', alpha=0.7, edgecolor='black')
        ax5.set_title('Concentration (Herfindahl)', fontsize=11, fontweight='bold')
        ax5.set_ylabel('Index', fontsize=10)
        ax5.set_xticklabels(ax5.get_xticklabels(), rotation=45, ha='right', fontsize=8)
        ax5.grid(True, alpha=0.3, axis='y')

        # 6. Diversification Ratio
        ax6 = fig.add_subplot(gs[2, 2])
        results_df['diversification_ratio'].plot(kind='bar', ax=ax6, color='purple', alpha=0.7, edgecolor='black')
        ax6.set_title('Diversification Ratio', fontsize=11, fontweight='bold')
        ax6.set_ylabel('Ratio', fontsize=10)
        ax6.set_xticklabels(ax6.get_xticklabels(), rotation=45, ha='right', fontsize=8)
        ax6.grid(True, alpha=0.3, axis='y')

        plt.suptitle('Portfolio Strategy Comparison Dashboard', fontsize=16, fontweight='bold', y=0.995)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nSaved visualization to: {save_path}")

        plt.tight_layout()
        plt.show()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Compare portfolio optimization strategies')
    parser.add_argument('--assets', type=int, default=10, help='Number of assets')
    parser.add_argument('--days', type=int, default=1000, help='Number of trading days')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--export-csv', type=str, help='Export results to CSV')
    parser.add_argument('--export-plot', type=str, default='outputs/figures/strategy_comparison_full.png',
                       help='Save plot to file')

    args = parser.parse_args()

    print("=" * 70)
    print("PORTFOLIO STRATEGY COMPARISON")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Assets: {args.assets}")
    print(f"  Days: {args.days}")
    print(f"  Seed: {args.seed}")
    print()

    # Initialize comparator
    comparator = StrategyComparator(n_assets=args.assets, n_days=args.days, seed=args.seed)

    # Define strategies to compare
    strategies = {
        'Equal Weight': {'type': 'equal_weight'},
        'Inverse Volatility': {'type': 'inverse_volatility'},
        'Risk Parity': {'type': 'risk_parity'},
        'Min Variance': {'type': 'min_variance', 'kwargs': {'iterations': 10000}},
        'Max Sharpe': {'type': 'max_sharpe', 'kwargs': {'iterations': 10000}},
        'Concentrated (5)': {'type': 'concentrated', 'kwargs': {'max_assets': 5, 'iterations': 10000}},
    }

    # Run comparison
    results_df, weights_dict = comparator.run_comparison(strategies)

    # Display results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(results_df.round(4))
    print("=" * 70)
    print(f"\nBest Sharpe Ratio: {results_df['sharpe'].max():.3f} ({results_df['sharpe'].idxmax()})")
    print(f"Lowest Volatility: {results_df['volatility'].min():.2%} ({results_df['volatility'].idxmin()})")
    print(f"Highest Return: {results_df['return'].max():.2%} ({results_df['return'].idxmax()})")
    print("=" * 70)

    # Export if requested
    if args.export_csv:
        results_df.to_csv(args.export_csv)
        print(f"\nExported results to: {args.export_csv}")

    # Create visualization
    comparator.plot_comparison(results_df, weights_dict, save_path=args.export_plot)

    return 0


if __name__ == '__main__':
    sys.exit(main())
