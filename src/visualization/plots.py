"""
Static Visualization Module for Portfolio Optimization

Provides reusable plotting functions for:
1. Price & Returns - Time series of asset prices and log-returns
2. Correlation Matrix - Heatmap of asset correlations
3. Factor Exposures - Bar chart of factor loadings
4. Forecasting Performance - Predicted vs actual scatter plots
5. Efficient Frontier - Risk-return trade-off with transaction costs
6. Portfolio Weights - Stacked area chart over time
7. Performance Metrics - Cumulative returns, drawdowns
8. Heuristic Convergence - GA/SA fitness over iterations
9. Runtime Comparison - Bar chart of solver times
10. Cost Analysis - Transaction costs breakdown
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Dict, Tuple
from pathlib import Path
import warnings

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 100


class PortfolioPlotter:
    """Collection of static plotting functions for portfolio analysis."""

    @staticmethod
    def plot_price_series(
        prices: pd.DataFrame,
        title: str = "Asset Price Time Series",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot time series of asset prices.

        Args:
            prices: DataFrame of prices (index: dates, columns: tickers)
            title: Plot title
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(14, 7))

        # Normalize prices to start at 100
        normalized_prices = 100 * prices / prices.iloc[0]

        for col in normalized_prices.columns:
            ax.plot(normalized_prices.index, normalized_prices[col], label=col, linewidth=2)

        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Normalized Price (Base=100)', fontsize=12)
        ax.legend(loc='best', ncol=2, fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    @staticmethod
    def plot_returns_distribution(
        returns: pd.DataFrame,
        title: str = "Returns Distribution",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot histogram of returns distribution.

        Args:
            returns: DataFrame of returns
            title: Plot title
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # Overall returns distribution
        all_returns = returns.values.flatten()
        axes[0].hist(all_returns, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        axes[0].set_title(f"{title} - All Assets", fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Returns', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Individual asset box plots
        returns.boxplot(ax=axes[1], rot=45)
        axes[1].set_title("Returns Distribution by Asset", fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Returns', fontsize=12)
        axes[1].axhline(0, color='red', linestyle='--', linewidth=2)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    @staticmethod
    def plot_correlation_matrix(
        returns: pd.DataFrame,
        title: str = "Asset Correlation Matrix",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot correlation matrix heatmap.

        Args:
            returns: DataFrame of returns
            title: Plot title
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 10))

        corr_matrix = returns.corr()

        # Create heatmap
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.5,
            cbar_kws={'label': 'Correlation'},
            ax=ax
        )

        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    @staticmethod
    def plot_efficient_frontier(
        returns: List[float],
        volatilities: List[float],
        labels: Optional[List[str]] = None,
        title: str = "Efficient Frontier",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot efficient frontier.

        Args:
            returns: List of portfolio returns
            volatilities: List of portfolio volatilities
            labels: Portfolio labels
            title: Plot title
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        # Convert to percentages
        returns_pct = np.array(returns) * 100
        volatilities_pct = np.array(volatilities) * 100

        # Scatter plot
        scatter = ax.scatter(
            volatilities_pct,
            returns_pct,
            c=np.array(returns) / np.array(volatilities),  # Sharpe ratio for color
            s=100,
            cmap='viridis',
            edgecolors='black',
            linewidth=1.5,
            alpha=0.8
        )

        # Add labels if provided
        if labels:
            for i, label in enumerate(labels):
                ax.annotate(
                    label,
                    (volatilities_pct[i], returns_pct[i]),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=10,
                    fontweight='bold'
                )

        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Sharpe Ratio', fontsize=12)

        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Volatility (%)', fontsize=12)
        ax.set_ylabel('Expected Return (%)', fontsize=12)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    @staticmethod
    def plot_portfolio_weights(
        weights: pd.Series,
        title: str = "Portfolio Weights",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot portfolio weights as bar chart.

        Args:
            weights: Series of portfolio weights
            title: Plot title
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        # Sort weights
        weights_sorted = weights[weights > 1e-6].sort_values(ascending=False)

        # Bar chart
        bars = ax.bar(
            range(len(weights_sorted)),
            weights_sorted.values * 100,
            color='steelblue',
            edgecolor='black',
            linewidth=1.5
        )

        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.5,
                f'{height:.1f}%',
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold'
            )

        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Assets', fontsize=12)
        ax.set_ylabel('Weight (%)', fontsize=12)
        ax.set_xticks(range(len(weights_sorted)))
        ax.set_xticklabels(weights_sorted.index, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, max(weights_sorted.values * 100) * 1.15)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    @staticmethod
    def plot_weights_over_time(
        weights_history: pd.DataFrame,
        title: str = "Portfolio Weights Over Time",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot stacked area chart of weights over time.

        Args:
            weights_history: DataFrame of weights (index: dates, columns: tickers)
            title: Plot title
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(14, 7))

        # Stacked area plot
        ax.stackplot(
            weights_history.index,
            *[weights_history[col].values for col in weights_history.columns],
            labels=weights_history.columns,
            alpha=0.8
        )

        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Weight', fontsize=12)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=1)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    @staticmethod
    def plot_cumulative_returns(
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        title: str = "Cumulative Returns",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot cumulative returns over time.

        Args:
            returns: Series of returns
            benchmark_returns: Optional benchmark returns
            title: Plot title
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(14, 7))

        # Calculate cumulative returns
        cumulative = (1 + returns).cumprod()

        ax.plot(cumulative.index, cumulative.values, label='Strategy', linewidth=2, color='steelblue')

        if benchmark_returns is not None:
            benchmark_cumulative = (1 + benchmark_returns).cumprod()
            ax.plot(
                benchmark_cumulative.index,
                benchmark_cumulative.values,
                label='Benchmark',
                linewidth=2,
                color='orange',
                linestyle='--'
            )

        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Cumulative Return', fontsize=12)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.axhline(1, color='black', linestyle='-', linewidth=1, alpha=0.5)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    @staticmethod
    def plot_drawdown(
        returns: pd.Series,
        title: str = "Drawdown Analysis",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot drawdown chart.

        Args:
            returns: Series of returns
            title: Plot title
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        # Cumulative returns
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()

        ax1.plot(cumulative.index, cumulative.values, label='Cumulative Return', linewidth=2)
        ax1.plot(running_max.index, running_max.values, label='Running Max', linewidth=2, linestyle='--')
        ax1.set_title(f"{title} - Cumulative Returns", fontsize=14, fontweight='bold')
        ax1.set_ylabel('Cumulative Return', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Drawdown
        drawdown = (cumulative - running_max) / running_max

        ax2.fill_between(drawdown.index, drawdown.values, 0, color='red', alpha=0.3, label='Drawdown')
        ax2.plot(drawdown.index, drawdown.values, color='darkred', linewidth=2)
        ax2.set_title("Drawdown", fontsize=14, fontweight='bold')
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Drawdown (%)', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Format y-axis as percentage
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.1f}%'))

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    @staticmethod
    def plot_convergence(
        fitness_history: List[float],
        title: str = "Optimization Convergence",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot convergence of optimization algorithm.

        Args:
            fitness_history: List of fitness values over iterations
            title: Plot title
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(fitness_history, linewidth=2, color='steelblue')
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Fitness (Sharpe Ratio)', fontsize=12)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    @staticmethod
    def plot_strategy_comparison(
        results: Dict[str, Dict[str, float]],
        metrics: List[str] = ['annual_return', 'annual_volatility', 'sharpe_ratio', 'max_drawdown'],
        title: str = "Strategy Comparison",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot comparison of multiple strategies.

        Args:
            results: Dict of {strategy_name: {metric: value}}
            metrics: List of metrics to compare
            title: Plot title
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6))

        if n_metrics == 1:
            axes = [axes]

        strategies = list(results.keys())
        colors = sns.color_palette('Set2', len(strategies))

        for idx, metric in enumerate(metrics):
            values = [results[strat].get(metric, 0) for strat in strategies]

            bars = axes[idx].bar(range(len(strategies)), values, color=colors, edgecolor='black', linewidth=1.5)

            # Add value labels
            for i, bar in enumerate(bars):
                height = bar.get_height()
                axes[idx].text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f'{height:.3f}' if abs(height) < 10 else f'{height:.1f}',
                    ha='center',
                    va='bottom' if height >= 0 else 'top',
                    fontsize=10,
                    fontweight='bold'
                )

            axes[idx].set_title(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
            axes[idx].set_xticks(range(len(strategies)))
            axes[idx].set_xticklabels(strategies, rotation=45, ha='right')
            axes[idx].grid(True, alpha=0.3, axis='y')

        fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig


def save_all_figures(output_dir: str = "outputs/figures"):
    """
    Create output directory for saving figures.

    Args:
        output_dir: Directory path
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Figures will be saved to: {output_dir}")


if __name__ == "__main__":
    # Example usage
    print("Portfolio Plotter Demo\n")

    from ..data.loader import AssetDataLoader

    # Load sample data
    loader = AssetDataLoader()
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']

    try:
        prices = loader.fetch_prices(tickers, '2022-01-01', '2023-12-31')
        returns = loader.compute_returns(prices)

        plotter = PortfolioPlotter()

        print("Generating plots...\n")

        # 1. Price series
        fig1 = plotter.plot_price_series(prices, title="Tech Stocks Price Series")
        print("✓ Price series plot")

        # 2. Correlation matrix
        fig2 = plotter.plot_correlation_matrix(returns, title="Returns Correlation Matrix")
        print("✓ Correlation matrix plot")

        # 3. Portfolio weights
        weights = pd.Series([0.3, 0.25, 0.2, 0.15, 0.1], index=tickers)
        fig3 = plotter.plot_portfolio_weights(weights, title="Sample Portfolio Weights")
        print("✓ Portfolio weights plot")

        # 4. Cumulative returns
        portfolio_returns = (returns * weights).sum(axis=1)
        fig4 = plotter.plot_cumulative_returns(portfolio_returns, title="Portfolio Performance")
        print("✓ Cumulative returns plot")

        # 5. Drawdown
        fig5 = plotter.plot_drawdown(portfolio_returns, title="Portfolio Drawdown")
        print("✓ Drawdown plot")

        print("\nAll plots generated successfully!")
        print("Close the plot windows to continue...")

        plt.show()

    except Exception as e:
        print(f"Error in demo: {str(e)}")
        import traceback
        traceback.print_exc()
