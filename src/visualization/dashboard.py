"""
Streamlit Dashboard for Portfolio Optimization

Interactive web application for exploring portfolio optimization strategies.

Run with:
    streamlit run src/visualization/dashboard.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Portfolio Optimizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #555;
        text-align: center;
        padding-bottom: 2rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def generate_synthetic_data(n_assets, n_days, seed):
    """Generate synthetic market data."""
    np.random.seed(seed)
    tickers = [f'ASSET_{i+1}' for i in range(n_assets)]

    # Factor model
    n_factors = 3
    factor_loadings = np.random.randn(n_assets, n_factors) * 0.3
    factor_returns = np.random.randn(n_days, n_factors) * 0.01
    idiosyncratic = np.random.randn(n_days, n_assets) * 0.005

    drift = np.random.uniform(0.0001, 0.0005, n_assets)
    returns_matrix = factor_returns @ factor_loadings.T + idiosyncratic + drift

    dates = pd.date_range('2020-01-01', periods=n_days, freq='D')
    returns = pd.DataFrame(returns_matrix, index=dates, columns=tickers)
    prices = (1 + returns).cumprod() * 100

    return prices, returns


def optimize_portfolio(returns, strategy, max_assets=None, risk_aversion=2.5):
    """Optimize portfolio using specified strategy."""
    n_assets = len(returns.columns)
    annual_returns = returns.mean() * 252
    annual_volatility = returns.std() * np.sqrt(252)
    cov_matrix = returns.cov() * 252

    if strategy == 'Equal Weight':
        weights = np.ones(n_assets) / n_assets

    elif strategy == 'Max Sharpe':
        best_sharpe = -np.inf
        best_weights = None

        for _ in range(10000):
            w = np.random.dirichlet(np.ones(n_assets))
            port_return = (w * annual_returns.values).sum()
            port_vol = np.sqrt(w @ cov_matrix.values @ w)
            sharpe = port_return / port_vol if port_vol > 0 else 0

            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_weights = w

        weights = best_weights

    elif strategy == 'Min Variance':
        min_var = np.inf
        min_var_weights = None

        for _ in range(10000):
            w = np.random.dirichlet(np.ones(n_assets))
            port_var = w @ cov_matrix.values @ w

            if port_var < min_var:
                min_var = port_var
                min_var_weights = w

        weights = min_var_weights

    elif strategy == 'Concentrated':
        sharpe_ratios = annual_returns / annual_volatility
        top_assets = sharpe_ratios.nlargest(max_assets).index

        best_sharpe = -np.inf
        best_weights = None

        for _ in range(10000):
            w = np.zeros(n_assets)
            top_indices = [returns.columns.get_loc(t) for t in top_assets]
            w[top_indices] = np.random.dirichlet(np.ones(max_assets))

            port_return = (w * annual_returns.values).sum()
            port_vol = np.sqrt(w @ cov_matrix.values @ w)
            sharpe = port_return / port_vol if port_vol > 0 else 0

            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_weights = w

        weights = best_weights

    else:
        weights = np.ones(n_assets) / n_assets

    return pd.Series(weights, index=returns.columns), annual_returns, cov_matrix


def evaluate_portfolio(weights, annual_returns, cov_matrix):
    """Evaluate portfolio metrics."""
    port_return = (weights * annual_returns).sum()
    port_vol = np.sqrt(weights.values @ cov_matrix.values @ weights.values)
    sharpe = port_return / port_vol if port_vol > 0 else 0
    n_assets = (weights > 1e-4).sum()

    return {
        'return': port_return,
        'volatility': port_vol,
        'sharpe': sharpe,
        'n_assets': n_assets
    }


# Main app
def main():
    # Header
    st.markdown('<div class="main-header">📊 Portfolio Optimization Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Interactive Mixed-Integer Optimization with ML-Driven Heuristics</div>', unsafe_allow_html=True)

    # Sidebar
    st.sidebar.header("⚙️ Configuration")

    # Data parameters
    st.sidebar.subheader("Data Parameters")
    n_assets = st.sidebar.slider("Number of Assets", 5, 20, 10)
    n_days = st.sidebar.slider("Number of Days", 250, 2000, 1000, step=250)
    seed = st.sidebar.number_input("Random Seed", 1, 1000, 42)

    # Strategy selection
    st.sidebar.subheader("Optimization Strategy")
    strategy = st.sidebar.selectbox(
        "Select Strategy",
        ['Equal Weight', 'Max Sharpe', 'Min Variance', 'Concentrated']
    )

    # Strategy-specific parameters
    if strategy == 'Concentrated':
        max_assets = st.sidebar.slider("Max Assets", 3, n_assets, min(5, n_assets))
    else:
        max_assets = None

    # Generate button
    if st.sidebar.button("🚀 Optimize Portfolio", type="primary"):
        with st.spinner("Generating data and optimizing..."):
            # Generate data
            prices, returns = generate_synthetic_data(n_assets, n_days, seed)

            # Optimize
            weights, annual_returns, cov_matrix = optimize_portfolio(
                returns, strategy, max_assets
            )
            metrics = evaluate_portfolio(weights, annual_returns, cov_matrix)

            # Store in session state
            st.session_state['prices'] = prices
            st.session_state['returns'] = returns
            st.session_state['weights'] = weights
            st.session_state['metrics'] = metrics
            st.session_state['annual_returns'] = annual_returns
            st.session_state['cov_matrix'] = cov_matrix

        st.sidebar.success("✅ Optimization Complete!")

    # Main content
    if 'weights' in st.session_state:
        weights = st.session_state['weights']
        metrics = st.session_state['metrics']
        prices = st.session_state['prices']
        returns = st.session_state['returns']

        # Metrics
        st.header("📈 Portfolio Metrics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="Expected Annual Return",
                value=f"{metrics['return']:.2%}",
                delta=None
            )

        with col2:
            st.metric(
                label="Annual Volatility",
                value=f"{metrics['volatility']:.2%}",
                delta=None
            )

        with col3:
            st.metric(
                label="Sharpe Ratio",
                value=f"{metrics['sharpe']:.3f}",
                delta=None
            )

        with col4:
            st.metric(
                label="Number of Assets",
                value=f"{int(metrics['n_assets'])}",
                delta=None
            )

        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Weights", "📉 Prices", "🔗 Correlation", "📈 Performance"])

        with tab1:
            st.subheader("Portfolio Weights")

            col1, col2 = st.columns([2, 1])

            with col1:
                # Bar chart
                fig, ax = plt.subplots(figsize=(10, 6))
                active_weights = weights[weights > 1e-4].sort_values(ascending=True)
                colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(active_weights)))
                ax.barh(range(len(active_weights)), active_weights.values, color=colors, edgecolor='black')
                ax.set_yticks(range(len(active_weights)))
                ax.set_yticklabels(active_weights.index)
                ax.set_xlabel('Weight', fontsize=12, fontweight='bold')
                ax.set_title(f'{strategy} Portfolio Weights', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='x')

                for i, v in enumerate(active_weights.values):
                    ax.text(v + 0.005, i, f'{v:.1%}', va='center', fontsize=10)

                st.pyplot(fig)

            with col2:
                # Weights table
                st.dataframe(
                    weights[weights > 1e-4].sort_values(ascending=False).to_frame('Weight').style.format("{:.2%}"),
                    height=400
                )

        with tab2:
            st.subheader("Asset Prices Over Time")

            fig, ax = plt.subplots(figsize=(12, 6))
            for col in prices.columns[:5]:  # Show top 5
                ax.plot(prices.index, prices[col], label=col, linewidth=2, alpha=0.7)

            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Price', fontsize=12)
            ax.set_title('Price Evolution', fontsize=14, fontweight='bold')
            ax.legend(loc='upper left', fontsize=10)
            ax.grid(True, alpha=0.3)

            st.pyplot(fig)

        with tab3:
            st.subheader("Return Correlation Matrix")

            fig, ax = plt.subplots(figsize=(10, 8))
            corr_matrix = returns.corr()
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                       center=0, square=True, linewidths=0.5, ax=ax,
                       cbar_kws={'label': 'Correlation'})
            ax.set_title('Asset Correlation Heatmap', fontsize=14, fontweight='bold')

            st.pyplot(fig)

        with tab4:
            st.subheader("Portfolio Performance")

            # Calculate portfolio returns
            portfolio_returns = (returns * weights.values).sum(axis=1)
            cumulative_portfolio = (1 + portfolio_returns).cumprod()

            # Benchmark (equal weight)
            equal_weight = pd.Series(1.0 / len(returns.columns), index=returns.columns)
            benchmark_returns = (returns * equal_weight.values).sum(axis=1)
            cumulative_benchmark = (1 + benchmark_returns).cumprod()

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(cumulative_portfolio.index, cumulative_portfolio.values,
                   label=f'{strategy} Portfolio', linewidth=2.5, color='green')
            ax.plot(cumulative_benchmark.index, cumulative_benchmark.values,
                   label='Equal Weight Benchmark', linewidth=2, linestyle='--', color='gray')

            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Cumulative Return', fontsize=12)
            ax.set_title('Cumulative Performance', fontsize=14, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)

            st.pyplot(fig)

    else:
        st.info("👈 Configure parameters in the sidebar and click 'Optimize Portfolio' to begin!")

        # Show example
        st.header("📚 How It Works")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            ### 🎯 Optimization Strategies

            - **Equal Weight**: Naive 1/N allocation
            - **Max Sharpe**: Maximize risk-adjusted return
            - **Min Variance**: Minimize portfolio volatility
            - **Concentrated**: Cardinality-constrained optimization
            """)

        with col2:
            st.markdown("""
            ### 📊 Features

            - Interactive parameter tuning
            - Real-time optimization
            - Multiple visualizations
            - Performance comparison
            - Exportable results
            """)


if __name__ == '__main__':
    main()
