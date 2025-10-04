"""
Enhanced Streamlit Dashboard for Portfolio Optimization V2

Features:
- Support for both synthetic and real market data
- Multiple data sources (Tech, Finance, Healthcare, S&P 500)
- All optimization strategies
- Enhanced visualizations
- Performance comparison

Run with:
    streamlit run src/visualization/dashboard_v2.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Page configuration
st.set_page_config(
    page_title="Portfolio Optimizer V2",
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
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.25rem;
        padding: 0.75rem 1.25rem;
        margin-bottom: 1rem;
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


@st.cache_data
def fetch_real_data(data_source, n_assets, period):
    """Fetch real market data."""
    try:
        from src.data.real_data_loader import RealDataLoader

        loader = RealDataLoader()

        if data_source == "S&P 500 Random":
            prices, returns, tickers = loader.fetch_sp500_subset(n_stocks=n_assets, seed=42)
        else:
            # Map source to category
            category_map = {
                "Tech Giants": "tech",
                "Financial Sector": "finance",
                "Healthcare": "healthcare",
                "Energy Sector": "energy",
                "Diversified": "diversified"
            }
            category = category_map.get(data_source, "diversified")
            tickers = loader.get_popular_tickers(category)[:n_assets]
            prices, returns = loader.fetch_data(tickers, period=period)

        return prices, returns, True

    except Exception as e:
        st.error(f"Failed to fetch real data: {e}")
        st.info("Falling back to synthetic data...")
        return None, None, False


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

    elif strategy == 'Risk Parity':
        # Simple risk parity approximation
        inv_vol = 1.0 / annual_volatility.values
        weights = inv_vol / inv_vol.sum()

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
    st.markdown('<div class="main-header">📊 Portfolio Optimization Dashboard V2</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Mixed-Integer Optimization with Real & Synthetic Data</div>', unsafe_allow_html=True)

    # Sidebar
    st.sidebar.header("⚙️ Configuration")

    # Data source selection
    st.sidebar.subheader("📊 Data Source")
    data_mode = st.sidebar.radio(
        "Select Data Mode",
        ["Synthetic Data", "Real Market Data"],
        help="Choose between synthetic data (fast, reproducible) or real market data (actual stocks)"
    )

    if data_mode == "Synthetic Data":
        # Synthetic data parameters
        st.sidebar.subheader("Data Parameters")
        n_assets = st.sidebar.slider("Number of Assets", 5, 20, 10)
        n_days = st.sidebar.slider("Number of Days", 250, 2000, 1000, step=250)
        seed = st.sidebar.number_input("Random Seed", 1, 1000, 42)

    else:
        # Real data parameters
        st.sidebar.subheader("Market Selection")
        data_source = st.sidebar.selectbox(
            "Choose Market Sector",
            ["S&P 500 Random", "Tech Giants", "Financial Sector", "Healthcare", "Energy Sector", "Diversified"]
        )
        n_assets = st.sidebar.slider("Number of Stocks", 5, 10, 8)
        period = st.sidebar.selectbox(
            "Historical Period",
            ["1y", "2y", "5y"],
            index=1
        )

    # Strategy selection
    st.sidebar.subheader("Optimization Strategy")
    strategy = st.sidebar.selectbox(
        "Select Strategy",
        ['Equal Weight', 'Max Sharpe', 'Min Variance', 'Concentrated', 'Risk Parity']
    )

    # Strategy-specific parameters
    if strategy == 'Concentrated':
        if data_mode == "Synthetic Data":
            max_assets = st.sidebar.slider("Max Assets", 3, n_assets, min(5, n_assets))
        else:
            max_assets = st.sidebar.slider("Max Assets", 3, n_assets, min(5, n_assets))
    else:
        max_assets = None

    # Generate/Optimize button
    if st.sidebar.button("🚀 Optimize Portfolio", type="primary"):
        with st.spinner("Loading data and optimizing..."):
            # Generate or fetch data
            if data_mode == "Synthetic Data":
                prices, returns = generate_synthetic_data(n_assets, n_days, seed)
                data_success = True
                st.session_state['data_mode'] = 'synthetic'
            else:
                prices, returns, data_success = fetch_real_data(data_source, n_assets, period)
                if data_success:
                    st.session_state['data_mode'] = 'real'
                    st.session_state['data_source'] = data_source
                else:
                    # Fallback to synthetic
                    prices, returns = generate_synthetic_data(10, 500, 42)
                    st.session_state['data_mode'] = 'synthetic_fallback'

            if data_success or st.session_state.get('data_mode') == 'synthetic_fallback':
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
                st.session_state['strategy'] = strategy

        if data_success:
            st.sidebar.success("✅ Optimization Complete!")
        else:
            st.sidebar.warning("⚠️ Using fallback synthetic data")

    # Main content
    if 'weights' in st.session_state:
        weights = st.session_state['weights']
        metrics = st.session_state['metrics']
        prices = st.session_state['prices']
        returns = st.session_state['returns']
        strategy_name = st.session_state.get('strategy', 'Unknown')

        # Data mode indicator
        data_mode_display = st.session_state.get('data_mode', 'unknown')
        if data_mode_display == 'real':
            source = st.session_state.get('data_source', 'Unknown')
            st.info(f"📈 Using Real Market Data: {source} | Period: {prices.index[0].strftime('%Y-%m-%d')} to {prices.index[-1].strftime('%Y-%m-%d')}")
        elif data_mode_display == 'synthetic':
            st.info(f"🎲 Using Synthetic Data | Assets: {len(prices.columns)} | Days: {len(prices)}")

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
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Weights", "📉 Prices", "🔗 Correlation", "📈 Performance", "📋 Summary"])

        with tab1:
            st.subheader(f"{strategy_name} Portfolio Weights")

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
                ax.set_title(f'{strategy_name} Portfolio Weights', fontsize=14, fontweight='bold')
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
            # Show top 5 weighted assets
            top_assets = weights.nlargest(5).index
            for col in top_assets:
                ax.plot(prices.index, prices[col], label=col, linewidth=2, alpha=0.7)

            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Price', fontsize=12)
            ax.set_title('Price Evolution (Top 5 Holdings)', fontsize=14, fontweight='bold')
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
                   label=f'{strategy_name} Portfolio', linewidth=2.5, color='green')
            ax.plot(cumulative_benchmark.index, cumulative_benchmark.values,
                   label='Equal Weight Benchmark', linewidth=2, linestyle='--', color='gray')

            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Cumulative Return', fontsize=12)
            ax.set_title('Cumulative Performance', fontsize=14, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)

            st.pyplot(fig)

            # Performance statistics
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Portfolio Statistics:**")
                total_return = (cumulative_portfolio.iloc[-1] - 1) * 100
                max_dd = ((cumulative_portfolio / cumulative_portfolio.cummax()) - 1).min() * 100
                st.write(f"Total Return: {total_return:.2f}%")
                st.write(f"Max Drawdown: {max_dd:.2f}%")

            with col2:
                st.markdown("**Benchmark Statistics:**")
                bench_return = (cumulative_benchmark.iloc[-1] - 1) * 100
                bench_max_dd = ((cumulative_benchmark / cumulative_benchmark.cummax()) - 1).min() * 100
                st.write(f"Total Return: {bench_return:.2f}%")
                st.write(f"Max Drawdown: {bench_max_dd:.2f}%")

        with tab5:
            st.subheader("Portfolio Summary")

            # Data info
            st.markdown("### 📊 Data Information")
            info_df = pd.DataFrame({
                'Metric': ['Number of Assets', 'Time Period', 'Start Date', 'End Date', 'Trading Days'],
                'Value': [
                    len(prices.columns),
                    f"{len(prices)} days",
                    prices.index[0].strftime('%Y-%m-%d'),
                    prices.index[-1].strftime('%Y-%m-%d'),
                    len(prices)
                ]
            })
            st.table(info_df)

            # Portfolio characteristics
            st.markdown("### 💼 Portfolio Characteristics")
            char_df = pd.DataFrame({
                'Characteristic': ['Strategy', 'Active Assets', 'Concentration (HHI)', 'Diversification Ratio'],
                'Value': [
                    strategy_name,
                    int(metrics['n_assets']),
                    f"{(weights**2).sum():.4f}",
                    f"{metrics['volatility'] / (weights * returns.std() * np.sqrt(252)).sum():.4f}"
                ]
            })
            st.table(char_df)

            # Top holdings
            st.markdown("### 🏆 Top 5 Holdings")
            top_holdings = weights.nlargest(5).to_frame('Weight')
            top_holdings['Weight'] = top_holdings['Weight'].apply(lambda x: f"{x:.2%}")
            st.table(top_holdings)

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
            - **Risk Parity**: Equal risk contribution
            """)

        with col2:
            st.markdown("""
            ### 📊 Data Sources

            - **Synthetic**: Simulated data with factor model
            - **Real Market**: Actual stock prices from Yahoo Finance
              - S&P 500 stocks
              - Sector-specific portfolios
              - Historical data up to 5 years
            """)

        st.markdown("""
        ### 🚀 New Features in V2

        - ✅ **Real Market Data Integration** - Trade actual stocks!
        - ✅ **Multiple Sectors** - Tech, Finance, Healthcare, Energy
        - ✅ **Risk Parity Strategy** - Equal risk contribution
        - ✅ **Enhanced Visualizations** - More detailed charts
        - ✅ **Performance Statistics** - Drawdown, total return
        - ✅ **Portfolio Summary** - Complete characteristics
        """)


if __name__ == '__main__':
    main()
