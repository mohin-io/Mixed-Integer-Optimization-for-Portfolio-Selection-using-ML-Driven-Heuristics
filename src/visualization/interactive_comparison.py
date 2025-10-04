"""
Interactive Multi-Strategy Comparison Dashboard

Compare multiple portfolio optimization strategies side-by-side with:
- Live strategy comparison
- Interactive parameter adjustment
- Performance metrics comparison
- Risk-return scatter plots
- Correlation analysis

Run with:
    streamlit run src/visualization/interactive_comparison.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.visualization.dashboard_v2 import generate_synthetic_data, optimize_portfolio, evaluate_portfolio

# Page configuration
st.set_page_config(
    page_title="Strategy Comparison Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


def calculate_all_strategies(returns, max_assets=5):
    """
    Calculate all strategies and return results.

    Args:
        returns: DataFrame with returns
        max_assets: Max assets for concentrated strategy

    Returns:
        Dictionary with all strategy results
    """
    strategies = {
        'Equal Weight': None,
        'Max Sharpe': None,
        'Min Variance': None,
        'Concentrated': max_assets,
        'Risk Parity': None
    }

    results = {}

    for strategy_name, param in strategies.items():
        weights, annual_returns, cov_matrix = optimize_portfolio(
            returns, strategy_name, max_assets=param
        )
        metrics = evaluate_portfolio(weights, annual_returns, cov_matrix)

        results[strategy_name] = {
            'weights': weights,
            'metrics': metrics,
            'annual_returns': annual_returns,
            'cov_matrix': cov_matrix
        }

    return results


def create_comparison_table(results):
    """Create comparison table for all strategies."""
    data = []

    for strategy, res in results.items():
        m = res['metrics']
        data.append({
            'Strategy': strategy,
            'Return': f"{m['return']:.2%}",
            'Volatility': f"{m['volatility']:.2%}",
            'Sharpe': f"{m['sharpe']:.3f}",
            'Active Assets': int(m['n_assets'])
        })

    return pd.DataFrame(data)


def create_risk_return_scatter(results):
    """Create interactive risk-return scatter plot."""
    fig = go.Figure()

    colors = px.colors.qualitative.Set2

    for idx, (strategy, res) in enumerate(results.items()):
        m = res['metrics']

        fig.add_trace(go.Scatter(
            x=[m['volatility']],
            y=[m['return']],
            mode='markers+text',
            name=strategy,
            marker=dict(size=20, color=colors[idx % len(colors)]),
            text=[strategy],
            textposition='top center',
            hovertemplate=f'<b>{strategy}</b><br>' +
                         f'Return: {m["return"]:.2%}<br>' +
                         f'Volatility: {m["volatility"]:.2%}<br>' +
                         f'Sharpe: {m["sharpe"]:.3f}<br>' +
                         '<extra></extra>'
        ))

    fig.update_layout(
        title='Risk-Return Profile Comparison',
        xaxis_title='Annual Volatility',
        yaxis_title='Expected Annual Return',
        hovermode='closest',
        height=500,
        showlegend=True
    )

    return fig


def create_weights_comparison(results):
    """Create stacked bar chart comparing weights across strategies."""
    # Get all unique assets
    all_assets = set()
    for res in results.values():
        all_assets.update(res['weights'][res['weights'] > 1e-4].index)

    all_assets = sorted(list(all_assets))

    fig = go.Figure()

    for asset in all_assets:
        weights_by_strategy = []
        for strategy in results.keys():
            w = results[strategy]['weights']
            weights_by_strategy.append(w.get(asset, 0) * 100)

        fig.add_trace(go.Bar(
            name=asset,
            x=list(results.keys()),
            y=weights_by_strategy,
            hovertemplate=f'<b>{asset}</b><br>' +
                         'Weight: %{y:.1f}%<br>' +
                         '<extra></extra>'
        ))

    fig.update_layout(
        title='Portfolio Weights Comparison',
        xaxis_title='Strategy',
        yaxis_title='Weight (%)',
        barmode='stack',
        height=500,
        hovermode='x unified'
    )

    return fig


def create_performance_comparison(returns, results):
    """Create cumulative performance comparison."""
    fig = go.Figure()

    colors = px.colors.qualitative.Set2

    for idx, (strategy, res) in enumerate(results.items()):
        weights = res['weights']
        portfolio_returns = (returns * weights.values).sum(axis=1)
        cumulative = (1 + portfolio_returns).cumprod()

        fig.add_trace(go.Scatter(
            x=cumulative.index,
            y=cumulative.values,
            mode='lines',
            name=strategy,
            line=dict(width=2.5, color=colors[idx % len(colors)]),
            hovertemplate='<b>' + strategy + '</b><br>' +
                         'Date: %{x}<br>' +
                         'Value: %{y:.3f}<br>' +
                         '<extra></extra>'
        ))

    fig.update_layout(
        title='Cumulative Performance Comparison',
        xaxis_title='Date',
        yaxis_title='Cumulative Return',
        hovermode='x unified',
        height=500
    )

    return fig


def create_sharpe_radar(results):
    """Create radar chart for strategy comparison."""
    categories = ['Return', 'Sharpe Ratio', 'Diversification', 'Concentration', 'Stability']

    fig = go.Figure()

    colors = px.colors.qualitative.Set2

    for idx, (strategy, res) in enumerate(results.items()):
        m = res['metrics']
        weights = res['weights']

        # Normalize metrics to 0-1 scale
        values = [
            min(m['return'] / 0.3, 1.0),  # Normalize return (30% max)
            min(m['sharpe'] / 4.0, 1.0),   # Normalize Sharpe (4.0 max)
            min(m['n_assets'] / 10.0, 1.0),  # Normalize diversification
            1.0 - (weights**2).sum(),  # Inverse HHI (more diverse = higher)
            min(1.0 / (m['volatility'] + 0.01), 1.0)  # Inverse volatility
        ]

        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],  # Close the polygon
            theta=categories + [categories[0]],
            fill='toself',
            name=strategy,
            line=dict(color=colors[idx % len(colors)])
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])
        ),
        title='Multi-Dimensional Strategy Comparison',
        height=500,
        showlegend=True
    )

    return fig


def main():
    # Header
    st.markdown('<div class="main-header">📊 Interactive Strategy Comparison Dashboard</div>', unsafe_allow_html=True)
    st.markdown("### Compare Multiple Portfolio Optimization Strategies Side-by-Side")

    # Sidebar
    st.sidebar.header("⚙️ Configuration")

    # Data parameters
    st.sidebar.subheader("📊 Data Parameters")
    n_assets = st.sidebar.slider("Number of Assets", 5, 20, 10, key='n_assets')
    n_days = st.sidebar.slider("Number of Days", 250, 2000, 500, step=250, key='n_days')
    seed = st.sidebar.number_input("Random Seed", 1, 1000, 42, key='seed')

    # Strategy parameters
    st.sidebar.subheader("🎯 Strategy Parameters")
    max_assets_concentrated = st.sidebar.slider(
        "Max Assets (Concentrated)",
        3, n_assets, min(5, n_assets),
        key='max_assets'
    )

    # Compare button
    if st.sidebar.button("🚀 Compare All Strategies", type="primary"):
        with st.spinner("Optimizing all strategies..."):
            # Generate data
            prices, returns = generate_synthetic_data(n_assets, n_days, seed)

            # Calculate all strategies
            results = calculate_all_strategies(returns, max_assets_concentrated)

            # Store in session state
            st.session_state['comparison_results'] = results
            st.session_state['comparison_prices'] = prices
            st.session_state['comparison_returns'] = returns

        st.sidebar.success("✅ Comparison Complete!")

    # Main content
    if 'comparison_results' in st.session_state:
        results = st.session_state['comparison_results']
        returns = st.session_state['comparison_returns']
        prices = st.session_state['comparison_prices']

        # Comparison table
        st.header("📋 Strategy Metrics Comparison")
        comparison_df = create_comparison_table(results)

        # Style the dataframe
        st.dataframe(
            comparison_df.style.background_gradient(subset=['Sharpe'], cmap='RdYlGn'),
            use_container_width=True,
            height=250
        )

        # Find best strategies
        sharpe_values = {s: r['metrics']['sharpe'] for s, r in results.items()}
        best_sharpe = max(sharpe_values, key=sharpe_values.get)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🏆 Best Sharpe Ratio", best_sharpe, f"{sharpe_values[best_sharpe]:.3f}")
        with col2:
            return_values = {s: r['metrics']['return'] for s, r in results.items()}
            best_return = max(return_values, key=return_values.get)
            st.metric("📈 Highest Return", best_return, f"{return_values[best_return]:.2%}")
        with col3:
            vol_values = {s: r['metrics']['volatility'] for s, r in results.items()}
            lowest_vol = min(vol_values, key=vol_values.get)
            st.metric("🛡️ Lowest Volatility", lowest_vol, f"{vol_values[lowest_vol]:.2%}")

        # Interactive visualizations
        st.header("📊 Interactive Visualizations")

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Risk-Return", "Weights", "Performance", "Radar Chart", "Detailed Analysis"
        ])

        with tab1:
            st.subheader("Risk-Return Scatter Plot")
            fig_scatter = create_risk_return_scatter(results)
            st.plotly_chart(fig_scatter, use_container_width=True)

            st.info("💡 **Tip:** Strategies in the upper-left quadrant offer better risk-adjusted returns (higher return, lower risk)")

        with tab2:
            st.subheader("Portfolio Weights Distribution")
            fig_weights = create_weights_comparison(results)
            st.plotly_chart(fig_weights, use_container_width=True)

            st.info("💡 **Tip:** Hover over bars to see exact weight allocations for each asset")

        with tab3:
            st.subheader("Cumulative Performance Over Time")
            fig_performance = create_performance_comparison(returns, results)
            st.plotly_chart(fig_performance, use_container_width=True)

            # Performance statistics
            st.subheader("Performance Statistics")
            perf_data = []
            for strategy, res in results.items():
                weights = res['weights']
                portfolio_returns = (returns * weights.values).sum(axis=1)
                cumulative = (1 + portfolio_returns).cumprod()

                total_return = (cumulative.iloc[-1] - 1) * 100
                max_dd = ((cumulative / cumulative.cummax()) - 1).min() * 100

                perf_data.append({
                    'Strategy': strategy,
                    'Total Return': f"{total_return:.2f}%",
                    'Max Drawdown': f"{max_dd:.2f}%",
                    'Final Value': f"${cumulative.iloc[-1]:.2f}"
                })

            perf_df = pd.DataFrame(perf_data)
            st.dataframe(perf_df, use_container_width=True)

        with tab4:
            st.subheader("Multi-Dimensional Comparison")
            fig_radar = create_sharpe_radar(results)
            st.plotly_chart(fig_radar, use_container_width=True)

            st.markdown("""
            **Dimensions Explained:**
            - **Return**: Expected annual return (higher is better)
            - **Sharpe Ratio**: Risk-adjusted return (higher is better)
            - **Diversification**: Number of active assets (higher is better)
            - **Concentration**: Inverse of Herfindahl index (higher = less concentrated)
            - **Stability**: Inverse of volatility (higher is better)
            """)

        with tab5:
            st.subheader("Detailed Strategy Analysis")

            # Strategy selector
            selected_strategy = st.selectbox(
                "Select Strategy for Detailed View",
                list(results.keys())
            )

            res = results[selected_strategy]
            weights = res['weights']
            metrics = res['metrics']

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Portfolio Characteristics")
                st.write(f"**Expected Return:** {metrics['return']:.2%}")
                st.write(f"**Volatility:** {metrics['volatility']:.2%}")
                st.write(f"**Sharpe Ratio:** {metrics['sharpe']:.3f}")
                st.write(f"**Active Assets:** {int(metrics['n_assets'])}")
                st.write(f"**HHI (Concentration):** {(weights**2).sum():.4f}")

            with col2:
                st.markdown("### Top 5 Holdings")
                top_5 = weights.nlargest(5).to_frame('Weight')
                top_5['Weight %'] = (top_5['Weight'] * 100).round(2)
                st.dataframe(top_5[['Weight %']], use_container_width=True)

            # Individual weight distribution
            st.markdown("### Weight Distribution")
            active_weights = weights[weights > 1e-4].sort_values(ascending=False)

            fig_ind = go.Figure(go.Bar(
                x=active_weights.index,
                y=active_weights.values * 100,
                marker_color='indianred',
                hovertemplate='<b>%{x}</b><br>Weight: %{y:.2f}%<extra></extra>'
            ))

            fig_ind.update_layout(
                title=f'{selected_strategy} - Portfolio Weights',
                xaxis_title='Asset',
                yaxis_title='Weight (%)',
                height=400
            )

            st.plotly_chart(fig_ind, use_container_width=True)

    else:
        # Welcome screen
        st.info("👈 Configure parameters in the sidebar and click 'Compare All Strategies' to begin!")

        st.header("📚 About This Dashboard")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            ### 🎯 Features

            - **5 Optimization Strategies**
              - Equal Weight
              - Max Sharpe Ratio
              - Min Variance
              - Concentrated
              - Risk Parity

            - **Interactive Visualizations**
              - Risk-return scatter plots
              - Weight distribution comparison
              - Cumulative performance tracking
              - Multi-dimensional radar charts
            """)

        with col2:
            st.markdown("""
            ### 📊 Comparison Metrics

            - Expected annual return
            - Annual volatility (risk)
            - Sharpe ratio (risk-adjusted return)
            - Number of active assets
            - Portfolio concentration (HHI)
            - Maximum drawdown
            - Total cumulative return

            **All visualizations are interactive** - hover, zoom, and explore!
            """)

        st.markdown("""
        ---
        ### 🚀 Quick Start Guide

        1. **Set Data Parameters** - Choose number of assets, days, and random seed
        2. **Configure Strategies** - Set max assets for concentrated portfolio
        3. **Click Compare** - Run optimization for all 5 strategies simultaneously
        4. **Explore Results** - Use interactive tabs to analyze and compare strategies
        5. **Deep Dive** - Select individual strategies for detailed analysis
        """)


if __name__ == '__main__':
    main()
