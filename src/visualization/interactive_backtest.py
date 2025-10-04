"""
Interactive Backtesting Dashboard

Perform rolling window backtesting with:
- Configurable rebalancing frequency
- Transaction cost simulation
- Performance attribution
- Drawdown analysis
- Monte Carlo simulations

Run with:
    streamlit run src/visualization/interactive_backtest.py
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
    page_title="Backtesting Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


def perform_backtest(returns, strategy, rebalance_freq=21, transaction_cost=0.001, max_assets=None):
    """
    Perform rolling window backtest.

    Args:
        returns: DataFrame with returns
        strategy: Strategy name
        rebalance_freq: Rebalancing frequency in days
        transaction_cost: Transaction cost as percentage
        max_assets: Max assets for concentrated strategy

    Returns:
        Dictionary with backtest results
    """
    n_days = len(returns)
    lookback = 126  # 6 months lookback for optimization

    portfolio_values = [1.0]
    weights_history = []
    turnover_history = []
    dates_history = [returns.index[lookback]]

    current_weights = None

    for i in range(lookback, n_days):
        # Check if rebalancing day
        if (i - lookback) % rebalance_freq == 0 or current_weights is None:
            # Optimize on historical data
            hist_returns = returns.iloc[max(0, i-lookback):i]

            new_weights, _, _ = optimize_portfolio(hist_returns, strategy, max_assets=max_assets)

            if current_weights is not None:
                # Calculate turnover
                turnover = np.abs(new_weights.values - current_weights.values).sum()
                turnover_history.append(turnover)

                # Apply transaction costs
                cost = turnover * transaction_cost
                portfolio_values[-1] *= (1 - cost)
            else:
                turnover_history.append(0)

            current_weights = new_weights.values
            weights_history.append(new_weights.copy())

        # Calculate daily return
        daily_return = (returns.iloc[i].values * current_weights).sum()
        new_value = portfolio_values[-1] * (1 + daily_return)
        portfolio_values.append(new_value)
        dates_history.append(returns.index[i])

    # Calculate metrics
    portfolio_series = pd.Series(portfolio_values, index=dates_history)
    returns_series = portfolio_series.pct_change().dropna()

    total_return = (portfolio_values[-1] - 1) * 100
    annual_return = ((portfolio_values[-1] ** (252 / len(portfolio_values))) - 1) * 100
    annual_vol = returns_series.std() * np.sqrt(252) * 100
    sharpe = (annual_return / annual_vol) if annual_vol > 0 else 0

    # Drawdown
    running_max = portfolio_series.expanding().max()
    drawdown = (portfolio_series / running_max - 1) * 100
    max_drawdown = drawdown.min()

    # Average turnover
    avg_turnover = np.mean(turnover_history) if len(turnover_history) > 0 else 0

    results = {
        'portfolio_values': portfolio_series,
        'returns': returns_series,
        'drawdown': drawdown,
        'total_return': total_return,
        'annual_return': annual_return,
        'annual_volatility': annual_vol,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'avg_turnover': avg_turnover,
        'n_rebalances': len(weights_history),
        'weights_history': weights_history
    }

    return results


def create_equity_curve(results_dict):
    """Create interactive equity curve comparison."""
    fig = go.Figure()

    colors = px.colors.qualitative.Set2

    for idx, (strategy, results) in enumerate(results_dict.items()):
        fig.add_trace(go.Scatter(
            x=results['portfolio_values'].index,
            y=results['portfolio_values'].values,
            mode='lines',
            name=strategy,
            line=dict(width=2.5, color=colors[idx % len(colors)]),
            hovertemplate='<b>' + strategy + '</b><br>' +
                         'Date: %{x}<br>' +
                         'Value: $%{y:.2f}<br>' +
                         '<extra></extra>'
        ))

    fig.update_layout(
        title='Equity Curve (Backtested)',
        xaxis_title='Date',
        yaxis_title='Portfolio Value ($)',
        hovermode='x unified',
        height=500
    )

    return fig


def create_drawdown_chart(results_dict):
    """Create drawdown comparison chart."""
    fig = go.Figure()

    colors = px.colors.qualitative.Set2

    for idx, (strategy, results) in enumerate(results_dict.items()):
        fig.add_trace(go.Scatter(
            x=results['drawdown'].index,
            y=results['drawdown'].values,
            mode='lines',
            name=strategy,
            fill='tozeroy',
            line=dict(width=1.5, color=colors[idx % len(colors)]),
            hovertemplate='<b>' + strategy + '</b><br>' +
                         'Date: %{x}<br>' +
                         'Drawdown: %{y:.2f}%<br>' +
                         '<extra></extra>'
        ))

    fig.update_layout(
        title='Underwater Plot (Drawdown Analysis)',
        xaxis_title='Date',
        yaxis_title='Drawdown (%)',
        hovermode='x unified',
        height=400
    )

    return fig


def create_returns_distribution(results_dict):
    """Create returns distribution histogram."""
    fig = go.Figure()

    colors = px.colors.qualitative.Set2

    for idx, (strategy, results) in enumerate(results_dict.items()):
        returns_pct = results['returns'] * 100

        fig.add_trace(go.Histogram(
            x=returns_pct.values,
            name=strategy,
            opacity=0.7,
            marker_color=colors[idx % len(colors)],
            nbinsx=50
        ))

    fig.update_layout(
        title='Daily Returns Distribution',
        xaxis_title='Daily Return (%)',
        yaxis_title='Frequency',
        barmode='overlay',
        height=400
    )

    return fig


def create_rolling_metrics(results, window=63):
    """Create rolling Sharpe ratio chart."""
    returns = results['returns']

    rolling_sharpe = (returns.rolling(window).mean() / returns.rolling(window).std()) * np.sqrt(252)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=rolling_sharpe.index,
        y=rolling_sharpe.values,
        mode='lines',
        fill='tozeroy',
        line=dict(width=2, color='green'),
        hovertemplate='Date: %{x}<br>Sharpe: %{y:.2f}<extra></extra>'
    ))

    fig.add_hline(y=0, line_dash="dash", line_color="red")
    fig.add_hline(y=1, line_dash="dash", line_color="orange", annotation_text="Sharpe = 1.0")

    fig.update_layout(
        title=f'Rolling Sharpe Ratio ({window} days)',
        xaxis_title='Date',
        yaxis_title='Sharpe Ratio',
        height=400
    )

    return fig


def main():
    st.title("📈 Interactive Backtesting Dashboard")
    st.markdown("### Test Portfolio Strategies with Rolling Window Backtesting")

    # Sidebar
    st.sidebar.header("⚙️ Configuration")

    # Data parameters
    st.sidebar.subheader("📊 Data Parameters")
    n_assets = st.sidebar.slider("Number of Assets", 5, 20, 10)
    n_days = st.sidebar.slider("Number of Days", 500, 2000, 1000, step=100)
    seed = st.sidebar.number_input("Random Seed", 1, 1000, 42)

    # Backtest parameters
    st.sidebar.subheader("🔄 Backtest Parameters")
    rebalance_freq = st.sidebar.select_slider(
        "Rebalancing Frequency",
        options=[1, 5, 10, 21, 42, 63],
        value=21,
        format_func=lambda x: f"{x} days (~{x//21} months)" if x >= 21 else f"{x} days"
    )

    transaction_cost = st.sidebar.slider(
        "Transaction Cost (%)",
        0.0, 1.0, 0.1,
        step=0.05,
        help="Cost as percentage of turnover"
    ) / 100

    # Strategy selection
    st.sidebar.subheader("🎯 Strategies to Test")
    test_equal_weight = st.sidebar.checkbox("Equal Weight", value=True)
    test_max_sharpe = st.sidebar.checkbox("Max Sharpe", value=True)
    test_min_variance = st.sidebar.checkbox("Min Variance", value=True)
    test_concentrated = st.sidebar.checkbox("Concentrated", value=False)
    test_risk_parity = st.sidebar.checkbox("Risk Parity", value=False)

    max_assets = st.sidebar.slider("Max Assets (Concentrated)", 3, n_assets, 5) if test_concentrated else None

    # Run backtest
    if st.sidebar.button("🚀 Run Backtest", type="primary"):
        with st.spinner("Running backtests..."):
            # Generate data
            prices, returns = generate_synthetic_data(n_assets, n_days, seed)

            # Run backtests
            strategies_to_test = []
            if test_equal_weight:
                strategies_to_test.append(('Equal Weight', None))
            if test_max_sharpe:
                strategies_to_test.append(('Max Sharpe', None))
            if test_min_variance:
                strategies_to_test.append(('Min Variance', None))
            if test_concentrated:
                strategies_to_test.append(('Concentrated', max_assets))
            if test_risk_parity:
                strategies_to_test.append(('Risk Parity', None))

            results_dict = {}
            for strategy_name, param in strategies_to_test:
                results = perform_backtest(
                    returns,
                    strategy_name,
                    rebalance_freq=rebalance_freq,
                    transaction_cost=transaction_cost,
                    max_assets=param
                )
                results_dict[strategy_name] = results

            # Store results
            st.session_state['backtest_results'] = results_dict
            st.session_state['backtest_params'] = {
                'rebalance_freq': rebalance_freq,
                'transaction_cost': transaction_cost * 100,
                'n_assets': n_assets,
                'n_days': n_days
            }

        st.sidebar.success("✅ Backtest Complete!")

    # Main content
    if 'backtest_results' in st.session_state:
        results_dict = st.session_state['backtest_results']
        params = st.session_state['backtest_params']

        # Summary metrics
        st.header("📊 Backtest Summary")

        st.info(f"""
        **Backtest Parameters:**
        - Rebalancing: Every {params['rebalance_freq']} days
        - Transaction Cost: {params['transaction_cost']:.2f}%
        - Assets: {params['n_assets']}
        - Time Period: {params['n_days']} days
        """)

        # Create metrics table
        metrics_data = []
        for strategy, results in results_dict.items():
            metrics_data.append({
                'Strategy': strategy,
                'Total Return': f"{results['total_return']:.2f}%",
                'Annual Return': f"{results['annual_return']:.2f}%",
                'Annual Vol': f"{results['annual_volatility']:.2f}%",
                'Sharpe Ratio': f"{results['sharpe_ratio']:.3f}",
                'Max Drawdown': f"{results['max_drawdown']:.2f}%",
                'Avg Turnover': f"{results['avg_turnover']:.2%}",
                'Rebalances': results['n_rebalances']
            })

        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(
            metrics_df.style.background_gradient(subset=['Sharpe Ratio'], cmap='RdYlGn'),
            use_container_width=True
        )

        # Find winners
        col1, col2, col3, col4 = st.columns(4)

        sharpe_values = {s: r['sharpe_ratio'] for s, r in results_dict.items()}
        best_sharpe = max(sharpe_values, key=sharpe_values.get)

        with col1:
            st.metric("🏆 Best Sharpe", best_sharpe, f"{sharpe_values[best_sharpe]:.3f}")

        return_values = {s: r['total_return'] for s, r in results_dict.items()}
        best_return = max(return_values, key=return_values.get)

        with col2:
            st.metric("📈 Highest Return", best_return, f"{return_values[best_return]:.2f}%")

        dd_values = {s: r['max_drawdown'] for s, r in results_dict.items()}
        lowest_dd = max(dd_values, key=dd_values.get)  # Max because drawdowns are negative

        with col3:
            st.metric("🛡️ Smallest Drawdown", lowest_dd, f"{dd_values[lowest_dd]:.2f}%")

        turnover_values = {s: r['avg_turnover'] for s, r in results_dict.items()}
        lowest_turnover = min(turnover_values, key=turnover_values.get)

        with col4:
            st.metric("💰 Lowest Turnover", lowest_turnover, f"{turnover_values[lowest_turnover]:.2%}")

        # Visualization tabs
        st.header("📊 Interactive Analysis")

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Equity Curve", "Drawdown", "Returns Dist", "Rolling Metrics", "Detailed Analysis"
        ])

        with tab1:
            st.subheader("Equity Curve Comparison")
            fig_equity = create_equity_curve(results_dict)
            st.plotly_chart(fig_equity, use_container_width=True)

            st.info("💡 **Tip:** Higher and smoother curves indicate better performance")

        with tab2:
            st.subheader("Drawdown Analysis")
            fig_dd = create_drawdown_chart(results_dict)
            st.plotly_chart(fig_dd, use_container_width=True)

            st.info("💡 **Tip:** Shallower drawdowns indicate better risk management")

        with tab3:
            st.subheader("Returns Distribution")
            fig_dist = create_returns_distribution(results_dict)
            st.plotly_chart(fig_dist, use_container_width=True)

            # Statistics
            st.markdown("### Distribution Statistics")
            dist_data = []
            for strategy, results in results_dict.items():
                rets = results['returns'] * 100
                dist_data.append({
                    'Strategy': strategy,
                    'Mean': f"{rets.mean():.3f}%",
                    'Median': f"{rets.median():.3f}%",
                    'Std Dev': f"{rets.std():.3f}%",
                    'Skewness': f"{rets.skew():.3f}",
                    'Kurtosis': f"{rets.kurtosis():.3f}"
                })

            dist_df = pd.DataFrame(dist_data)
            st.dataframe(dist_df, use_container_width=True)

        with tab4:
            st.subheader("Rolling Performance Metrics")

            # Strategy selector
            selected_strategy = st.selectbox(
                "Select Strategy",
                list(results_dict.keys())
            )

            results = results_dict[selected_strategy]

            # Rolling Sharpe
            fig_rolling = create_rolling_metrics(results, window=63)
            st.plotly_chart(fig_rolling, use_container_width=True)

            # Rolling volatility
            rolling_vol = results['returns'].rolling(63).std() * np.sqrt(252) * 100

            fig_vol = go.Figure()
            fig_vol.add_trace(go.Scatter(
                x=rolling_vol.index,
                y=rolling_vol.values,
                mode='lines',
                fill='tozeroy',
                line=dict(width=2, color='red')
            ))

            fig_vol.update_layout(
                title='Rolling Volatility (63 days)',
                xaxis_title='Date',
                yaxis_title='Annualized Volatility (%)',
                height=400
            )

            st.plotly_chart(fig_vol, use_container_width=True)

        with tab5:
            st.subheader("Detailed Strategy Analysis")

            selected_strategy = st.selectbox(
                "Select Strategy for Details",
                list(results_dict.keys()),
                key='detailed_select'
            )

            results = results_dict[selected_strategy]

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Performance Metrics")
                st.write(f"**Total Return:** {results['total_return']:.2f}%")
                st.write(f"**Annualized Return:** {results['annual_return']:.2f}%")
                st.write(f"**Annualized Volatility:** {results['annual_volatility']:.2f}%")
                st.write(f"**Sharpe Ratio:** {results['sharpe_ratio']:.3f}")
                st.write(f"**Max Drawdown:** {results['max_drawdown']:.2f}%")

            with col2:
                st.markdown("### Trading Activity")
                st.write(f"**Number of Rebalances:** {results['n_rebalances']}")
                st.write(f"**Average Turnover:** {results['avg_turnover']:.2%}")
                st.write(f"**Est. Total Costs:** {results['avg_turnover'] * results['n_rebalances'] * transaction_cost * 100:.2f}%")

                final_value = results['portfolio_values'].iloc[-1]
                st.write(f"**Final Portfolio Value:** ${final_value:.2f}")

            # Monthly returns heatmap
            st.markdown("### Monthly Returns Heatmap")
            monthly_returns = results['returns'].resample('M').apply(lambda x: (1 + x).prod() - 1) * 100

            # Create year-month pivot
            monthly_returns.index = pd.to_datetime(monthly_returns.index)
            monthly_df = monthly_returns.to_frame('Return')
            monthly_df['Year'] = monthly_df.index.year
            monthly_df['Month'] = monthly_df.index.month_name()

            pivot = monthly_df.pivot_table(values='Return', index='Year', columns='Month')

            # Reorder months
            month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                          'July', 'August', 'September', 'October', 'November', 'December']
            pivot = pivot[[m for m in month_order if m in pivot.columns]]

            fig_heatmap = go.Figure(data=go.Heatmap(
                z=pivot.values,
                x=pivot.columns,
                y=pivot.index,
                colorscale='RdYlGn',
                zmid=0,
                text=pivot.values,
                texttemplate='%{text:.1f}%',
                textfont={"size": 10}
            ))

            fig_heatmap.update_layout(
                title='Monthly Returns (%)',
                xaxis_title='Month',
                yaxis_title='Year',
                height=300
            )

            st.plotly_chart(fig_heatmap, use_container_width=True)

    else:
        # Welcome screen
        st.info("👈 Configure backtest parameters and click 'Run Backtest' to begin!")

        st.header("📚 About Backtesting")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            ### 🎯 What is Backtesting?

            Backtesting simulates how a trading strategy would have performed
            in the past using historical data.

            **Key Features:**
            - **Rolling Window**: Optimize on historical data only
            - **Realistic Trading**: Include transaction costs
            - **Rebalancing**: Periodic portfolio adjustments
            - **Out-of-Sample**: Test on unseen future data
            """)

        with col2:
            st.markdown("""
            ### 📊 Metrics Explained

            - **Total Return**: Cumulative profit/loss
            - **Sharpe Ratio**: Risk-adjusted return
            - **Max Drawdown**: Largest peak-to-trough decline
            - **Turnover**: Portfolio changes per rebalance
            - **Transaction Costs**: Fees from trading

            **Higher Sharpe and lower drawdown = better strategy**
            """)


if __name__ == '__main__':
    main()
