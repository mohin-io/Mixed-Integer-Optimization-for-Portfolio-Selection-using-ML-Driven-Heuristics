"""
Interactive Risk Analytics Dashboard

Comprehensive risk analysis with:
- Value at Risk (VaR) and CVaR
- Risk decomposition
- Factor exposure analysis
- Stress testing
- Monte Carlo simulation

Run with:
    streamlit run src/visualization/interactive_risk.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.visualization.dashboard_v2 import generate_synthetic_data, optimize_portfolio, evaluate_portfolio

# Page configuration
st.set_page_config(
    page_title="Risk Analytics Dashboard",
    page_icon="⚠️",
    layout="wide",
    initial_sidebar_state="expanded"
)


def calculate_var(returns, confidence=0.95, method='historical'):
    """
    Calculate Value at Risk.

    Args:
        returns: Series of returns
        confidence: Confidence level (0.95 = 95%)
        method: 'historical', 'parametric', or 'monte_carlo'

    Returns:
        VaR value (negative number)
    """
    if method == 'historical':
        var = np.percentile(returns, (1 - confidence) * 100)
    elif method == 'parametric':
        mean = returns.mean()
        std = returns.std()
        var = mean - stats.norm.ppf(confidence) * std
    else:  # monte_carlo
        simulated = np.random.normal(returns.mean(), returns.std(), 10000)
        var = np.percentile(simulated, (1 - confidence) * 100)

    return var


def calculate_cvar(returns, confidence=0.95):
    """Calculate Conditional Value at Risk (Expected Shortfall)."""
    var = calculate_var(returns, confidence, 'historical')
    cvar = returns[returns <= var].mean()
    return cvar


def monte_carlo_simulation(weights, annual_returns, cov_matrix, n_sims=1000, n_days=252):
    """
    Perform Monte Carlo simulation for portfolio returns.

    Args:
        weights: Portfolio weights
        annual_returns: Expected annual returns
        cov_matrix: Covariance matrix
        n_sims: Number of simulations
        n_days: Number of days to simulate

    Returns:
        Simulated paths
    """
    daily_returns = annual_returns / 252
    daily_cov = cov_matrix / 252

    simulations = np.zeros((n_sims, n_days))

    for i in range(n_sims):
        # Generate correlated random returns
        random_returns = np.random.multivariate_normal(
            daily_returns.values,
            daily_cov.values,
            n_days
        )

        # Calculate portfolio returns
        port_returns = (random_returns * weights.values).sum(axis=1)

        # Cumulative returns
        simulations[i] = np.cumprod(1 + port_returns)

    return simulations


def stress_test(weights, returns, scenarios):
    """
    Perform stress testing under different scenarios.

    Args:
        weights: Portfolio weights
        returns: Historical returns
        scenarios: Dictionary of scenario definitions

    Returns:
        Dictionary of stress test results
    """
    results = {}

    for scenario_name, shock in scenarios.items():
        shocked_returns = returns.copy()

        # Apply shocks
        for asset, shock_pct in shock.items():
            if asset in shocked_returns.columns:
                shocked_returns[asset] *= (1 + shock_pct)

        # Calculate portfolio impact
        port_returns = (shocked_returns * weights.values).sum(axis=1)
        cumulative = (1 + port_returns).cumprod()

        results[scenario_name] = {
            'total_return': (cumulative.iloc[-1] - 1) * 100,
            'max_drawdown': ((cumulative / cumulative.cummax()) - 1).min() * 100
        }

    return results


def create_var_chart(returns_dict, confidence=0.95):
    """Create VaR comparison chart."""
    fig = go.Figure()

    var_data = []
    cvar_data = []
    strategies = list(returns_dict.keys())

    for strategy, returns in returns_dict.items():
        var = calculate_var(returns, confidence) * 100
        cvar = calculate_cvar(returns, confidence) * 100

        var_data.append(var)
        cvar_data.append(cvar)

    # VaR bars
    fig.add_trace(go.Bar(
        name='VaR (95%)',
        x=strategies,
        y=var_data,
        marker_color='indianred',
        text=[f'{v:.2f}%' for v in var_data],
        textposition='auto'
    ))

    # CVaR bars
    fig.add_trace(go.Bar(
        name='CVaR (95%)',
        x=strategies,
        y=cvar_data,
        marker_color='darkred',
        text=[f'{v:.2f}%' for v in cvar_data],
        textposition='auto'
    ))

    fig.update_layout(
        title='Value at Risk (VaR) and Conditional VaR',
        xaxis_title='Strategy',
        yaxis_title='Loss (%)',
        barmode='group',
        height=400
    )

    return fig


def create_monte_carlo_fan(simulations, dates):
    """Create Monte Carlo simulation fan chart."""
    fig = go.Figure()

    # Calculate percentiles
    percentiles = [5, 25, 50, 75, 95]
    percentile_values = np.percentile(simulations, percentiles, axis=0)

    # Add percentile bands
    fig.add_trace(go.Scatter(
        x=dates, y=percentile_values[0],
        mode='lines',
        name='5th Percentile',
        line=dict(width=0.5, color='rgba(255, 0, 0, 0.3)'),
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=dates, y=percentile_values[4],
        mode='lines',
        name='95th Percentile',
        line=dict(width=0.5, color='rgba(0, 255, 0, 0.3)'),
        fill='tonexty',
        fillcolor='rgba(100, 100, 100, 0.2)',
        showlegend=True
    ))

    # Median
    fig.add_trace(go.Scatter(
        x=dates, y=percentile_values[2],
        mode='lines',
        name='Median',
        line=dict(width=3, color='blue')
    ))

    # Sample paths
    for i in range(min(20, len(simulations))):
        fig.add_trace(go.Scatter(
            x=dates, y=simulations[i],
            mode='lines',
            line=dict(width=0.5, color='gray'),
            opacity=0.3,
            showlegend=False,
            hoverinfo='skip'
        ))

    fig.update_layout(
        title='Monte Carlo Simulation (1000 paths)',
        xaxis_title='Days',
        yaxis_title='Portfolio Value',
        height=500
    )

    return fig


def main():
    st.title("⚠️ Interactive Risk Analytics Dashboard")
    st.markdown("### Comprehensive Risk Analysis and Stress Testing")

    # Sidebar
    st.sidebar.header("⚙️ Configuration")

    # Data parameters
    st.sidebar.subheader("📊 Data Parameters")
    n_assets = st.sidebar.slider("Number of Assets", 5, 20, 10)
    n_days = st.sidebar.slider("Number of Days", 250, 1000, 500)
    seed = st.sidebar.number_input("Random Seed", 1, 1000, 42)

    # Strategy selection
    st.sidebar.subheader("🎯 Strategy")
    strategy = st.sidebar.selectbox(
        "Select Strategy for Analysis",
        ['Equal Weight', 'Max Sharpe', 'Min Variance', 'Concentrated', 'Risk Parity']
    )

    max_assets = st.sidebar.slider("Max Assets (Concentrated)", 3, n_assets, 5) if strategy == 'Concentrated' else None

    # Risk parameters
    st.sidebar.subheader("⚠️ Risk Parameters")
    var_confidence = st.sidebar.slider("VaR Confidence Level", 0.90, 0.99, 0.95, 0.01)
    n_monte_carlo = st.sidebar.slider("Monte Carlo Simulations", 100, 5000, 1000, 100)

    # Analyze button
    if st.sidebar.button("🚀 Analyze Risk", type="primary"):
        with st.spinner("Performing risk analysis..."):
            # Generate data
            prices, returns = generate_synthetic_data(n_assets, n_days, seed)

            # Optimize
            weights, annual_returns, cov_matrix = optimize_portfolio(
                returns, strategy, max_assets=max_assets
            )
            metrics = evaluate_portfolio(weights, annual_returns, cov_matrix)

            # Calculate portfolio returns
            portfolio_returns = (returns * weights.values).sum(axis=1)

            # VaR calculations
            var_hist = calculate_var(portfolio_returns, var_confidence, 'historical')
            var_param = calculate_var(portfolio_returns, var_confidence, 'parametric')
            cvar = calculate_cvar(portfolio_returns, var_confidence)

            # Monte Carlo simulation
            simulations = monte_carlo_simulation(
                weights, annual_returns, cov_matrix,
                n_sims=n_monte_carlo, n_days=252
            )

            # Store results
            st.session_state['risk_analysis'] = {
                'weights': weights,
                'metrics': metrics,
                'returns': returns,
                'portfolio_returns': portfolio_returns,
                'var_hist': var_hist,
                'var_param': var_param,
                'cvar': cvar,
                'simulations': simulations,
                'strategy': strategy
            }

        st.sidebar.success("✅ Analysis Complete!")

    # Main content
    if 'risk_analysis' in st.session_state:
        analysis = st.session_state['risk_analysis']

        # Summary metrics
        st.header("📊 Risk Summary")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "VaR (95%, Hist)",
                f"{analysis['var_hist'] * 100:.2f}%",
                help="5% chance of losing more than this in one day"
            )

        with col2:
            st.metric(
                "CVaR (95%)",
                f"{analysis['cvar'] * 100:.2f}%",
                help="Expected loss if VaR is exceeded"
            )

        with col3:
            port_vol = analysis['portfolio_returns'].std() * np.sqrt(252) * 100
            st.metric(
                "Annual Volatility",
                f"{port_vol:.2f}%"
            )

        with col4:
            downside_dev = analysis['portfolio_returns'][analysis['portfolio_returns'] < 0].std() * np.sqrt(252) * 100
            st.metric(
                "Downside Deviation",
                f"{downside_dev:.2f}%"
            )

        # Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "VaR Analysis", "Monte Carlo", "Risk Decomposition", "Stress Testing", "Advanced Metrics"
        ])

        with tab1:
            st.subheader("Value at Risk Analysis")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### VaR Methods Comparison")
                var_comparison = pd.DataFrame({
                    'Method': ['Historical', 'Parametric'],
                    'VaR (95%)': [
                        f"{analysis['var_hist'] * 100:.2f}%",
                        f"{analysis['var_param'] * 100:.2f}%"
                    ],
                    '1-Day Loss ($1000)': [
                        f"${abs(analysis['var_hist'] * 1000):.2f}",
                        f"${abs(analysis['var_param'] * 1000):.2f}"
                    ]
                })
                st.dataframe(var_comparison, use_container_width=True)

                st.info("""
                **VaR Interpretation:**
                - **Historical VaR**: Based on actual past returns
                - **Parametric VaR**: Assumes normal distribution
                - **CVaR**: Average loss beyond VaR threshold
                """)

            with col2:
                # Returns distribution with VaR
                fig_dist = go.Figure()

                returns_pct = analysis['portfolio_returns'] * 100

                fig_dist.add_trace(go.Histogram(
                    x=returns_pct,
                    nbinsx=50,
                    name='Daily Returns',
                    marker_color='lightblue'
                ))

                # VaR line
                fig_dist.add_vline(
                    x=analysis['var_hist'] * 100,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"VaR ({var_confidence:.0%})"
                )

                # CVaR line
                fig_dist.add_vline(
                    x=analysis['cvar'] * 100,
                    line_dash="dash",
                    line_color="darkred",
                    annotation_text="CVaR"
                )

                fig_dist.update_layout(
                    title='Returns Distribution with VaR',
                    xaxis_title='Daily Return (%)',
                    yaxis_title='Frequency',
                    height=400
                )

                st.plotly_chart(fig_dist, use_container_width=True)

            # VaR over time
            st.markdown("### Rolling VaR (63-day window)")
            rolling_var = analysis['portfolio_returns'].rolling(63).apply(
                lambda x: calculate_var(x, var_confidence, 'historical')
            ) * 100

            fig_rolling_var = go.Figure()

            fig_rolling_var.add_trace(go.Scatter(
                x=rolling_var.index,
                y=rolling_var.values,
                mode='lines',
                fill='tozeroy',
                line=dict(width=2, color='red'),
                name='Rolling VaR'
            ))

            fig_rolling_var.update_layout(
                title=f'Rolling VaR ({var_confidence:.0%} confidence)',
                xaxis_title='Date',
                yaxis_title='VaR (%)',
                height=400
            )

            st.plotly_chart(fig_rolling_var, use_container_width=True)

        with tab2:
            st.subheader("Monte Carlo Simulation")

            # Fan chart
            dates = list(range(252))
            fig_mc = create_monte_carlo_fan(analysis['simulations'], dates)
            st.plotly_chart(fig_mc, use_container_width=True)

            # Statistics
            col1, col2, col3 = st.columns(3)

            final_values = analysis['simulations'][:, -1]

            with col1:
                st.metric("Median Final Value", f"${np.median(final_values):.2f}")

            with col2:
                prob_profit = (final_values > 1.0).sum() / len(final_values) * 100
                st.metric("Probability of Profit", f"{prob_profit:.1f}%")

            with col3:
                prob_loss_10 = (final_values < 0.9).sum() / len(final_values) * 100
                st.metric("Prob of >10% Loss", f"{prob_loss_10:.1f}%")

            # Distribution of final values
            st.markdown("### Distribution of Final Values")

            fig_final = go.Figure()

            fig_final.add_trace(go.Histogram(
                x=final_values,
                nbinsx=50,
                marker_color='lightgreen'
            ))

            fig_final.add_vline(x=1.0, line_dash="dash", annotation_text="Break-even")
            fig_final.add_vline(x=np.median(final_values), line_dash="dash", line_color="blue", annotation_text="Median")

            fig_final.update_layout(
                title='Distribution of 1-Year Portfolio Values',
                xaxis_title='Final Value',
                yaxis_title='Frequency',
                height=400
            )

            st.plotly_chart(fig_final, use_container_width=True)

        with tab3:
            st.subheader("Risk Decomposition")

            weights = analysis['weights']
            returns = analysis['returns']

            # Individual asset contributions to risk
            individual_vols = returns.std() * np.sqrt(252) * 100
            weight_contributions = weights * individual_vols

            fig_risk_decomp = go.Figure()

            active_assets = weights[weights > 1e-4].sort_values(ascending=True)

            fig_risk_decomp.add_trace(go.Bar(
                y=active_assets.index,
                x=(active_assets * individual_vols[active_assets.index]).values,
                orientation='h',
                marker_color='coral',
                text=[f'{v:.2f}%' for v in (active_assets * individual_vols[active_assets.index]).values],
                textposition='auto'
            ))

            fig_risk_decomp.update_layout(
                title='Risk Contribution by Asset',
                xaxis_title='Contribution to Portfolio Risk (%)',
                yaxis_title='Asset',
                height=400
            )

            st.plotly_chart(fig_risk_decomp, use_container_width=True)

            # Correlation-based risk
            st.markdown("### Diversification Benefit")

            total_undiversified = (weights * individual_vols).sum()
            actual_risk = analysis['portfolio_returns'].std() * np.sqrt(252) * 100
            diversification_benefit = ((total_undiversified - actual_risk) / total_undiversified * 100)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Undiversified Risk", f"{total_undiversified:.2f}%")

            with col2:
                st.metric("Actual Portfolio Risk", f"{actual_risk:.2f}%")

            with col3:
                st.metric("Diversification Benefit", f"{diversification_benefit:.1f}%")

            st.info("Diversification benefit shows how much risk is reduced through asset correlation")

        with tab4:
            st.subheader("Stress Testing")

            st.markdown("### Scenario Analysis")

            # Define stress scenarios
            scenarios = {
                'Market Crash (-20%)': {asset: -0.20 for asset in returns.columns},
                'Moderate Correction (-10%)': {asset: -0.10 for asset in returns.columns},
                'Sector Rotation': {returns.columns[i]: 0.10 if i % 2 == 0 else -0.10
                                   for i in range(len(returns.columns))},
                'High Volatility Shock': {asset: np.random.uniform(-0.15, 0.15)
                                         for asset in returns.columns}
            }

            stress_results = stress_test(weights, returns, scenarios)

            # Display results
            stress_data = []
            for scenario, results in stress_results.items():
                stress_data.append({
                    'Scenario': scenario,
                    'Total Return': f"{results['total_return']:.2f}%",
                    'Max Drawdown': f"{results['max_drawdown']:.2f}%"
                })

            stress_df = pd.DataFrame(stress_data)
            st.dataframe(
                stress_df.style.background_gradient(subset=['Total Return'], cmap='RdYlGn'),
                use_container_width=True
            )

            # Visualization
            fig_stress = go.Figure()

            scenarios_list = list(stress_results.keys())
            returns_list = [stress_results[s]['total_return'] for s in scenarios_list]

            fig_stress.add_trace(go.Bar(
                x=scenarios_list,
                y=returns_list,
                marker_color=['red' if r < 0 else 'green' for r in returns_list],
                text=[f'{r:.1f}%' for r in returns_list],
                textposition='auto'
            ))

            fig_stress.update_layout(
                title='Stress Test Results',
                xaxis_title='Scenario',
                yaxis_title='Portfolio Return (%)',
                height=400
            )

            st.plotly_chart(fig_stress, use_container_width=True)

        with tab5:
            st.subheader("Advanced Risk Metrics")

            port_returns = analysis['portfolio_returns']

            # Calculate advanced metrics
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Return-Based Metrics")

                # Sortino ratio
                downside_returns = port_returns[port_returns < 0]
                downside_std = downside_returns.std() * np.sqrt(252)
                sortino = (port_returns.mean() * 252) / downside_std if downside_std > 0 else 0

                st.write(f"**Sortino Ratio:** {sortino:.3f}")
                st.write(f"**Skewness:** {port_returns.skew():.3f}")
                st.write(f"**Kurtosis:** {port_returns.kurtosis():.3f}")

                # Calmar ratio
                cumulative = (1 + port_returns).cumprod()
                max_dd = ((cumulative / cumulative.cummax()) - 1).min()
                calmar = (port_returns.mean() * 252) / abs(max_dd) if max_dd != 0 else 0

                st.write(f"**Calmar Ratio:** {calmar:.3f}")

            with col2:
                st.markdown("### Distribution Metrics")

                # Percentile analysis
                percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
                percentile_values = np.percentile(port_returns * 100, percentiles)

                perc_df = pd.DataFrame({
                    'Percentile': [f'{p}%' for p in percentiles],
                    'Return': [f'{v:.2f}%' for v in percentile_values]
                })

                st.dataframe(perc_df, use_container_width=True, height=300)

            # Q-Q plot
            st.markdown("### Normality Check (Q-Q Plot)")

            fig_qq = go.Figure()

            # Theoretical quantiles
            theoretical = stats.probplot(port_returns.dropna(), dist="norm")[0][0]
            sample = stats.probplot(port_returns.dropna(), dist="norm")[0][1]

            fig_qq.add_trace(go.Scatter(
                x=theoretical,
                y=sample,
                mode='markers',
                name='Actual',
                marker=dict(size=5, color='blue')
            ))

            # 45-degree line
            min_val = min(theoretical.min(), sample.min())
            max_val = max(theoretical.max(), sample.max())
            fig_qq.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Normal',
                line=dict(color='red', dash='dash')
            ))

            fig_qq.update_layout(
                title='Q-Q Plot (Normal Distribution)',
                xaxis_title='Theoretical Quantiles',
                yaxis_title='Sample Quantiles',
                height=400
            )

            st.plotly_chart(fig_qq, use_container_width=True)

            st.info("Points on the line indicate normal distribution. Deviations suggest fat tails or skewness.")

    else:
        # Welcome screen
        st.info("👈 Configure parameters and click 'Analyze Risk' to begin!")

        st.header("📚 About Risk Analytics")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            ### ⚠️ Risk Metrics

            **Value at Risk (VaR)**
            - Maximum loss at given confidence level
            - 95% VaR = 5% chance of exceeding loss

            **Conditional VaR (CVaR)**
            - Expected loss beyond VaR
            - Also called Expected Shortfall

            **Stress Testing**
            - Portfolio behavior in extreme scenarios
            - Market crashes, sector rotations
            """)

        with col2:
            st.markdown("""
            ### 📊 Analysis Methods

            **Monte Carlo Simulation**
            - Simulate thousands of possible outcomes
            - Estimate probability distributions

            **Risk Decomposition**
            - Individual asset contributions
            - Diversification benefits

            **Advanced Metrics**
            - Sortino ratio (downside risk)
            - Calmar ratio (drawdown-adjusted)
            - Distribution analysis
            """)


if __name__ == '__main__':
    main()
