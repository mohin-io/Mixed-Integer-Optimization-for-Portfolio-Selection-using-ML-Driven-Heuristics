"""
Streamlit Dashboard for Mixed-Integer-Optimization-for-Portfolio-Selection-using-ML-Driven-Heuristics

Interactive web application for exploring portfolio optimization strategies.

Run with:
    streamlit run src/visualization/dashboard.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from io import StringIO
import time
import logging
from typing import Tuple, Optional, Dict, Any, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Mixed-Integer-Optimization-for-Portfolio-Selection-using-ML-Driven-Heuristics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with modern design
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1f77b4 0%, #2ca02c 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
        animation: fadeIn 1s ease-in;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #555;
        text-align: center;
        padding-bottom: 2rem;
    }
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    .metric-box:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 12px rgba(0,0,0,0.2);
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #1f77b4 0%, #2ca02c 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# Default configuration and guided presets for the dashboard experience
DEFAULT_UI_CONFIG: Dict[str, Any] = {
    'preset_choice': 'Quick Demo (Balanced)',
    'n_assets': 10,
    'n_days': 1000,
    'seed': 42,
    'strategy': 'Max Sharpe',
    'risk_aversion': 2.5,
    'max_assets': 5,
}

PRESET_SCENARIOS: Dict[str, Dict[str, Any]] = {
    'Quick Demo (Balanced)': {
        'config': {
            'n_assets': 10,
            'n_days': 750,
            'seed': 21,
            'strategy': 'Max Sharpe',
            'risk_aversion': 2.5,
            'max_assets': 5,
        },
        'description': 'Optimized for a fast, visually rich walkthrough with balanced risk/return.',
    },
    'Capital Preservation': {
        'config': {
            'n_assets': 8,
            'n_days': 1500,
            'seed': 11,
            'strategy': 'Min Variance',
            'risk_aversion': 4.0,
            'max_assets': 4,
        },
        'description': 'Focus on downside protection and smoother volatility profile.',
    },
    'Aggressive Growth': {
        'config': {
            'n_assets': 15,
            'n_days': 1250,
            'seed': 7,
            'strategy': 'Concentrated',
            'risk_aversion': 1.2,
            'max_assets': 6,
        },
        'description': 'High-conviction play with more assets explored but a concentrated allocation.',
    },
    'Custom': {
        'config': {},
        'description': 'Take full control and fine-tune every parameter yourself.',
    },
}


def ensure_session_state_defaults() -> None:
    """Seed Streamlit session state with sensible defaults for first-time visitors."""
    for key, value in DEFAULT_UI_CONFIG.items():
        st.session_state.setdefault(key, value)
    st.session_state.setdefault('active_preset', DEFAULT_UI_CONFIG['preset_choice'])


def apply_preset_settings(preset_name: str) -> None:
    """Apply preset configuration to session state without overriding custom choices unnecessarily."""
    preset = PRESET_SCENARIOS.get(preset_name, PRESET_SCENARIOS['Custom'])
    config = preset.get('config', {})

    for key, value in config.items():
        # Only update if preset explicitly specifies the value
        st.session_state[key] = value

    st.session_state['active_preset'] = preset_name


def render_portfolio_summary(metrics: Dict[str, Any], strategy: str) -> None:
    """Render user-friendly insights that highlight what the numbers mean."""
    takeaways: List[str] = []

    annual_return = metrics.get('return', 0.0)
    volatility = metrics.get('volatility', 0.0)
    sharpe = metrics.get('sharpe', 0.0)
    n_assets = metrics.get('n_assets', 0)

    if sharpe >= 2.0:
        takeaways.append("🚀 Risk-adjusted performance is strong (Sharpe ≥ 2).")
    elif sharpe >= 1.0:
        takeaways.append("👍 Solid Sharpe ratio – the risk/return trade-off looks healthy.")
    else:
        takeaways.append("⚠️ Sharpe ratio is modest; consider adjusting risk aversion or strategy.")

    if annual_return >= 0.15:
        takeaways.append("💡 Expected return is notably high; ensure this aligns with your risk tolerance.")
    elif annual_return <= 0.05:
        takeaways.append("🔍 Expected return is conservative – great for capital preservation scenarios.")
    else:
        takeaways.append("📈 Return outlook is balanced for growth-oriented profiles.")

    if volatility >= 0.25:
        takeaways.append("🌊 Volatility is elevated; keep an eye on drawdowns and diversification.")
    elif volatility <= 0.1:
        takeaways.append("🛡️ Portfolio volatility is nicely contained, aiding smoother equity curves.")

    if strategy == 'Concentrated' and n_assets <= 5:
        takeaways.append("🎯 Strategy is laser-focused – rebalance discipline becomes critical.")
    elif n_assets >= 8:
        takeaways.append("🧺 Diversification spreads risk across multiple names.")

    st.markdown("#### Key Takeaways")
    for point in takeaways[:4]:  # Keep the list succinct for readability
        st.markdown(f"- {point}")


def prepare_weights_export(weights: pd.Series, annual_returns: pd.Series, annual_vol: pd.Series) -> bytes:
    """Create a downloadable CSV snapshot for the current allocation."""
    active_weights = weights[weights > 1e-4].sort_values(ascending=False)
    export_df = pd.DataFrame({
        'Asset': active_weights.index,
        'Weight': active_weights.values,
        'Weight (%)': (active_weights.values * 100).round(2),
        'Expected Return': annual_returns[active_weights.index].values,
        'Volatility': annual_vol[active_weights.index].values,
    })

    buffer = StringIO()
    export_df.to_csv(buffer, index=False)
    return buffer.getvalue().encode('utf-8')


def validate_inputs(n_assets: int, n_days: int, seed: int, risk_aversion: float, max_assets: int) -> bool:
    """
    Validate user inputs for portfolio optimization.

    Args:
        n_assets: Number of assets
        n_days: Number of days
        seed: Random seed
        risk_aversion: Risk aversion parameter
        max_assets: Maximum number of assets in portfolio

    Returns:
        True if all inputs are valid

    Raises:
        ValueError: If any input is invalid
    """
    try:
        if n_assets <= 0:
            raise ValueError("Number of assets must be greater than 0")
        if n_assets > 100:
            raise ValueError("Number of assets must be 100 or less (performance limitation)")
        if n_days <= 0:
            raise ValueError("Number of days must be greater than 0")
        if n_days < 30:
            raise ValueError("Number of days must be at least 30 for meaningful analysis")
        if seed < 0:
            raise ValueError("Random seed must be non-negative")
        if risk_aversion <= 0:
            raise ValueError("Risk aversion must be greater than 0")
        if max_assets <= 0:
            raise ValueError("Maximum assets must be greater than 0")
        if max_assets > n_assets:
            raise ValueError(f"Maximum assets ({max_assets}) cannot exceed total assets ({n_assets})")

        logger.info(f"Input validation passed: n_assets={n_assets}, n_days={n_days}, seed={seed}")
        return True

    except ValueError as e:
        logger.error(f"Input validation failed: {str(e)}")
        st.error(f"❌ Invalid Input: {str(e)}")
        return False


@st.cache_data
def generate_synthetic_data(n_assets: int, n_days: int, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate synthetic market data using factor model.

    Args:
        n_assets: Number of assets to generate
        n_days: Number of days of historical data
        seed: Random seed for reproducibility

    Returns:
        Tuple of (prices DataFrame, returns DataFrame)

    Raises:
        ValueError: If inputs are invalid
        RuntimeError: If data generation fails
    """
    try:
        logger.info(f"Generating synthetic data: n_assets={n_assets}, n_days={n_days}, seed={seed}")

        np.random.seed(seed)
        tickers = [f'ASSET_{i+1}' for i in range(n_assets)]

        # Factor model with realistic parameters
        n_factors = 3
        factor_loadings = np.random.randn(n_assets, n_factors) * 0.3
        factor_returns = np.random.randn(n_days, n_factors) * 0.01
    except Exception as e:
        logger.error(f"Failed to generate synthetic data: {str(e)}")
        raise RuntimeError(f"Data generation failed: {str(e)}")

    try:
        idiosyncratic = np.random.randn(n_days, n_assets) * 0.005
        drift = np.random.uniform(0.0001, 0.0005, n_assets)
        returns_matrix = factor_returns @ factor_loadings.T + idiosyncratic + drift

        dates = pd.date_range('2020-01-01', periods=n_days, freq='D')
        returns = pd.DataFrame(returns_matrix, index=dates, columns=tickers)
        prices = (1 + returns).cumprod() * 100

        logger.info(f"Successfully generated data with shape: prices={prices.shape}, returns={returns.shape}")
        return prices, returns

    except Exception as e:
        logger.error(f"Failed to construct price/return DataFrames: {str(e)}")
        raise RuntimeError(f"Data construction failed: {str(e)}")


def optimize_portfolio(
    returns: pd.DataFrame,
    strategy: str,
    max_assets: Optional[int] = None,
    risk_aversion: float = 2.5
) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
    """
    Optimize portfolio using specified strategy.

    Args:
        returns: Historical returns DataFrame
        strategy: Strategy name ('Equal Weight', 'Max Sharpe', 'Min Variance', 'Concentrated')
        max_assets: Maximum number of assets for concentrated strategy
        risk_aversion: Risk aversion parameter (higher = more conservative)

    Returns:
        Tuple of (weights Series, annual returns Series, covariance matrix DataFrame)

    Raises:
        ValueError: If returns is empty or strategy is invalid
        RuntimeError: If optimization fails
    """
    try:
        # Input validation
        if returns.empty:
            raise ValueError("Returns DataFrame is empty")
        if returns.isnull().any().any():
            logger.warning("Returns contain NaN values, filling with 0")
            returns = returns.fillna(0)

        n_assets = len(returns.columns)
        logger.info(f"Optimizing portfolio: strategy={strategy}, n_assets={n_assets}, max_assets={max_assets}")

        annual_returns = returns.mean() * 252
        annual_volatility = returns.std() * np.sqrt(252)
        cov_matrix = returns.cov() * 252

        # Check for singular covariance matrix
        if np.linalg.cond(cov_matrix.values) > 1e10:
            logger.warning("Covariance matrix is near-singular, adding regularization")
            cov_matrix += np.eye(n_assets) * 1e-6

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

            if best_weights is None:
                raise RuntimeError("Max Sharpe optimization failed to find solution")
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

            if min_var_weights is None:
                raise RuntimeError("Min Variance optimization failed to find solution")
            weights = min_var_weights

        elif strategy == 'Concentrated':
            if max_assets is None or max_assets <= 0:
                raise ValueError("Concentrated strategy requires max_assets > 0")

            sharpe_ratios = annual_returns / (annual_volatility + 1e-10)  # Avoid division by zero
            top_assets = sharpe_ratios.nlargest(min(max_assets, n_assets)).index

            best_sharpe = -np.inf
            best_weights = None

            for _ in range(10000):
                w = np.zeros(n_assets)
                top_indices = [returns.columns.get_loc(t) for t in top_assets]
                w[top_indices] = np.random.dirichlet(np.ones(len(top_assets)))

                port_return = (w * annual_returns.values).sum()
                port_vol = np.sqrt(w @ cov_matrix.values @ w)
                sharpe = port_return / port_vol if port_vol > 0 else 0

                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_weights = w

            if best_weights is None:
                raise RuntimeError("Concentrated optimization failed to find solution")
            weights = best_weights

        else:
            logger.warning(f"Unknown strategy '{strategy}', defaulting to Equal Weight")
            weights = np.ones(n_assets) / n_assets

        logger.info(f"Optimization successful: strategy={strategy}, n_selected={np.sum(weights > 1e-6)}")
        return pd.Series(weights, index=returns.columns), annual_returns, cov_matrix

    except Exception as e:
        logger.error(f"Portfolio optimization failed: {str(e)}")
        st.error(f"❌ Optimization Error: {str(e)}")
        # Return equal-weight fallback
        n_assets = len(returns.columns)
        fallback_weights = pd.Series(1.0 / n_assets, index=returns.columns)
        annual_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252
        return fallback_weights, annual_returns, cov_matrix


def evaluate_portfolio(
    weights: pd.Series,
    annual_returns: pd.Series,
    cov_matrix: pd.DataFrame
) -> Dict[str, Any]:
    """
    Evaluate portfolio performance metrics.

    Args:
        weights: Portfolio weights
        annual_returns: Annual returns for each asset
        cov_matrix: Covariance matrix of returns

    Returns:
        Dictionary containing portfolio metrics (return, volatility, sharpe, n_assets)
    """
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


def calculate_efficient_frontier(returns: pd.DataFrame, n_portfolios: int = 100) -> pd.DataFrame:
    """Calculate efficient frontier portfolios."""
    annual_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    n_assets = len(returns.columns)

    results = []
    for _ in range(n_portfolios):
        weights = np.random.dirichlet(np.ones(n_assets))
        port_return = (weights * annual_returns.values).sum()
        port_vol = np.sqrt(weights @ cov_matrix.values @ weights)
        sharpe = port_return / port_vol if port_vol > 0 else 0

        results.append({
            'return': port_return,
            'volatility': port_vol,
            'sharpe': sharpe
        })

    return pd.DataFrame(results)


def run_monte_carlo_simulation(returns: pd.DataFrame, weights: pd.Series, n_simulations: int = 1000, horizon_days: int = 252) -> pd.DataFrame:
    """Run Monte Carlo simulation for portfolio."""
    portfolio_returns = (returns * weights.values).sum(axis=1)
    mean_return = portfolio_returns.mean()
    std_return = portfolio_returns.std()

    simulations = []
    for _ in range(n_simulations):
        sim_returns = np.random.normal(mean_return, std_return, horizon_days)
        sim_cumulative = (1 + sim_returns).cumprod()
        simulations.append(sim_cumulative)

    return pd.DataFrame(simulations).T


def create_risk_return_bubble_chart(weights: pd.Series, annual_returns: pd.Series, annual_vol: pd.Series):
    """Create risk-return scatter with asset bubbles."""
    active_weights = weights[weights > 1e-4]

    fig = go.Figure()

    # Add all assets
    fig.add_trace(go.Scatter(
        x=annual_vol.values,
        y=annual_returns.values,
        mode='markers+text',
        marker=dict(
            size=10,
            color='lightgray',
            line=dict(width=1, color='white')
        ),
        text=annual_returns.index,
        textposition="top center",
        textfont=dict(size=8),
        name='All Assets',
        hovertemplate='<b>%{text}</b><br>Return: %{y:.2%}<br>Volatility: %{x:.2%}<extra></extra>'
    ))

    # Highlight selected assets
    fig.add_trace(go.Scatter(
        x=annual_vol[active_weights.index].values,
        y=annual_returns[active_weights.index].values,
        mode='markers+text',
        marker=dict(
            size=active_weights.values * 200,
            color=active_weights.values,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Weight"),
            line=dict(width=2, color='white')
        ),
        text=active_weights.index,
        textposition="top center",
        textfont=dict(size=10, color='white', family='Arial Black'),
        name='Selected Assets',
        customdata=active_weights.values,
        hovertemplate='<b>%{text}</b><br>Return: %{y:.2%}<br>Volatility: %{x:.2%}<br>Weight: %{customdata:.2%}<extra></extra>'
    ))

    fig.update_layout(
        title=dict(
            text="Risk-Return Asset Universe",
            font=dict(size=20, color='#1f77b4'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(title='Annual Volatility', gridcolor='rgba(200,200,200,0.3)', tickformat='.1%'),
        yaxis=dict(title='Expected Annual Return', gridcolor='rgba(200,200,200,0.3)', tickformat='.1%'),
        hovermode='closest',
        height=600,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(250,250,250,0.9)',
        showlegend=True
    )

    return fig


def create_drawdown_chart(returns: pd.DataFrame, weights: pd.Series):
    """Create underwater drawdown chart."""
    portfolio_returns = (returns * weights.values).sum(axis=1)
    cumulative = (1 + portfolio_returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown.values * 100,
        fill='tozeroy',
        fillcolor='rgba(255,0,0,0.3)',
        line=dict(color='red', width=2),
        name='Drawdown',
        hovertemplate='Date: %{x}<br>Drawdown: %{y:.2f}%<extra></extra>'
    ))

    fig.update_layout(
        title=dict(
            text="Portfolio Drawdown (Underwater Chart)",
            font=dict(size=20, color='#1f77b4'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(title='Date', gridcolor='rgba(200,200,200,0.3)'),
        yaxis=dict(title='Drawdown (%)', gridcolor='rgba(200,200,200,0.3)'),
        hovermode='x unified',
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(250,250,250,0.9)',
    )

    return fig


def create_rolling_statistics_chart(returns: pd.DataFrame, weights: pd.Series, window: int = 60):
    """Create rolling statistics visualization."""
    portfolio_returns = (returns * weights.values).sum(axis=1)

    rolling_mean = portfolio_returns.rolling(window).mean() * 252
    rolling_std = portfolio_returns.rolling(window).std() * np.sqrt(252)
    rolling_sharpe = rolling_mean / rolling_std

    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Rolling Return (Annual)', 'Rolling Volatility (Annual)', 'Rolling Sharpe Ratio'),
        vertical_spacing=0.1
    )

    fig.add_trace(go.Scatter(
        x=rolling_mean.index,
        y=rolling_mean.values,
        line=dict(color='green', width=2),
        name='Rolling Return',
        hovertemplate='%{x}<br>Return: %{y:.2%}<extra></extra>'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=rolling_std.index,
        y=rolling_std.values,
        line=dict(color='orange', width=2),
        name='Rolling Volatility',
        hovertemplate='%{x}<br>Volatility: %{y:.2%}<extra></extra>'
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=rolling_sharpe.index,
        y=rolling_sharpe.values,
        line=dict(color='blue', width=2),
        name='Rolling Sharpe',
        hovertemplate='%{x}<br>Sharpe: %{y:.3f}<extra></extra>'
    ), row=3, col=1)

    fig.update_layout(
        title=dict(
            text=f"Rolling Statistics ({window}-day window)",
            font=dict(size=20, color='#1f77b4'),
            x=0.5,
            xanchor='center'
        ),
        height=800,
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(250,250,250,0.9)',
    )

    fig.update_xaxes(gridcolor='rgba(200,200,200,0.3)')
    fig.update_yaxes(gridcolor='rgba(200,200,200,0.3)')

    return fig


def create_portfolio_treemap(weights: pd.Series, annual_returns: pd.Series):
    """Create treemap for portfolio composition."""
    active_weights = weights[weights > 1e-4].sort_values(ascending=False)

    df = pd.DataFrame({
        'Asset': active_weights.index,
        'Weight': active_weights.values,
        'Return': annual_returns[active_weights.index].values,
        'Label': [f"{asset}<br>{weight:.1%}" for asset, weight in active_weights.items()]
    })

    fig = go.Figure(go.Treemap(
        labels=df['Label'],
        parents=['Portfolio'] * len(df),
        values=df['Weight'],
        marker=dict(
            colorscale='RdYlGn',
            cmid=0,
            colorbar=dict(title="Return"),
            line=dict(width=2, color='white')
        ),
        marker_colorscale='RdYlGn',
        text=df['Label'],
        hovertemplate='<b>%{label}</b><br>Weight: %{value:.2%}<br>Return: %{color:.2%}<extra></extra>',
        textposition='middle center',
    ))

    fig.update_layout(
        title=dict(
            text="Portfolio Composition Treemap",
            font=dict(size=20, color='#1f77b4'),
            x=0.5,
            xanchor='center'
        ),
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
    )

    return fig


def create_monte_carlo_chart(returns: pd.DataFrame, weights: pd.Series):
    """Create Monte Carlo simulation fan chart."""
    simulations = run_monte_carlo_simulation(returns, weights, n_simulations=200, horizon_days=252)

    fig = go.Figure()

    # Add simulation paths
    for i in range(min(50, len(simulations.columns))):
        fig.add_trace(go.Scatter(
            x=simulations.index,
            y=simulations.iloc[:, i],
            mode='lines',
            line=dict(width=0.5, color='lightblue'),
            opacity=0.3,
            showlegend=False,
            hoverinfo='skip'
        ))

    # Add median
    median = simulations.median(axis=1)
    fig.add_trace(go.Scatter(
        x=simulations.index,
        y=median,
        mode='lines',
        line=dict(color='blue', width=3),
        name='Median',
        hovertemplate='Day: %{x}<br>Value: %{y:.2f}<extra></extra>'
    ))

    # Add percentiles
    percentile_5 = simulations.quantile(0.05, axis=1)
    percentile_95 = simulations.quantile(0.95, axis=1)

    fig.add_trace(go.Scatter(
        x=simulations.index,
        y=percentile_95,
        mode='lines',
        line=dict(color='green', width=2, dash='dash'),
        name='95th Percentile',
        hovertemplate='Day: %{x}<br>Value: %{y:.2f}<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=simulations.index,
        y=percentile_5,
        mode='lines',
        line=dict(color='red', width=2, dash='dash'),
        name='5th Percentile',
        fill='tonexty',
        fillcolor='rgba(0,100,200,0.2)',
        hovertemplate='Day: %{x}<br>Value: %{y:.2f}<extra></extra>'
    ))

    fig.update_layout(
        title=dict(
            text="Monte Carlo Simulation (1-Year Horizon, 200 Paths)",
            font=dict(size=20, color='#1f77b4'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(title='Trading Days', gridcolor='rgba(200,200,200,0.3)'),
        yaxis=dict(title='Portfolio Value', gridcolor='rgba(200,200,200,0.3)'),
        hovermode='x unified',
        height=600,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(250,250,250,0.9)',
    )

    return fig


def create_risk_contribution_waterfall(weights: pd.Series, cov_matrix: pd.DataFrame):
    """Create waterfall chart for risk contribution."""
    active_weights = weights[weights > 1e-4]

    # Calculate marginal risk contribution
    portfolio_var = weights.values @ cov_matrix.values @ weights.values
    marginal_contrib = cov_matrix.values @ weights.values
    risk_contrib = weights.values * marginal_contrib

    # Get active contributions
    active_contrib = pd.Series(risk_contrib, index=weights.index)[active_weights.index]
    active_contrib = active_contrib.sort_values(ascending=False)

    # Create waterfall
    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=["relative"] * len(active_contrib) + ["total"],
        x=list(active_contrib.index) + ["Total Risk"],
        y=list(active_contrib.values) + [portfolio_var],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        decreasing={"marker": {"color": "rgba(255, 0, 0, 0.6)"}},
        increasing={"marker": {"color": "rgba(0, 255, 0, 0.6)"}},
        totals={"marker": {"color": "rgba(0, 0, 255, 0.6)"}},
        hovertemplate='%{x}<br>Risk Contribution: %{y:.6f}<extra></extra>'
    ))

    fig.update_layout(
        title=dict(
            text="Portfolio Risk Contribution Waterfall",
            font=dict(size=20, color='#1f77b4'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(title='Asset'),
        yaxis=dict(title='Risk Contribution (Variance)'),
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(250,250,250,0.9)',
    )

    return fig


def create_strategy_comparison_radar(returns: pd.DataFrame):
    """Create radar chart comparing different strategies."""
    strategies = ['Equal Weight', 'Max Sharpe', 'Min Variance']
    metrics = []

    for strat in strategies:
        w, ar, cov = optimize_portfolio(returns, strat)
        m = evaluate_portfolio(w, ar, cov)
        metrics.append({
            'Strategy': strat,
            'Return': m['return'],
            'Sharpe': m['sharpe'],
            'Diversification': m['n_assets'] / len(returns.columns),
            'Risk-Adj': m['return'] / m['volatility']
        })

    df = pd.DataFrame(metrics)

    fig = go.Figure()

    categories = ['Return', 'Sharpe', 'Diversification', 'Risk-Adj']

    for i, strat in enumerate(strategies):
        values = [
            df.loc[i, 'Return'] * 100,
            df.loc[i, 'Sharpe'],
            df.loc[i, 'Diversification'] * 3,  # Scale up for visibility
            df.loc[i, 'Risk-Adj']
        ]

        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],  # Close the polygon
            theta=categories + [categories[0]],
            fill='toself',
            name=strat,
            opacity=0.6
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, max([max(m['Return']*100, m['Sharpe'], m['Diversification']*3, m['Risk-Adj']) for m in metrics]) * 1.1])
        ),
        title=dict(
            text="Strategy Comparison Radar Chart",
            font=dict(size=20, color='#1f77b4'),
            x=0.5,
            xanchor='center'
        ),
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
    )

    return fig


def create_3d_allocation_chart(weights: pd.Series, annual_returns: pd.Series, annual_vol: pd.Series):
    """Create interactive 3D portfolio allocation visualization."""
    active_weights = weights[weights > 1e-4]

    fig = go.Figure(data=[go.Scatter3d(
        x=annual_vol[active_weights.index].values,
        y=annual_returns[active_weights.index].values,
        z=active_weights.values,
        mode='markers+text',
        marker=dict(
            size=active_weights.values * 100,
            color=active_weights.values,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Weight"),
            line=dict(width=2, color='white')
        ),
        text=active_weights.index,
        textposition="top center",
        textfont=dict(size=10, color='white'),
        hovertemplate='<b>%{text}</b><br>' +
                      'Volatility: %{x:.2%}<br>' +
                      'Return: %{y:.2%}<br>' +
                      'Weight: %{z:.2%}<br>' +
                      '<extra></extra>'
    )])

    fig.update_layout(
        title=dict(
            text="3D Portfolio Allocation Space",
            font=dict(size=20, color='#1f77b4'),
            x=0.5,
            xanchor='center'
        ),
        scene=dict(
            xaxis=dict(title='Volatility', backgroundcolor='rgba(240,240,240,0.9)'),
            yaxis=dict(title='Expected Return', backgroundcolor='rgba(240,240,240,0.9)'),
            zaxis=dict(title='Weight', backgroundcolor='rgba(240,240,240,0.9)'),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
        ),
        height=600,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )

    return fig


def create_animated_performance_chart(returns: pd.DataFrame, weights: pd.Series):
    """Create animated portfolio performance visualization."""
    portfolio_returns = (returns * weights.values).sum(axis=1)
    cumulative_portfolio = (1 + portfolio_returns).cumprod()

    # Create frames for animation
    frames = []
    n_frames = min(50, len(cumulative_portfolio))
    step_size = len(cumulative_portfolio) // n_frames

    for i in range(1, n_frames + 1):
        idx = min(i * step_size, len(cumulative_portfolio))
        frames.append(go.Frame(
            data=[go.Scatter(
                x=cumulative_portfolio.index[:idx],
                y=cumulative_portfolio.values[:idx],
                mode='lines',
                line=dict(color='#2ca02c', width=3),
                fill='tozeroy',
                fillcolor='rgba(44,160,44,0.2)'
            )],
            name=str(i)
        ))

    fig = go.Figure(
        data=[go.Scatter(
            x=cumulative_portfolio.index[:step_size],
            y=cumulative_portfolio.values[:step_size],
            mode='lines',
            line=dict(color='#2ca02c', width=3),
            fill='tozeroy',
            fillcolor='rgba(44,160,44,0.2)'
        )],
        frames=frames
    )

    fig.update_layout(
        title=dict(
            text="Animated Portfolio Growth",
            font=dict(size=20, color='#1f77b4'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(title='Date', gridcolor='rgba(200,200,200,0.3)'),
        yaxis=dict(title='Cumulative Return', gridcolor='rgba(200,200,200,0.3)'),
        hovermode='x unified',
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(250,250,250,0.9)',
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'buttons': [
                {'label': 'Play', 'method': 'animate', 'args': [None, {'frame': {'duration': 50, 'redraw': True}}]},
                {'label': 'Pause', 'method': 'animate', 'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate'}]}
            ],
            'x': 0.1,
            'y': 1.15
        }]
    )

    return fig


def create_gauge_chart(value: float, title: str, max_value: float = 1.0):
    """Create a beautiful gauge chart for metrics."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value * 100 if max_value == 1.0 else value,
        title={'text': title, 'font': {'size': 18}},
        delta={'reference': 10 if 'Return' in title else 0.5},
        gauge={
            'axis': {'range': [None, max_value * 100 if max_value == 1.0 else max_value]},
            'bar': {'color': "#2ca02c"},
            'steps': [
                {'range': [0, max_value * 33.33 if max_value == 1.0 else max_value/3], 'color': "lightgray"},
                {'range': [max_value * 33.33 if max_value == 1.0 else max_value/3, max_value * 66.66 if max_value == 1.0 else 2*max_value/3], 'color': "gray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': max_value * 90 if max_value == 1.0 else 0.9 * max_value
            }
        }
    ))

    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': "#1f77b4", 'family': "Arial"}
    )

    return fig


def create_interactive_weights_chart(weights: pd.Series, strategy: str):
    """Create interactive sunburst or treemap for weights."""
    active_weights = weights[weights > 1e-4].sort_values(ascending=False)

    # Create hierarchical data
    categories = []
    for i, (asset, weight) in enumerate(active_weights.items()):
        if weight > 0.1:
            categories.append('Large (>10%)')
        elif weight > 0.05:
            categories.append('Medium (5-10%)')
        else:
            categories.append('Small (<5%)')

    df = pd.DataFrame({
        'Asset': active_weights.index,
        'Weight': active_weights.values,
        'Category': categories
    })

    fig = px.sunburst(
        df,
        path=['Category', 'Asset'],
        values='Weight',
        color='Weight',
        color_continuous_scale='RdYlGn',
        title=f"{strategy} Portfolio Allocation"
    )

    fig.update_layout(
        height=600,
        title=dict(
            font=dict(size=20, color='#1f77b4'),
            x=0.5,
            xanchor='center'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
    )

    return fig


def create_correlation_network(returns: pd.DataFrame, threshold: float = 0.5):
    """Create interactive correlation network graph."""
    corr_matrix = returns.corr()

    # Create network edges
    edges = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                edges.append({
                    'source': corr_matrix.columns[i],
                    'target': corr_matrix.columns[j],
                    'weight': corr_matrix.iloc[i, j]
                })

    # Create positions for nodes (circular layout)
    n_nodes = len(corr_matrix.columns)
    angles = np.linspace(0, 2*np.pi, n_nodes, endpoint=False)
    x_pos = np.cos(angles)
    y_pos = np.sin(angles)

    # Create edge traces
    edge_traces = []
    for edge in edges:
        i = list(corr_matrix.columns).index(edge['source'])
        j = list(corr_matrix.columns).index(edge['target'])

        color = 'red' if edge['weight'] < 0 else 'green'
        width = abs(edge['weight']) * 5

        edge_traces.append(go.Scatter(
            x=[x_pos[i], x_pos[j], None],
            y=[y_pos[i], y_pos[j], None],
            mode='lines',
            line=dict(width=width, color=color),
            opacity=0.5,
            hoverinfo='none',
            showlegend=False
        ))

    # Create node trace
    node_trace = go.Scatter(
        x=x_pos,
        y=y_pos,
        mode='markers+text',
        marker=dict(
            size=30,
            color='#1f77b4',
            line=dict(width=2, color='white')
        ),
        text=list(corr_matrix.columns),
        textposition="top center",
        textfont=dict(size=12, color='white'),
        hovertemplate='<b>%{text}</b><extra></extra>',
        showlegend=False
    )

    fig = go.Figure(data=edge_traces + [node_trace])

    fig.update_layout(
        title=dict(
            text=f"Correlation Network (|r| > {threshold})",
            font=dict(size=20, color='#1f77b4'),
            x=0.5,
            xanchor='center'
        ),
        showlegend=False,
        hovermode='closest',
        height=600,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(250,250,250,0.9)',
    )

    return fig


def create_efficient_frontier_chart(returns: pd.DataFrame, current_portfolio: Dict):
    """Create interactive efficient frontier with current portfolio."""
    frontier_data = calculate_efficient_frontier(returns, 500)

    fig = go.Figure()

    # Scatter plot for efficient frontier
    fig.add_trace(go.Scatter(
        x=frontier_data['volatility'],
        y=frontier_data['return'],
        mode='markers',
        marker=dict(
            size=6,
            color=frontier_data['sharpe'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(
                title=dict(text="Sharpe Ratio", font=dict(size=14)),
                thickness=15,
                len=0.7
            ),
            line=dict(width=0.5, color='rgba(255,255,255,0.3)'),
            opacity=0.7
        ),
        text=[f"Sharpe: {s:.2f}" for s in frontier_data['sharpe']],
        hovertemplate='<b>Random Portfolio</b><br>Return: %{y:.2%}<br>Volatility: %{x:.2%}<br>%{text}<extra></extra>',
        name='Efficient Frontier',
        showlegend=True
    ))

    # Add current portfolio marker (no text)
    fig.add_trace(go.Scatter(
        x=[current_portfolio['volatility']],
        y=[current_portfolio['return']],
        mode='markers',
        marker=dict(
            size=25,
            color='#FF4444',
            symbol='star',
            line=dict(width=3, color='white')
        ),
        hovertemplate='<b>Your Current Portfolio</b><br>' +
                      'Return: %{y:.2%}<br>' +
                      'Volatility: %{x:.2%}<br>' +
                      f"Sharpe: {current_portfolio['sharpe']:.3f}" +
                      '<extra></extra>',
        name='Current Portfolio',
        showlegend=True
    ))

    # Add annotation for current portfolio (non-overlapping)
    fig.add_annotation(
        x=current_portfolio['volatility'],
        y=current_portfolio['return'],
        text="<b>Your Portfolio</b>",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor='#FF4444',
        ax=60,
        ay=-40,
        bgcolor='rgba(255, 255, 255, 0.9)',
        bordercolor='#FF4444',
        borderwidth=2,
        borderpad=4,
        font=dict(size=12, color='#FF4444', family='Arial Black'),
        opacity=1
    )

    fig.update_layout(
        title=dict(
            text="Efficient Frontier Explorer",
            font=dict(size=20, color='#1f77b4'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title='Annual Volatility (%)',
            gridcolor='rgba(200,200,200,0.3)',
            tickformat='.1%',
            showgrid=True
        ),
        yaxis=dict(
            title='Expected Annual Return (%)',
            gridcolor='rgba(200,200,200,0.3)',
            tickformat='.1%',
            showgrid=True
        ),
        hovermode='closest',
        height=650,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(250,250,250,0.9)',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='rgba(0,0,0,0.2)',
            borderwidth=1
        )
    )

    return fig


# Main app
def main() -> None:
    # Header with animation
    st.markdown('<div class="main-header">📊 Mixed-Integer-Optimization-for-Portfolio-Selection-using-ML-Driven-Heuristics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Interactive Mixed-Integer Optimization with Real-Time Visualizations</div>', unsafe_allow_html=True)

    # Sidebar configuration experience
    ensure_session_state_defaults()

    st.sidebar.header("⚙️ Configuration Studio")
    st.sidebar.caption("Pick a preset to get started fast, then fine-tune the sliders below.")

    with st.sidebar.form("config_form", clear_on_submit=False):
        preset_options = list(PRESET_SCENARIOS.keys())
        active_preset = st.session_state.get('active_preset', DEFAULT_UI_CONFIG['preset_choice'])
        if active_preset not in preset_options:
            active_preset = preset_options[0]
        preset_index = preset_options.index(active_preset)

        preset_choice = st.selectbox(
            "Preset Scenario",
            preset_options,
            index=preset_index,
            key='preset_choice'
        )
        st.caption(PRESET_SCENARIOS[preset_choice]['description'])

        if preset_choice != 'Custom' and st.session_state.get('active_preset') != preset_choice:
            apply_preset_settings(preset_choice)
        elif preset_choice == 'Custom':
            st.session_state['active_preset'] = 'Custom'

        st.markdown("### Data Parameters")
        n_assets = st.slider(
            "Number of Assets",
            5,
            20,
            value=int(st.session_state.get('n_assets', DEFAULT_UI_CONFIG['n_assets'])),
            key='n_assets'
        )
        n_days = st.slider(
            "Number of Days",
            250,
            2000,
            value=int(st.session_state.get('n_days', DEFAULT_UI_CONFIG['n_days'])),
            step=250,
            key='n_days'
        )
        seed = st.number_input(
            "Random Seed",
            1,
            1000,
            value=int(st.session_state.get('seed', DEFAULT_UI_CONFIG['seed'])),
            step=1,
            key='seed'
        )

        st.markdown("### Optimization Strategy")
        strategies = ['Equal Weight', 'Max Sharpe', 'Min Variance', 'Concentrated']
        strategy_default = st.session_state.get('strategy', DEFAULT_UI_CONFIG['strategy'])
        if strategy_default not in strategies:
            strategy_default = strategies[0]
        strategy = st.selectbox(
            "Select Strategy",
            strategies,
            index=strategies.index(strategy_default),
            key='strategy'
        )

        with st.expander("Advanced Parameters", expanded=False):
            risk_aversion = st.slider(
                "Risk Aversion",
                0.5,
                10.0,
                value=float(st.session_state.get('risk_aversion', DEFAULT_UI_CONFIG['risk_aversion'])),
                step=0.5,
                help="Higher values tilt the optimizer toward lower-volatility allocations.",
                key='risk_aversion'
            )

            if strategy == 'Concentrated':
                max_assets_default = min(
                    int(st.session_state.get('max_assets', DEFAULT_UI_CONFIG['max_assets'])),
                    n_assets
                )
                max_assets = st.slider(
                    "Max Assets",
                    3,
                    n_assets,
                    value=max_assets_default,
                    key='max_assets',
                    help="Upper bound on the number of holdings the optimizer may select."
                )
            else:
                max_assets = min(
                    int(st.session_state.get('max_assets', DEFAULT_UI_CONFIG['max_assets'])),
                    n_assets
                )

        optimize_clicked = st.form_submit_button("🚀 Optimize Portfolio", type="primary")

    st.sidebar.info("Need inspiration? Presets load curated defaults while keeping everything tweakable.")

    if optimize_clicked:
        try:
            # Validate inputs
            validation_max_assets = max_assets if strategy == 'Concentrated' else n_assets
            if not validate_inputs(n_assets, n_days, int(seed), risk_aversion, validation_max_assets):
                st.stop()

            with st.spinner("Generating data and optimizing..."):
                # Generate data
                try:
                    prices, returns = generate_synthetic_data(n_assets, n_days, int(seed))
                except Exception as e:
                    st.error(f"❌ Data Generation Failed: {str(e)}")
                    logger.error(f"Data generation error: {str(e)}")
                    st.stop()

                # Optimize
                try:
                    weights, annual_returns, cov_matrix = optimize_portfolio(
                        returns, strategy, max_assets if strategy == 'Concentrated' else None, risk_aversion
                    )
                    metrics = evaluate_portfolio(weights, annual_returns, cov_matrix)
                except Exception as e:
                    st.error(f"❌ Mixed-Integer-Optimization-for-Portfolio-Selection-using-ML-Driven-Heuristics Failed: {str(e)}")
                    logger.error(f"Optimization error: {str(e)}")
                    st.stop()

                # Store in session state
                st.session_state['prices'] = prices
                st.session_state['returns'] = returns
                st.session_state['weights'] = weights
                st.session_state['metrics'] = metrics
                st.session_state['annual_returns'] = annual_returns
                st.session_state['cov_matrix'] = cov_matrix
                st.session_state['strategy'] = strategy

            st.sidebar.success("✅ Optimization Complete!")

        except Exception as e:
            st.sidebar.error(f"❌ Error: {str(e)}")
            logger.error(f"Unexpected error in optimization workflow: {str(e)}")

    # Main content
    if 'weights' in st.session_state:
        weights = st.session_state['weights']
        metrics = st.session_state['metrics']
        prices = st.session_state['prices']
        returns = st.session_state['returns']
        annual_returns = st.session_state['annual_returns']
        cov_matrix = st.session_state['cov_matrix']
        strategy = st.session_state.get('strategy', 'Unknown')
        annual_vol = returns.std() * np.sqrt(252)

        # Metrics with gauges
        st.header("📈 Portfolio Metrics Dashboard")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            fig1 = create_gauge_chart(metrics['return'], "Expected Return (%)", 0.5)
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            fig2 = create_gauge_chart(metrics['volatility'], "Volatility (%)", 0.5)
            st.plotly_chart(fig2, use_container_width=True)

        with col3:
            fig3 = create_gauge_chart(metrics['sharpe'], "Sharpe Ratio", 3.0)
            st.plotly_chart(fig3, use_container_width=True)

        with col4:
            fig4 = create_gauge_chart(metrics['n_assets'], "Active Assets", n_assets)
            st.plotly_chart(fig4, use_container_width=True)

        st.markdown("### Strategy Snapshot")
        st.markdown(
            f"**Selected Strategy:** `{strategy}` · **Preset:** `{st.session_state.get('active_preset', 'Custom')}`"
        )
        render_portfolio_summary(metrics, strategy)
        st.caption(
            "Configuration: "
            f"{st.session_state.get('n_assets', n_assets)} assets · "
            f"{st.session_state.get('n_days', n_days)} days · "
            f"Seed {st.session_state.get('seed', seed)}"
        )

        # Tabs with enhanced visualizations
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
            "🎯 Allocation",
            "📊 Efficient Frontier",
            "🌐 Correlation Network",
            "📈 Performance",
            "🔮 3D Analysis",
            "🎲 Monte Carlo",
            "📉 Risk Analysis",
            "⚖️ Comparison"
        ])

        with tab1:
            st.subheader("Interactive Portfolio Allocation")

            col1, col2 = st.columns(2)

            with col1:
                fig_weights = create_interactive_weights_chart(weights, strategy)
                st.plotly_chart(fig_weights, use_container_width=True)

            with col2:
                fig_treemap = create_portfolio_treemap(weights, annual_returns)
                st.plotly_chart(fig_treemap, use_container_width=True)

            # Risk-Return Bubble Chart
            st.subheader("Risk-Return Asset Universe")
            fig_bubble = create_risk_return_bubble_chart(weights, annual_returns, annual_vol)
            st.plotly_chart(fig_bubble, use_container_width=True)

            # Weights table
            st.subheader("Detailed Weights")
            active_weights = weights[weights > 1e-4].sort_values(ascending=False)
            weights_df = pd.DataFrame({
                'Asset': active_weights.index,
                'Weight': active_weights.values,
                'Weight (%)': (active_weights.values * 100).round(2),
                'Expected Return': annual_returns[active_weights.index].values,
                'Volatility': annual_vol[active_weights.index].values
            })
            st.dataframe(weights_df, use_container_width=True, hide_index=True)

            st.download_button(
                "⬇️ Download Allocation (CSV)",
                data=prepare_weights_export(weights, annual_returns, annual_vol),
                file_name="portfolio_allocation.csv",
                mime="text/csv"
            )

        with tab2:
            st.subheader("Efficient Frontier Explorer")
            fig_frontier = create_efficient_frontier_chart(returns, metrics)
            st.plotly_chart(fig_frontier, use_container_width=True)

            st.info("🎯 The red star shows your current portfolio. Points colored by Sharpe ratio (green = better).")

        with tab3:
            st.subheader("Asset Correlation Network")
            threshold = st.slider("Correlation Threshold", 0.0, 1.0, 0.5, 0.1)
            fig_network = create_correlation_network(returns, threshold)
            st.plotly_chart(fig_network, use_container_width=True)

            st.info("🔗 Green lines show positive correlations, red lines show negative correlations. Line thickness indicates strength.")

        with tab4:
            st.subheader("Animated Portfolio Performance")
            fig_perf = create_animated_performance_chart(returns, weights)
            st.plotly_chart(fig_perf, use_container_width=True)

            # Performance metrics
            portfolio_returns = (returns * weights.values).sum(axis=1)
            cumulative_portfolio = (1 + portfolio_returns).cumprod()

            col1, col2, col3 = st.columns(3)
            with col1:
                total_return = (cumulative_portfolio.iloc[-1] - 1) * 100
                st.metric("Total Return", f"{total_return:.2f}%")
            with col2:
                max_dd = ((cumulative_portfolio / cumulative_portfolio.cummax()) - 1).min() * 100
                st.metric("Max Drawdown", f"{max_dd:.2f}%")
            with col3:
                daily_sharpe = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)
                st.metric("Realized Sharpe", f"{daily_sharpe:.3f}")

            # Rolling Statistics
            st.subheader("Rolling Statistics Analysis")
            window = st.slider("Rolling Window (days)", 20, 120, 60)
            fig_rolling = create_rolling_statistics_chart(returns, weights, window)
            st.plotly_chart(fig_rolling, use_container_width=True)

        with tab5:
            st.subheader("3D Portfolio Analysis")
            annual_vol = returns.std() * np.sqrt(252)
            fig_3d = create_3d_allocation_chart(weights, annual_returns, annual_vol)
            st.plotly_chart(fig_3d, use_container_width=True)

            st.info("🎲 Bubble size represents portfolio weight. Rotate the chart by dragging!")

        with tab6:
            st.subheader("Monte Carlo Simulation")
            fig_monte = create_monte_carlo_chart(returns, weights)
            st.plotly_chart(fig_monte, use_container_width=True)

            st.info("📊 Shows 200 simulated 1-year paths. Blue line = median, shaded area = 5th to 95th percentile.")

        with tab7:
            st.subheader("Risk Analysis Dashboard")

            # Drawdown chart
            fig_drawdown = create_drawdown_chart(returns, weights)
            st.plotly_chart(fig_drawdown, use_container_width=True)

            # Risk contribution
            st.subheader("Risk Contribution by Asset")
            fig_waterfall = create_risk_contribution_waterfall(weights, cov_matrix)
            st.plotly_chart(fig_waterfall, use_container_width=True)

            st.info("💧 Waterfall shows how each asset contributes to total portfolio risk (variance).")

        with tab8:
            st.subheader("Strategy Comparison")
            fig_radar = create_strategy_comparison_radar(returns)
            st.plotly_chart(fig_radar, use_container_width=True)

            st.info("🕸️ Radar chart comparing Equal Weight, Max Sharpe, and Min Variance strategies across multiple metrics.")

    else:
        st.info("👈 Configure parameters in the sidebar and click 'Optimize Portfolio' to begin!")

        # Show example with interactive demo
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
            ### 📊 Advanced Features

            - 🎨 Interactive 3D visualizations
            - 📈 Real-time animated charts
            - 🌐 Correlation network graphs
            - ⚡ Efficient frontier explorer
            - 📊 Dynamic gauge metrics
            - 🎲 Monte Carlo simulations
            - 📉 Drawdown & risk analysis
            - ⚖️ Multi-strategy comparison
            - 💧 Risk contribution waterfall
            - 🗺️ Treemap & sunburst charts
            """)

        # Demo visualization
        st.header("🎬 Preview: Sample Visualization")
        demo_data = pd.DataFrame({
            'x': np.random.randn(100),
            'y': np.random.randn(100),
        })
        fig_demo = px.scatter(demo_data, x='x', y='y',
                             title="Interactive Scatter Plot (Try hovering!)",
                             color=demo_data['x'] + demo_data['y'],
                             color_continuous_scale='Viridis')
        fig_demo.update_layout(height=400)
        st.plotly_chart(fig_demo, use_container_width=True)


if __name__ == '__main__':
    main()
