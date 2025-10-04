"""
Tail Risk Hedging Strategies for Portfolio Protection.

Implements:
- Black Swan protection strategies
- Put option overlay strategies
- VIX-based hedging
- Tail-risk parity
- Dynamic hedging based on market volatility
- Extreme value theory (EVT) for tail estimation
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from scipy import stats
from scipy.optimize import minimize
import warnings


@dataclass
class HedgeConfig:
    """Configuration for tail risk hedging."""
    hedge_ratio: float = 0.05  # Percentage of portfolio to allocate to hedge
    rehedge_threshold: float = 0.02  # Trigger rehedge when ratio drifts
    var_confidence: float = 0.99  # VaR confidence level
    lookback_window: int = 252  # Days for volatility estimation
    max_hedge_cost: float = 0.02  # Maximum annual cost for hedging (2%)


class TailRiskEstimator:
    """
    Estimate tail risk using Extreme Value Theory (EVT).
    """

    def __init__(self, confidence_level: float = 0.99):
        """
        Initialize tail risk estimator.

        Args:
            confidence_level: Confidence level for tail risk (e.g., 0.99 for 99%)
        """
        self.confidence_level = confidence_level

    def estimate_var_cvar_evt(
        self,
        returns: np.ndarray,
        confidence_level: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Estimate VaR and CVaR using Extreme Value Theory (Peaks Over Threshold).

        Args:
            returns: Historical returns
            confidence_level: Override default confidence level

        Returns:
            Tuple of (VaR, CVaR)
        """
        if confidence_level is None:
            confidence_level = self.confidence_level

        # Use negative returns (losses)
        losses = -returns

        # Select threshold (e.g., 90th percentile)
        threshold = np.percentile(losses, 90)

        # Get exceedances
        exceedances = losses[losses > threshold] - threshold

        if len(exceedances) < 10:
            # Fall back to empirical VaR/CVaR
            var = np.percentile(losses, confidence_level * 100)
            cvar = np.mean(losses[losses >= var])
            return var, cvar

        # Fit Generalized Pareto Distribution (GPD)
        shape, loc, scale = stats.genpareto.fit(exceedances)

        # Calculate VaR using GPD
        n = len(losses)
        n_u = len(exceedances)
        p = confidence_level

        var = threshold + (scale / shape) * (
            ((n / n_u) * (1 - p)) ** (-shape) - 1
        )

        # Calculate CVaR
        cvar = (var + scale - shape * threshold) / (1 - shape)

        return var, cvar

    def estimate_tail_index(self, returns: np.ndarray) -> float:
        """
        Estimate tail index (higher = fatter tails).

        Uses Hill estimator.

        Args:
            returns: Historical returns

        Returns:
            Tail index estimate
        """
        losses = -np.sort(-returns)  # Sort in descending order
        k = int(len(losses) * 0.1)  # Use top 10%

        if k < 2:
            return 3.0  # Default

        # Hill estimator
        tail_index = k / np.sum(np.log(losses[:k] / losses[k]))

        return tail_index


class PutOptionHedge:
    """
    Put option overlay strategy for tail risk hedging.
    """

    def __init__(
        self,
        hedge_ratio: float = 0.05,
        strike_percentage: float = 0.95,
        option_maturity_days: int = 30
    ):
        """
        Initialize put option hedge.

        Args:
            hedge_ratio: Percentage of portfolio to hedge
            strike_percentage: Strike as % of current price (0.95 = 5% OTM)
            option_maturity_days: Days to option expiration
        """
        self.hedge_ratio = hedge_ratio
        self.strike_percentage = strike_percentage
        self.option_maturity_days = option_maturity_days

    def calculate_black_scholes_put(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float
    ) -> float:
        """
        Black-Scholes put option pricing.

        Args:
            S: Current price
            K: Strike price
            T: Time to maturity (years)
            r: Risk-free rate
            sigma: Volatility

        Returns:
            Put option price
        """
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        put_price = K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)

        return put_price

    def design_put_hedge(
        self,
        portfolio_value: float,
        current_price: float,
        volatility: float,
        risk_free_rate: float = 0.02
    ) -> Dict:
        """
        Design put option hedge.

        Args:
            portfolio_value: Current portfolio value
            current_price: Current index/asset price
            volatility: Implied volatility
            risk_free_rate: Risk-free rate

        Returns:
            Hedge details dictionary
        """
        # Calculate strike price
        strike = current_price * self.strike_percentage

        # Time to maturity in years
        T = self.option_maturity_days / 365.0

        # Calculate put price
        put_price = self.calculate_black_scholes_put(
            S=current_price,
            K=strike,
            T=T,
            r=risk_free_rate,
            sigma=volatility
        )

        # Number of puts needed
        hedge_value = portfolio_value * self.hedge_ratio
        notional_per_put = current_price  # Assuming 1:1 multiplier

        num_puts = hedge_value / notional_per_put
        total_cost = num_puts * put_price

        # Cost as percentage of portfolio
        hedge_cost_pct = total_cost / portfolio_value

        return {
            'num_puts': num_puts,
            'strike': strike,
            'put_price': put_price,
            'total_cost': total_cost,
            'hedge_cost_pct': hedge_cost_pct,
            'notional_hedged': hedge_value,
            'maturity_days': self.option_maturity_days
        }


class VIXHedge:
    """
    VIX-based hedging strategy.
    """

    def __init__(self, target_vix_exposure: float = 0.10):
        """
        Initialize VIX hedge.

        Args:
            target_vix_exposure: Target VIX exposure as % of portfolio
        """
        self.target_vix_exposure = target_vix_exposure

    def calculate_vix_hedge_ratio(
        self,
        portfolio_beta: float,
        vix_level: float,
        vix_historical_mean: float = 15.0
    ) -> float:
        """
        Calculate optimal VIX hedge ratio.

        Args:
            portfolio_beta: Portfolio beta to market
            vix_level: Current VIX level
            vix_historical_mean: Historical VIX mean

        Returns:
            Recommended VIX exposure
        """
        # Higher VIX = lower hedge needed (expensive)
        # Lower VIX = higher hedge (cheap insurance)

        vix_ratio = vix_level / vix_historical_mean

        # Adjust target exposure inversely with VIX
        adjusted_exposure = self.target_vix_exposure / vix_ratio

        # Scale by portfolio beta
        hedge_ratio = adjusted_exposure * portfolio_beta

        # Cap at maximum
        hedge_ratio = min(hedge_ratio, 0.20)  # Max 20%

        return hedge_ratio


class TailRiskParity:
    """
    Tail-Risk Parity: Equalize contribution to tail risk.
    """

    def __init__(self, confidence_level: float = 0.99):
        """
        Initialize tail-risk parity.

        Args:
            confidence_level: Confidence level for CVaR
        """
        self.confidence_level = confidence_level

    def calculate_cvar_contribution(
        self,
        weights: np.ndarray,
        returns: pd.DataFrame
    ) -> np.ndarray:
        """
        Calculate each asset's contribution to portfolio CVaR.

        Args:
            weights: Portfolio weights
            returns: Asset returns

        Returns:
            Array of CVaR contributions
        """
        n_assets = len(weights)
        contributions = np.zeros(n_assets)

        # Portfolio returns
        portfolio_returns = returns @ weights

        # VaR threshold
        var = np.percentile(portfolio_returns, (1 - self.confidence_level) * 100)

        # CVaR
        tail_returns = portfolio_returns[portfolio_returns <= var]

        if len(tail_returns) == 0:
            return contributions

        # Marginal contribution to CVaR
        for i in range(n_assets):
            asset_tail_returns = returns.iloc[:, i].values[portfolio_returns <= var]
            contributions[i] = weights[i] * np.mean(asset_tail_returns)

        return contributions

    def optimize_tail_risk_parity(
        self,
        returns: pd.DataFrame,
        max_iterations: int = 100
    ) -> np.ndarray:
        """
        Optimize for tail-risk parity (equal CVaR contributions).

        Args:
            returns: Historical returns
            max_iterations: Maximum optimization iterations

        Returns:
            Optimal weights
        """
        n_assets = returns.shape[1]

        # Objective: minimize variance of CVaR contributions
        def objective(weights):
            contributions = self.calculate_cvar_contribution(weights, returns)
            return np.std(contributions)

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Budget
        ]

        # Bounds
        bounds = [(0, 1) for _ in range(n_assets)]

        # Initial guess
        x0 = np.ones(n_assets) / n_assets

        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': max_iterations}
        )

        weights = result.x / result.x.sum()

        return weights


class DynamicHedging:
    """
    Dynamic hedging that adjusts based on market conditions.
    """

    def __init__(self, config: HedgeConfig):
        """
        Initialize dynamic hedging.

        Args:
            config: Hedge configuration
        """
        self.config = config
        self.current_hedge_ratio = 0.0

    def calculate_volatility_regime(
        self,
        returns: np.ndarray,
        short_window: int = 20,
        long_window: int = 60
    ) -> str:
        """
        Classify current volatility regime.

        Args:
            returns: Recent returns
            short_window: Short-term window
            long_window: Long-term window

        Returns:
            Regime: 'low', 'normal', 'high', 'crisis'
        """
        if len(returns) < long_window:
            return 'normal'

        # Calculate volatilities
        recent_vol = np.std(returns[-short_window:]) * np.sqrt(252)
        long_vol = np.std(returns[-long_window:]) * np.sqrt(252)

        # Regime classification
        if recent_vol < long_vol * 0.7:
            return 'low'
        elif recent_vol < long_vol * 1.3:
            return 'normal'
        elif recent_vol < long_vol * 2.0:
            return 'high'
        else:
            return 'crisis'

    def determine_hedge_ratio(
        self,
        returns: np.ndarray,
        portfolio_value: float
    ) -> Dict:
        """
        Determine optimal hedge ratio based on market conditions.

        Args:
            returns: Historical returns
            portfolio_value: Current portfolio value

        Returns:
            Hedge decision dictionary
        """
        # Classify regime
        regime = self.calculate_volatility_regime(returns)

        # Regime-dependent hedge ratios
        regime_ratios = {
            'low': 0.02,      # Minimal hedging when calm
            'normal': 0.05,    # Standard hedging
            'high': 0.10,      # Increased hedging
            'crisis': 0.15     # Maximum hedging
        }

        target_ratio = regime_ratios[regime]

        # Check if rehedge needed
        drift = abs(target_ratio - self.current_hedge_ratio)
        needs_rehedge = drift > self.config.rehedge_threshold

        decision = {
            'regime': regime,
            'target_hedge_ratio': target_ratio,
            'current_hedge_ratio': self.current_hedge_ratio,
            'needs_rehedge': needs_rehedge,
            'drift': drift,
            'hedge_value': portfolio_value * target_ratio
        }

        if needs_rehedge:
            self.current_hedge_ratio = target_ratio

        return decision


class ComprehensiveTailRiskStrategy:
    """
    Comprehensive tail risk management combining multiple strategies.
    """

    def __init__(self, config: HedgeConfig):
        """
        Initialize comprehensive strategy.

        Args:
            config: Hedge configuration
        """
        self.config = config
        self.tail_estimator = TailRiskEstimator()
        self.put_hedge = PutOptionHedge(hedge_ratio=config.hedge_ratio)
        self.vix_hedge = VIXHedge()
        self.dynamic_hedge = DynamicHedging(config)

    def analyze_tail_risk(
        self,
        returns: pd.DataFrame,
        portfolio_weights: np.ndarray
    ) -> Dict:
        """
        Comprehensive tail risk analysis.

        Args:
            returns: Historical returns
            portfolio_weights: Current portfolio weights

        Returns:
            Analysis results
        """
        # Portfolio returns
        portfolio_returns = (returns @ portfolio_weights).values

        # Tail risk metrics
        var, cvar = self.tail_estimator.estimate_var_cvar_evt(portfolio_returns)
        tail_index = self.tail_estimator.estimate_tail_index(portfolio_returns)

        # Regime analysis
        regime = self.dynamic_hedge.calculate_volatility_regime(portfolio_returns)

        # Current volatility
        current_vol = np.std(portfolio_returns[-60:]) * np.sqrt(252)

        return {
            'var_99': var,
            'cvar_99': cvar,
            'tail_index': tail_index,
            'volatility_regime': regime,
            'current_volatility': current_vol,
            'annualized_volatility': current_vol
        }

    def recommend_hedge(
        self,
        portfolio_value: float,
        returns: pd.DataFrame,
        portfolio_weights: np.ndarray,
        current_price: float,
        vix_level: float = 15.0
    ) -> Dict:
        """
        Recommend comprehensive hedging strategy.

        Args:
            portfolio_value: Portfolio value
            returns: Historical returns
            portfolio_weights: Portfolio weights
            current_price: Current index price
            vix_level: Current VIX level

        Returns:
            Hedge recommendations
        """
        # Analyze tail risk
        tail_analysis = self.analyze_tail_risk(returns, portfolio_weights)

        # Dynamic hedge ratio
        portfolio_returns = (returns @ portfolio_weights).values
        hedge_decision = self.dynamic_hedge.determine_hedge_ratio(
            portfolio_returns,
            portfolio_value
        )

        # Put option hedge
        put_design = self.put_hedge.design_put_hedge(
            portfolio_value,
            current_price,
            volatility=tail_analysis['current_volatility']
        )

        # VIX hedge
        portfolio_beta = 1.0  # Simplified
        vix_ratio = self.vix_hedge.calculate_vix_hedge_ratio(
            portfolio_beta,
            vix_level
        )

        # Combined recommendation
        recommendation = {
            'tail_analysis': tail_analysis,
            'hedge_decision': hedge_decision,
            'put_hedge': put_design,
            'vix_hedge_ratio': vix_ratio,
            'total_hedge_cost': put_design['total_cost'],
            'hedge_cost_pct': put_design['hedge_cost_pct'],
            'action_required': hedge_decision['needs_rehedge']
        }

        return recommendation


if __name__ == "__main__":
    print("Testing Tail Risk Hedging Strategies...")

    # Generate synthetic returns
    np.random.seed(42)
    n_periods = 1000
    n_assets = 5

    # Generate returns with fat tails
    returns = np.random.standard_t(df=4, size=(n_periods, n_assets)) * 0.01

    returns_df = pd.DataFrame(
        returns,
        columns=[f'Asset_{i}' for i in range(n_assets)]
    )

    portfolio_weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

    print("\n1. Tail Risk Estimation (EVT)")
    estimator = TailRiskEstimator(confidence_level=0.99)
    portfolio_returns = (returns_df @ portfolio_weights).values

    var, cvar = estimator.estimate_var_cvar_evt(portfolio_returns)
    tail_index = estimator.estimate_tail_index(portfolio_returns)

    print(f"   VaR (99%): {var:.4f}")
    print(f"   CVaR (99%): {cvar:.4f}")
    print(f"   Tail Index: {tail_index:.2f}")

    print("\n2. Put Option Hedge Design")
    put_hedge = PutOptionHedge(hedge_ratio=0.05)
    put_design = put_hedge.design_put_hedge(
        portfolio_value=1000000,
        current_price=100,
        volatility=0.20
    )

    print(f"   Number of Puts: {put_design['num_puts']:.0f}")
    print(f"   Strike: ${put_design['strike']:.2f}")
    print(f"   Total Cost: ${put_design['total_cost']:,.2f}")
    print(f"   Cost %: {put_design['hedge_cost_pct']*100:.2f}%")

    print("\n3. Dynamic Hedging Decision")
    config = HedgeConfig(hedge_ratio=0.05, rehedge_threshold=0.02)
    dynamic = DynamicHedging(config)

    decision = dynamic.determine_hedge_ratio(portfolio_returns, 1000000)
    print(f"   Volatility Regime: {decision['regime']}")
    print(f"   Target Hedge Ratio: {decision['target_hedge_ratio']*100:.1f}%")
    print(f"   Needs Rehedge: {decision['needs_rehedge']}")

    print("\n4. Comprehensive Strategy")
    strategy = ComprehensiveTailRiskStrategy(config)

    recommendation = strategy.recommend_hedge(
        portfolio_value=1000000,
        returns=returns_df,
        portfolio_weights=portfolio_weights,
        current_price=100,
        vix_level=15.0
    )

    print(f"   Regime: {recommendation['tail_analysis']['volatility_regime']}")
    print(f"   CVaR (99%): {recommendation['tail_analysis']['cvar_99']:.4f}")
    print(f"   Hedge Cost: ${recommendation['total_hedge_cost']:,.2f}")
    print(f"   Action Required: {recommendation['action_required']}")

    print("\n✅ Tail risk hedging implementation complete!")
