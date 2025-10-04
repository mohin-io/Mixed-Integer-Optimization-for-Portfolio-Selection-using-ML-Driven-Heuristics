"""
ESG (Environmental, Social, Governance) Scoring Integration.

Implements:
- ESG data fetching from multiple sources
- ESG score calculation and normalization
- ESG-constrained portfolio optimization
- ESG factor integration with existing models
- Sustainable investing metrics
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import warnings

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    warnings.warn("yfinance not available. ESG data fetching will be limited.")


@dataclass
class ESGScore:
    """ESG score for a single asset."""
    ticker: str
    environmental_score: float  # 0-100
    social_score: float  # 0-100
    governance_score: float  # 0-100
    total_score: float  # 0-100
    controversy_score: Optional[float] = None
    data_source: str = "Unknown"


class ESGDataProvider:
    """
    Fetches and manages ESG data from various sources.
    """

    def __init__(self, data_source: str = 'yahoo'):
        """
        Initialize ESG data provider.

        Args:
            data_source: 'yahoo', 'msci', 'sustainalytics', or 'synthetic'
        """
        self.data_source = data_source

    def fetch_esg_scores(self, tickers: List[str]) -> Dict[str, ESGScore]:
        """
        Fetch ESG scores for given tickers.

        Args:
            tickers: List of stock tickers

        Returns:
            Dictionary mapping ticker to ESGScore
        """
        if self.data_source == 'yahoo' and YFINANCE_AVAILABLE:
            return self._fetch_yahoo_esg(tickers)
        else:
            # Use synthetic ESG scores for testing
            return self._generate_synthetic_esg(tickers)

    def _fetch_yahoo_esg(self, tickers: List[str]) -> Dict[str, ESGScore]:
        """Fetch ESG scores from Yahoo Finance."""
        esg_scores = {}

        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info

                # Yahoo Finance provides ESG scores
                if 'esgScores' in info and info['esgScores']:
                    esg_data = info['esgScores']

                    esg_scores[ticker] = ESGScore(
                        ticker=ticker,
                        environmental_score=esg_data.get('environmentScore', 50),
                        social_score=esg_data.get('socialScore', 50),
                        governance_score=esg_data.get('governanceScore', 50),
                        total_score=esg_data.get('totalEsg', 50),
                        controversy_score=esg_data.get('highestControversy', 0),
                        data_source='Yahoo Finance'
                    )
                else:
                    # Fallback to synthetic
                    esg_scores[ticker] = self._generate_synthetic_esg([ticker])[ticker]

            except Exception as e:
                warnings.warn(f"Failed to fetch ESG for {ticker}: {e}")
                esg_scores[ticker] = self._generate_synthetic_esg([ticker])[ticker]

        return esg_scores

    def _generate_synthetic_esg(self, tickers: List[str]) -> Dict[str, ESGScore]:
        """Generate synthetic ESG scores for testing."""
        np.random.seed(42)
        esg_scores = {}

        for ticker in tickers:
            # Generate scores with some correlation
            env = np.random.uniform(30, 90)
            soc = env + np.random.normal(0, 10)
            gov = (env + soc) / 2 + np.random.normal(0, 10)

            # Clip to valid range
            env = np.clip(env, 0, 100)
            soc = np.clip(soc, 0, 100)
            gov = np.clip(gov, 0, 100)

            total = (env + soc + gov) / 3

            esg_scores[ticker] = ESGScore(
                ticker=ticker,
                environmental_score=env,
                social_score=soc,
                governance_score=gov,
                total_score=total,
                controversy_score=np.random.uniform(0, 5),
                data_source='Synthetic'
            )

        return esg_scores


class ESGConstrainedOptimizer:
    """
    Portfolio optimizer with ESG constraints.
    """

    def __init__(
        self,
        min_esg_score: float = 50.0,
        esg_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize ESG-constrained optimizer.

        Args:
            min_esg_score: Minimum acceptable portfolio ESG score
            esg_weights: Weights for E, S, G components (default equal)
        """
        self.min_esg_score = min_esg_score

        if esg_weights is None:
            self.esg_weights = {'E': 1/3, 'S': 1/3, 'G': 1/3}
        else:
            self.esg_weights = esg_weights

    def compute_portfolio_esg(
        self,
        weights: np.ndarray,
        esg_scores: Dict[str, ESGScore],
        tickers: List[str]
    ) -> Dict[str, float]:
        """
        Compute portfolio-level ESG scores.

        Args:
            weights: Portfolio weights
            esg_scores: ESG scores for each asset
            tickers: List of tickers

        Returns:
            Dictionary with E, S, G, and total scores
        """
        env_score = 0
        soc_score = 0
        gov_score = 0

        for i, ticker in enumerate(tickers):
            if ticker in esg_scores:
                esg = esg_scores[ticker]
                env_score += weights[i] * esg.environmental_score
                soc_score += weights[i] * esg.social_score
                gov_score += weights[i] * esg.governance_score

        total_score = (
            self.esg_weights['E'] * env_score +
            self.esg_weights['S'] * soc_score +
            self.esg_weights['G'] * gov_score
        )

        return {
            'environmental': env_score,
            'social': soc_score,
            'governance': gov_score,
            'total': total_score
        }

    def filter_by_esg(
        self,
        tickers: List[str],
        esg_scores: Dict[str, ESGScore],
        min_score: Optional[float] = None
    ) -> List[str]:
        """
        Filter assets based on minimum ESG score.

        Args:
            tickers: List of all tickers
            esg_scores: ESG scores
            min_score: Minimum ESG score (uses self.min_esg_score if None)

        Returns:
            Filtered list of tickers
        """
        if min_score is None:
            min_score = self.min_esg_score

        filtered = [
            ticker for ticker in tickers
            if ticker in esg_scores and esg_scores[ticker].total_score >= min_score
        ]

        return filtered

    def optimize_with_esg_constraint(
        self,
        expected_returns: pd.Series,
        covariance: pd.DataFrame,
        esg_scores: Dict[str, ESGScore],
        risk_aversion: float = 2.5
    ) -> Dict:
        """
        Optimize portfolio with ESG constraints.

        Uses a penalty method: penalize portfolios with low ESG scores.

        Args:
            expected_returns: Expected returns
            covariance: Covariance matrix
            esg_scores: ESG scores
            risk_aversion: Risk aversion parameter

        Returns:
            Optimization results with ESG metrics
        """
        from scipy.optimize import minimize

        n_assets = len(expected_returns)
        tickers = expected_returns.index.tolist()

        # Objective function with ESG penalty
        def objective(weights):
            # Portfolio return
            ret = np.dot(weights, expected_returns.values)

            # Portfolio risk
            risk = weights @ covariance.values @ weights

            # ESG score
            portfolio_esg = self.compute_portfolio_esg(weights, esg_scores, tickers)

            # ESG penalty (if below threshold)
            esg_penalty = 0
            if portfolio_esg['total'] < self.min_esg_score:
                esg_penalty = 100 * (self.min_esg_score - portfolio_esg['total']) ** 2

            # Combined objective
            objective_value = -(ret - risk_aversion * risk) + esg_penalty

            return objective_value

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
            constraints=constraints
        )

        optimal_weights = result.x / result.x.sum()

        # Compute metrics
        portfolio_esg = self.compute_portfolio_esg(optimal_weights, esg_scores, tickers)
        portfolio_return = np.dot(optimal_weights, expected_returns.values)
        portfolio_variance = optimal_weights @ covariance.values @ optimal_weights
        portfolio_volatility = np.sqrt(portfolio_variance)

        return {
            'weights': pd.Series(optimal_weights, index=tickers),
            'expected_return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0,
            'esg_scores': portfolio_esg,
            'optimization_status': result.message
        }


class ESGFactor:
    """
    ESG as a factor in multi-factor models.
    """

    def __init__(self):
        """Initialize ESG factor model."""
        pass

    def create_esg_factor(
        self,
        returns: pd.DataFrame,
        esg_scores: Dict[str, ESGScore],
        method: str = 'long_short'
    ) -> pd.Series:
        """
        Create ESG factor returns (high ESG - low ESG).

        Args:
            returns: Asset returns
            esg_scores: ESG scores
            method: 'long_short' or 'weighted'

        Returns:
            ESG factor return series
        """
        tickers = returns.columns.tolist()

        # Get ESG scores for all assets
        scores = np.array([
            esg_scores[t].total_score if t in esg_scores else 50
            for t in tickers
        ])

        if method == 'long_short':
            # Long top 30%, short bottom 30%
            top_30_pct = np.percentile(scores, 70)
            bottom_30_pct = np.percentile(scores, 30)

            long_mask = scores >= top_30_pct
            short_mask = scores <= bottom_30_pct

            # Equal weight within each group
            long_weights = long_mask / long_mask.sum() if long_mask.sum() > 0 else 0
            short_weights = short_mask / short_mask.sum() if short_mask.sum() > 0 else 0

            # Factor returns: long - short
            factor_returns = (returns @ long_weights) - (returns @ short_weights)

        elif method == 'weighted':
            # Weight by ESG score
            weights = scores / scores.sum()
            factor_returns = returns @ weights

        else:
            raise ValueError(f"Unknown method: {method}")

        return factor_returns

    def estimate_esg_beta(
        self,
        asset_returns: pd.Series,
        esg_factor: pd.Series
    ) -> float:
        """
        Estimate beta (loading) on ESG factor.

        Args:
            asset_returns: Returns for a single asset
            esg_factor: ESG factor returns

        Returns:
            ESG beta
        """
        from sklearn.linear_model import LinearRegression

        # Align data
        common_idx = asset_returns.index.intersection(esg_factor.index)
        y = asset_returns.loc[common_idx].values.reshape(-1, 1)
        X = esg_factor.loc[common_idx].values.reshape(-1, 1)

        # Regression
        model = LinearRegression()
        model.fit(X, y)

        return model.coef_[0][0]


class SustainableInvestingMetrics:
    """
    Calculate sustainable investing metrics.
    """

    @staticmethod
    def carbon_footprint(
        weights: np.ndarray,
        carbon_intensity: Dict[str, float],
        tickers: List[str]
    ) -> float:
        """
        Calculate portfolio carbon footprint.

        Args:
            weights: Portfolio weights
            carbon_intensity: Carbon intensity per ticker (tons CO2 / $M revenue)
            tickers: List of tickers

        Returns:
            Weighted average carbon intensity
        """
        footprint = 0
        for i, ticker in enumerate(tickers):
            if ticker in carbon_intensity:
                footprint += weights[i] * carbon_intensity[ticker]

        return footprint

    @staticmethod
    def esg_momentum(
        esg_scores_current: Dict[str, ESGScore],
        esg_scores_previous: Dict[str, ESGScore]
    ) -> Dict[str, float]:
        """
        Calculate ESG score momentum (improvement/decline).

        Args:
            esg_scores_current: Current ESG scores
            esg_scores_previous: Previous ESG scores

        Returns:
            Dictionary of ESG momentum scores
        """
        momentum = {}

        for ticker in esg_scores_current:
            if ticker in esg_scores_previous:
                current = esg_scores_current[ticker].total_score
                previous = esg_scores_previous[ticker].total_score
                momentum[ticker] = current - previous

        return momentum

    @staticmethod
    def sdg_alignment(
        tickers: List[str],
        sdg_mapping: Dict[str, List[int]]
    ) -> Dict[int, List[str]]:
        """
        Map portfolio to UN Sustainable Development Goals.

        Args:
            tickers: Portfolio tickers
            sdg_mapping: Mapping of ticker to SDG numbers (1-17)

        Returns:
            Dictionary mapping SDG to contributing tickers
        """
        sdg_portfolio = {}

        for ticker in tickers:
            if ticker in sdg_mapping:
                for sdg in sdg_mapping[ticker]:
                    if sdg not in sdg_portfolio:
                        sdg_portfolio[sdg] = []
                    sdg_portfolio[sdg].append(ticker)

        return sdg_portfolio


if __name__ == "__main__":
    print("Testing ESG Scoring Integration...")

    # Test ESG data provider
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'XOM']
    provider = ESGDataProvider(data_source='synthetic')
    esg_scores = provider.fetch_esg_scores(tickers)

    print(f"\nESG Scores:")
    for ticker, score in esg_scores.items():
        print(f"  {ticker}: Total={score.total_score:.1f}, "
              f"E={score.environmental_score:.1f}, "
              f"S={score.social_score:.1f}, "
              f"G={score.governance_score:.1f}")

    # Test ESG-constrained optimization
    np.random.seed(42)
    expected_returns = pd.Series(
        np.random.uniform(0.08, 0.15, len(tickers)),
        index=tickers
    )

    corr = np.eye(len(tickers))
    for i in range(len(tickers)):
        for j in range(i+1, len(tickers)):
            corr[i, j] = corr[j, i] = 0.3

    volatilities = np.random.uniform(0.15, 0.30, len(tickers))
    covariance = pd.DataFrame(
        np.outer(volatilities, volatilities) * corr,
        index=tickers,
        columns=tickers
    )

    optimizer = ESGConstrainedOptimizer(min_esg_score=60.0)
    result = optimizer.optimize_with_esg_constraint(
        expected_returns,
        covariance,
        esg_scores
    )

    print(f"\nESG-Constrained Portfolio:")
    print(f"  Weights: {result['weights'].to_dict()}")
    print(f"  Expected Return: {result['expected_return']:.4f}")
    print(f"  Volatility: {result['volatility']:.4f}")
    print(f"  Sharpe Ratio: {result['sharpe_ratio']:.4f}")
    print(f"  ESG Scores: {result['esg_scores']}")

    # Test ESG filtering
    filtered = optimizer.filter_by_esg(tickers, esg_scores, min_score=65)
    print(f"\nFiltered tickers (ESG >= 65): {filtered}")

    # Test sustainable metrics
    weights = result['weights'].values
    carbon_intensity = {t: np.random.uniform(50, 200) for t in tickers}

    footprint = SustainableInvestingMetrics.carbon_footprint(
        weights,
        carbon_intensity,
        tickers
    )
    print(f"\nCarbon Footprint: {footprint:.2f} tons CO2/$M revenue")

    print("\n✅ ESG scoring integration complete!")
