"""
Asset Clustering Module

Pre-selects diverse assets using clustering algorithms:
- K-Means clustering
- Hierarchical clustering
- Reduces problem size for optimization
"""

import pandas as pd
import numpy as np
from typing import List, Tuple
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform


class AssetClusterer:
    """
    Clusters assets based on correlation/risk to enable diverse subset selection.
    """

    def __init__(self, method: str = 'kmeans'):
        """
        Initialize asset clusterer.

        Args:
            method: Clustering method ('kmeans', 'hierarchical')
        """
        self.method = method
        self.model = None
        self.labels_ = None

    def fit(
        self,
        returns: pd.DataFrame,
        n_clusters: int = 5,
        linkage_method: str = 'ward'
    ):
        """
        Fit clustering model on asset returns.

        Args:
            returns: DataFrame of asset returns
            n_clusters: Number of clusters
            linkage_method: Linkage method for hierarchical clustering
        """
        # Use correlation as distance metric
        corr_matrix = returns.corr()
        # Distance = 1 - |correlation|
        distance_matrix = 1 - corr_matrix.abs()

        if self.method == 'kmeans':
            # K-Means on returns features
            features = returns.T.values  # Each asset is a sample

            self.model = KMeans(
                n_clusters=n_clusters,
                random_state=42,
                n_init=10
            )
            self.labels_ = self.model.fit_predict(features)

        elif self.method == 'hierarchical':
            # Hierarchical clustering
            # Convert distance matrix to condensed form
            condensed_dist = squareform(distance_matrix.values, checks=False)

            self.model = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage=linkage_method
            )

            self.labels_ = self.model.fit_predict(distance_matrix.values)

        else:
            raise ValueError(f"Unknown clustering method: {self.method}")

        return self

    def select_representatives(
        self,
        returns: pd.DataFrame,
        n_per_cluster: int = 2,
        selection_criterion: str = 'sharpe'
    ) -> List[str]:
        """
        Select representative assets from each cluster.

        Args:
            returns: DataFrame of asset returns
            n_per_cluster: Number of assets to select per cluster
            selection_criterion: How to rank assets ('sharpe', 'return', 'volatility')

        Returns:
            List of selected asset tickers
        """
        if self.labels_ is None:
            raise RuntimeError("Must fit clusterer before selection")

        assets = returns.columns.tolist()
        selected = []

        # Group assets by cluster
        clusters = {}
        for i, label in enumerate(self.labels_):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(assets[i])

        # Select top assets from each cluster
        for cluster_id, cluster_assets in clusters.items():
            cluster_returns = returns[cluster_assets]

            # Rank assets based on criterion
            if selection_criterion == 'sharpe':
                mean_ret = cluster_returns.mean() * 252
                std_ret = cluster_returns.std() * np.sqrt(252)
                scores = mean_ret / std_ret
            elif selection_criterion == 'return':
                scores = cluster_returns.mean() * 252
            elif selection_criterion == 'volatility':
                scores = -cluster_returns.std() * np.sqrt(252)  # Negative for ascending sort
            else:
                raise ValueError(f"Unknown criterion: {selection_criterion}")

            # Select top n assets
            top_assets = scores.nlargest(min(n_per_cluster, len(cluster_assets))).index.tolist()
            selected.extend(top_assets)

        print(f"Selected {len(selected)} assets from {len(clusters)} clusters")

        return selected

    def get_cluster_summary(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Get summary statistics for each cluster.

        Args:
            returns: DataFrame of asset returns

        Returns:
            DataFrame with cluster statistics
        """
        if self.labels_ is None:
            raise RuntimeError("Must fit clusterer first")

        assets = returns.columns.tolist()
        summary = []

        for cluster_id in np.unique(self.labels_):
            cluster_mask = self.labels_ == cluster_id
            cluster_assets = [assets[i] for i, m in enumerate(cluster_mask) if m]
            cluster_returns = returns[cluster_assets]

            summary.append({
                'cluster_id': cluster_id,
                'n_assets': len(cluster_assets),
                'mean_return': cluster_returns.mean().mean() * 252,
                'mean_volatility': cluster_returns.std().mean() * np.sqrt(252),
                'avg_correlation': cluster_returns.corr().values[np.triu_indices_from(
                    cluster_returns.corr().values, k=1
                )].mean() if len(cluster_assets) > 1 else 1.0,
                'assets': ', '.join(cluster_assets[:3]) + ('...' if len(cluster_assets) > 3 else '')
            })

        return pd.DataFrame(summary)


def compute_linkage_matrix(returns: pd.DataFrame, method: str = 'ward') -> np.ndarray:
    """
    Compute hierarchical clustering linkage matrix for dendrogram.

    Args:
        returns: Asset returns
        method: Linkage method

    Returns:
        Linkage matrix
    """
    corr_matrix = returns.corr()
    distance_matrix = 1 - corr_matrix.abs()
    condensed_dist = squareform(distance_matrix.values, checks=False)

    Z = linkage(condensed_dist, method=method)

    return Z


if __name__ == "__main__":
    # Example usage
    from src.data.loader import AssetDataLoader

    # Load data
    loader = AssetDataLoader()
    tickers = loader._get_default_tickers()[:20]  # Use 20 assets for demo
    prices = loader.fetch_prices(tickers, '2020-01-01', '2023-12-31')
    returns = loader.compute_returns(prices)

    print("=== Asset Clustering ===\n")

    # K-Means clustering
    print("1. K-Means Clustering")
    kmeans_clusterer = AssetClusterer(method='kmeans')
    kmeans_clusterer.fit(returns, n_clusters=5)

    summary = kmeans_clusterer.get_cluster_summary(returns)
    print(summary.to_string(index=False))

    # Select representatives
    selected = kmeans_clusterer.select_representatives(returns, n_per_cluster=2)
    print(f"\nSelected assets: {selected}")

    # Hierarchical clustering
    print("\n2. Hierarchical Clustering")
    hier_clusterer = AssetClusterer(method='hierarchical')
    hier_clusterer.fit(returns, n_clusters=5)

    summary = hier_clusterer.get_cluster_summary(returns)
    print(summary.to_string(index=False))
