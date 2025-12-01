"""
Asset Clustering Engine for Hierarchical Multi-Asset Training.

This module implements automatic asset clustering based on correlation structure,
enabling hierarchical training where dominant clusters (Majors) inject market
signals into subordinate clusters (Alts).

Problem:
--------
- Global training fails on Altcoins (different physics than BTC)
- Isolated training fails (Alts depend on BTC for context)

Solution:
---------
- Auto-cluster assets by correlation structure
- Identify dominant cluster (contains BTC or highest volume)
- Extract market factor from dominant cluster
- Inject as feature to subordinate clusters

Result: "Majors lead, Alts follow (but have their own brain)"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA


@dataclass
class ClusterResult:
    """
    Result of asset clustering.
    
    Attributes
    ----------
    cluster_map : dict
        Mapping of asset symbol to cluster ID
    dominant_cluster_id : int
        ID of the dominant cluster (contains BTC or highest volume)
    cluster_members : dict
        Mapping of cluster ID to list of asset symbols
    linkage_matrix : ndarray
        Hierarchical clustering linkage matrix
    correlation_matrix : pd.DataFrame
        Correlation matrix used for clustering
    """
    cluster_map: Dict[str, int]
    dominant_cluster_id: int
    cluster_members: Dict[int, List[str]]
    linkage_matrix: np.ndarray
    correlation_matrix: pd.DataFrame


class AssetClusterer:
    """
    Automatic asset clustering based on correlation structure.
    
    Uses hierarchical clustering (Ward's method) on correlation distance
    to group assets with similar market behavior.
    
    Parameters
    ----------
    n_clusters : int, optional
        Number of clusters to form. If None, uses dynamic selection.
    distance_threshold : float, optional
        Distance threshold for dynamic clustering (if n_clusters is None)
    method : str, default='ward'
        Linkage method for hierarchical clustering
    feature_column : str, default='frac_diff'
        Column to use for correlation calculation
    min_overlap : int, default=100
        Minimum number of overlapping timestamps required
    """
    
    def __init__(
        self,
        n_clusters: Optional[int] = None,
        distance_threshold: Optional[float] = None,
        method: str = 'ward',
        feature_column: str = 'frac_diff',
        min_overlap: int = 100,
    ):
        self.n_clusters = n_clusters
        self.distance_threshold = distance_threshold
        self.method = method
        self.feature_column = feature_column
        self.min_overlap = min_overlap
        
        # Default to 3 clusters if neither n_clusters nor distance_threshold specified
        if self.n_clusters is None and self.distance_threshold is None:
            self.n_clusters = 3
    
    def _align_timestamps(
        self,
        data_dict: Dict[str, pd.DataFrame],
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Align timestamps across all assets and extract feature matrix.
        
        Parameters
        ----------
        data_dict : dict
            Dictionary mapping asset symbol to DataFrame
        
        Returns
        -------
        feature_matrix : pd.DataFrame
            Aligned feature matrix with assets as columns
        symbols : list
            List of asset symbols (column order)
        """
        # Extract feature series for each asset
        series_dict = {}
        
        for symbol, df in data_dict.items():
            if self.feature_column not in df.columns:
                print(f"  [Warning] {self.feature_column} not found in {symbol}, skipping")
                continue
            
            # Get feature series with timestamp index
            series = df[self.feature_column].copy()
            
            # Ensure index is datetime
            if not isinstance(series.index, pd.DatetimeIndex):
                if 'timestamp' in df.columns:
                    series.index = pd.to_datetime(df['timestamp'])
                else:
                    print(f"  [Warning] No timestamp index for {symbol}, skipping")
                    continue
            
            series_dict[symbol] = series
        
        if len(series_dict) < 2:
            raise ValueError("Need at least 2 assets for clustering")
        
        # Combine into DataFrame (outer join to get all timestamps)
        feature_matrix = pd.DataFrame(series_dict)
        
        # Drop rows with too many NaNs
        # Keep rows where at least 50% of assets have data
        min_valid = max(2, len(series_dict) // 2)
        feature_matrix = feature_matrix.dropna(thresh=min_valid)
        
        # Forward fill small gaps (max 3 periods)
        feature_matrix = feature_matrix.fillna(method='ffill', limit=3)
        
        # Drop remaining NaNs
        feature_matrix = feature_matrix.dropna()
        
        if len(feature_matrix) < self.min_overlap:
            raise ValueError(
                f"Insufficient overlap: {len(feature_matrix)} timestamps "
                f"(need at least {self.min_overlap})"
            )
        
        symbols = list(feature_matrix.columns)
        
        print(f"  [Alignment] {len(feature_matrix)} timestamps across {len(symbols)} assets")
        
        return feature_matrix, symbols
    
    def _compute_correlation_matrix(
        self,
        feature_matrix: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute correlation matrix from feature matrix.
        
        Parameters
        ----------
        feature_matrix : pd.DataFrame
            Aligned feature matrix
        
        Returns
        -------
        corr_matrix : pd.DataFrame
            Correlation matrix
        """
        corr_matrix = feature_matrix.corr(method='pearson')
        
        # Handle any remaining NaNs (replace with 0)
        corr_matrix = corr_matrix.fillna(0)
        
        # Ensure diagonal is 1.0
        np.fill_diagonal(corr_matrix.values, 1.0)
        
        return corr_matrix
    
    def _identify_dominant_cluster(
        self,
        cluster_members: Dict[int, List[str]],
        data_dict: Dict[str, pd.DataFrame],
    ) -> int:
        """
        Identify the dominant cluster.
        
        Dominant cluster is defined as:
        1. The cluster containing BTC (if present)
        2. Otherwise, the cluster with highest total volume
        
        Parameters
        ----------
        cluster_members : dict
            Mapping of cluster ID to list of symbols
        data_dict : dict
            Dictionary mapping asset symbol to DataFrame
        
        Returns
        -------
        dominant_cluster_id : int
            ID of the dominant cluster
        """
        # Check if BTC is in any cluster
        for cluster_id, members in cluster_members.items():
            if 'BTCUSDT' in members or 'BTC' in members:
                print(f"  [Dominant] Cluster {cluster_id} (contains BTC)")
                return cluster_id
        
        # Otherwise, find cluster with highest total volume
        cluster_volumes = {}
        
        for cluster_id, members in cluster_members.items():
            total_volume = 0.0
            
            for symbol in members:
                if symbol in data_dict:
                    df = data_dict[symbol]
                    if 'volume' in df.columns:
                        total_volume += df['volume'].sum()
            
            cluster_volumes[cluster_id] = total_volume
        
        dominant_cluster_id = max(cluster_volumes, key=cluster_volumes.get)
        
        print(f"  [Dominant] Cluster {dominant_cluster_id} (highest volume)")
        
        return dominant_cluster_id
    
    def fit(
        self,
        data_dict: Dict[str, pd.DataFrame],
    ) -> ClusterResult:
        """
        Fit the clustering model and return cluster assignments.
        
        Parameters
        ----------
        data_dict : dict
            Dictionary mapping asset symbol to DataFrame
        
        Returns
        -------
        result : ClusterResult
            Clustering result with cluster assignments and metadata
        """
        print("\n" + "=" * 72)
        print("ASSET CLUSTERING ENGINE")
        print("=" * 72)
        
        # Step 1: Align timestamps and extract features
        print("\n[Step 1] Aligning timestamps...")
        feature_matrix, symbols = self._align_timestamps(data_dict)
        
        # Step 2: Compute correlation matrix
        print("\n[Step 2] Computing correlation matrix...")
        corr_matrix = self._compute_correlation_matrix(feature_matrix)
        
        print(f"  Correlation range: [{corr_matrix.values.min():.3f}, {corr_matrix.values.max():.3f}]")
        
        # Step 3: Convert correlation to distance
        # Distance = 1 - |correlation|
        # This treats both positive and negative correlation as similarity
        distance_matrix = 1.0 - np.abs(corr_matrix.values)
        
        # Ensure distance matrix is symmetric and has zero diagonal
        distance_matrix = (distance_matrix + distance_matrix.T) / 2
        np.fill_diagonal(distance_matrix, 0)
        
        # Convert to condensed distance matrix for linkage
        condensed_dist = squareform(distance_matrix, checks=False)
        
        # Step 4: Hierarchical clustering
        print(f"\n[Step 3] Hierarchical clustering (method={self.method})...")
        linkage_matrix = linkage(condensed_dist, method=self.method)
        
        # Step 5: Form clusters
        if self.n_clusters is not None:
            print(f"  Forming {self.n_clusters} clusters...")
            cluster_labels = fcluster(linkage_matrix, self.n_clusters, criterion='maxclust')
        else:
            print(f"  Forming clusters with distance threshold {self.distance_threshold}...")
            cluster_labels = fcluster(linkage_matrix, self.distance_threshold, criterion='distance')
        
        # Convert to 0-indexed
        cluster_labels = cluster_labels - 1
        
        # Step 6: Create cluster map and members
        cluster_map = {symbol: int(label) for symbol, label in zip(symbols, cluster_labels)}
        
        cluster_members = {}
        for symbol, cluster_id in cluster_map.items():
            if cluster_id not in cluster_members:
                cluster_members[cluster_id] = []
            cluster_members[cluster_id].append(symbol)
        
        n_clusters_formed = len(cluster_members)
        print(f"  Formed {n_clusters_formed} clusters")
        
        # Step 7: Identify dominant cluster
        print("\n[Step 4] Identifying dominant cluster...")
        dominant_cluster_id = self._identify_dominant_cluster(cluster_members, data_dict)
        
        # Step 8: Print cluster summary
        print("\n" + "-" * 72)
        print("CLUSTER SUMMARY")
        print("-" * 72)
        
        for cluster_id in sorted(cluster_members.keys()):
            members = cluster_members[cluster_id]
            is_dominant = " (DOMINANT)" if cluster_id == dominant_cluster_id else ""
            print(f"Cluster {cluster_id}{is_dominant}: {', '.join(sorted(members))}")
        
        # Create result
        result = ClusterResult(
            cluster_map=cluster_map,
            dominant_cluster_id=dominant_cluster_id,
            cluster_members=cluster_members,
            linkage_matrix=linkage_matrix,
            correlation_matrix=corr_matrix,
        )
        
        return result
    
    def plot_dendrogram(
        self,
        result: ClusterResult,
        figsize: Tuple[int, int] = (12, 6),
    ):
        """
        Plot hierarchical clustering dendrogram.
        
        Parameters
        ----------
        result : ClusterResult
            Clustering result
        figsize : tuple
            Figure size (width, height)
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("  [Warning] matplotlib not available, skipping dendrogram plot")
            return
        
        plt.figure(figsize=figsize)
        
        # Get symbols in order
        symbols = list(result.cluster_map.keys())
        
        dendrogram(
            result.linkage_matrix,
            labels=symbols,
            leaf_rotation=90,
            leaf_font_size=10,
        )
        
        plt.title("Asset Clustering Dendrogram")
        plt.xlabel("Asset")
        plt.ylabel("Distance")
        plt.tight_layout()
        
        return plt.gcf()


class MarketFactorExtractor:
    """
    Extract market factor from dominant cluster.
    
    The market factor represents the overall market trend/sentiment
    from the dominant assets (e.g., BTC, ETH).
    
    Parameters
    ----------
    method : str, default='pca'
        Method for factor extraction:
        - 'pca': First principal component
        - 'mean': Simple mean of frac_diff
        - 'weighted_mean': Volume-weighted mean
    feature_column : str, default='frac_diff'
        Column to use for factor extraction
    """
    
    def __init__(
        self,
        method: str = 'pca',
        feature_column: str = 'frac_diff',
    ):
        self.method = method
        self.feature_column = feature_column
        self.pca_ = None
    
    def extract_factor(
        self,
        data_dict: Dict[str, pd.DataFrame],
        dominant_symbols: List[str],
    ) -> pd.Series:
        """
        Extract market factor from dominant cluster assets.
        
        Parameters
        ----------
        data_dict : dict
            Dictionary mapping asset symbol to DataFrame
        dominant_symbols : list
            List of symbols in dominant cluster
        
        Returns
        -------
        market_factor : pd.Series
            Market factor time series (indexed by timestamp)
        """
        print("\n" + "=" * 72)
        print("MARKET FACTOR EXTRACTION")
        print("=" * 72)
        
        print(f"\n[Method] {self.method}")
        print(f"[Dominant Assets] {', '.join(sorted(dominant_symbols))}")
        
        # Extract feature series for dominant assets
        series_dict = {}
        
        for symbol in dominant_symbols:
            if symbol not in data_dict:
                print(f"  [Warning] {symbol} not in data_dict, skipping")
                continue
            
            df = data_dict[symbol]
            
            if self.feature_column not in df.columns:
                print(f"  [Warning] {self.feature_column} not in {symbol}, skipping")
                continue
            
            series = df[self.feature_column].copy()
            
            # Ensure index is datetime
            if not isinstance(series.index, pd.DatetimeIndex):
                if 'timestamp' in df.columns:
                    series.index = pd.to_datetime(df['timestamp'])
            
            series_dict[symbol] = series
        
        if len(series_dict) == 0:
            raise ValueError("No valid dominant assets found")
        
        # Combine into DataFrame
        feature_matrix = pd.DataFrame(series_dict)
        
        # Align timestamps (forward fill small gaps)
        feature_matrix = feature_matrix.fillna(method='ffill', limit=3)
        feature_matrix = feature_matrix.dropna()
        
        print(f"  [Data] {len(feature_matrix)} timestamps, {len(series_dict)} assets")
        
        # Extract factor based on method
        if self.method == 'pca':
            # First principal component
            self.pca_ = PCA(n_components=1, random_state=42)
            factor_values = self.pca_.fit_transform(feature_matrix.values).flatten()
            
            explained_var = self.pca_.explained_variance_ratio_[0]
            print(f"  [PCA] Explained variance: {explained_var:.2%}")
            
            market_factor = pd.Series(factor_values, index=feature_matrix.index, name='market_factor')
        
        elif self.method == 'mean':
            # Simple mean
            market_factor = feature_matrix.mean(axis=1)
            market_factor.name = 'market_factor'
            
            print(f"  [Mean] Simple average of {len(series_dict)} assets")
        
        elif self.method == 'weighted_mean':
            # Volume-weighted mean
            weights = []
            
            for symbol in feature_matrix.columns:
                df = data_dict[symbol]
                if 'volume' in df.columns:
                    # Use mean volume as weight
                    weight = df['volume'].mean()
                else:
                    weight = 1.0
                weights.append(weight)
            
            weights = np.array(weights)
            weights = weights / weights.sum()  # Normalize
            
            market_factor = (feature_matrix.values @ weights)
            market_factor = pd.Series(market_factor, index=feature_matrix.index, name='market_factor')
            
            print(f"  [Weighted Mean] Volume-weighted average")
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Standardize factor (zero mean, unit variance)
        market_factor = (market_factor - market_factor.mean()) / (market_factor.std() + 1e-8)
        
        print(f"  Market factor extracted: {len(market_factor)} timestamps")
        print(f"    Mean: {market_factor.mean():.6f}, Std: {market_factor.std():.6f}")
        
        return market_factor
