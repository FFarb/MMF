"""
Structural Feature Selector implementing Block-Diagonal Regularization.
Transforms 'Big Data' into a 'Structured Archipelago' of uncorrelated signals.
"""
import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from sklearn.feature_selection import mutual_info_regression
from typing import List, Dict, Sequence

class AlphaCouncil:
    """
    Structural Feature Selector implementing Block-Diagonal Regularization.
    Transforms 'Big Data' into a 'Structured Archipelago' of uncorrelated signals.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def _get_correlation_clusters(self, df: pd.DataFrame, threshold: float = 0.7) -> Dict[int, List[str]]:
        """
        Uses Hierarchical Clustering (Ward's method) to find the block-diagonal structure.
        """
        # 1. Compute Correlation Matrix
        corr_matrix = df.corr(method='spearman').abs()
        
        # 2. Hierarchical Clustering
        # Fill NaNs with 0 to avoid linkage errors
        dist_matrix = 1 - corr_matrix.fillna(0)
        np.fill_diagonal(dist_matrix.values, 0)
        condensed_dist = squareform(dist_matrix)
        
        linkage_matrix = hierarchy.linkage(condensed_dist, method='ward')
        
        # 3. Form Flat Clusters
        cluster_labels = hierarchy.fcluster(linkage_matrix, t=threshold, criterion='distance')
        
        clusters = {}
        for feature, label in zip(df.columns, cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(feature)
            
        return clusters

    def _evaluate_block_strength(self, df: pd.DataFrame, features: List[str], target: pd.Series) -> float:
        """
        Calculates the aggregate predictive power of a block using Mutual Information.
        """
        if not features:
            return 0.0
            
        # Represents the block by its mean signal (Principal Component proxy)
        block_signal = df[features].mean(axis=1)
        
        # Clean NaNs for metric calculation
        mask = ~block_signal.isna() & ~target.isna()
        if mask.sum() < 10:
            return 0.0
            
        mi = mutual_info_regression(
            block_signal[mask].values.reshape(-1, 1), 
            target[mask].values,
            random_state=self.random_state
        )
        return mi[0]

    def _apply_leader_follower_constraint(self, df: pd.DataFrame, features: List[str]) -> List[str]:
        """
        Selects non-redundant leaders from a block.
        """
        if len(features) < 2:
            return features
            
        # Sort by variance (activity)
        variances = df[features].var()
        sorted_features = variances.sort_values(ascending=False).index.tolist()
        
        selected = []
        for f in sorted_features:
            is_redundant = False
            for existing in selected:
                if abs(df[f].corr(df[existing])) > 0.85: # Hard cutoff for redundancy within block
                    is_redundant = True
                    break
            if not is_redundant:
                selected.append(f)
                
        return selected

    def screen_features(self, X: pd.DataFrame, y: pd.Series, n_features: int = 25) -> List[str]:
        """
        Main pipeline: Cluster -> Rank Blocks -> Harvest Leaders.
        """
        print(f"    [Alpha Council] Structuring {X.shape[1]} raw features...")
        
        # A. Identify Blocks
        clusters = self._get_correlation_clusters(X, threshold=0.5)
        print(f"    [Alpha Council] Identified {len(clusters)} structural blocks.")
        
        # B. Rank Blocks (Profitability First)
        block_scores = []
        for label, feats in clusters.items():
            score = self._evaluate_block_strength(X, feats, y)
            block_scores.append((label, score, feats))
            
        block_scores.sort(key=lambda x: x[1], reverse=True)
        
        # C. Harvest Features
        final_selection = []
        
        # Distribute feature budget proportional to block score
        total_score = sum(s for _, s, _ in block_scores) + 1e-9
        
        for label, score, feats in block_scores:
            if len(final_selection) >= n_features:
                break
            
            # Determine how many to take from this block
            allocation = max(1, int(n_features * (score / total_score)))
            
            # Filter redundant features within block
            refined_feats = self._apply_leader_follower_constraint(X, feats)
            
            # Add top K from this block
            final_selection.extend(refined_feats[:allocation])
            
        # Hard cap
        return final_selection[:n_features]


__all__ = ["AlphaCouncil"]
