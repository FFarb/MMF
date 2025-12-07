"""
Structural Feature Selector implementing Block-Diagonal Regularization.
Transforms 'Big Data' into a 'Structured Archipelago' of uncorrelated signals.
"""
import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from sklearn.feature_selection import mutual_info_regression
from typing import List, Dict, Sequence, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from ..config import (
    ALPHA_COUNCIL_MIN_FEATURES,
    ALPHA_COUNCIL_MAX_FEATURES,
    ALPHA_COUNCIL_FEATURE_PENALTY,
    ALPHA_COUNCIL_ENABLE_DYNAMIC_BUDGET,
    TP_PCT,
    SL_PCT,
)

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
        dist_array = np.array(dist_matrix.values, copy=True)  # Force writable copy
        np.fill_diagonal(dist_array, 0)
        condensed_dist = squareform(dist_array)
        
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
        if not ALPHA_COUNCIL_ENABLE_DYNAMIC_BUDGET:
            print("    [Alpha Council] Using fixed feature budget (Legacy Mode).")
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
        
        else:
            print("    [Alpha Council] Using dynamic feature budget (Smart Mode).")
            # 1. Flatten all candidates sorted by block quality
            candidate_pool = []
            for label, score, feats in block_scores:
                refined_feats = self._apply_leader_follower_constraint(X, feats)
                candidate_pool.extend(refined_feats)
            
            # 2. Greedy Selection with Validation Proxy
            return self._dynamic_greedy_selection(X, y, candidate_pool)

    def _dynamic_greedy_selection(self, X: pd.DataFrame, y: pd.Series, candidates: List[str]) -> List[str]:
        """
        Iteratively adds features if they improve the Sharpe/Expectancy proxy.
        """
        # Split for proxy validation (chronological 70/30)
        split_idx = int(len(X) * 0.7)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        
        selected: List[str] = []
        best_metric = -float('inf')
        
        # Pre-compute class weights once
        n_pos = (y_train == 1).sum()
        n_neg = (y_train == 0).sum()
        class_weight = {0: 1.0, 1: n_neg / max(1, n_pos)}
        
        print(f"    [Alpha Council] Optimizing feature set (Min: {ALPHA_COUNCIL_MIN_FEATURES}, Max: {ALPHA_COUNCIL_MAX_FEATURES})...")
        
        # Batch size for evaluation to speed up
        batch_size = 1
        
        for i in range(0, len(candidates), batch_size):
            # Stop if max reached
            if len(selected) >= ALPHA_COUNCIL_MAX_FEATURES:
                break
                
            batch = candidates[i : i + batch_size]
            trial_set = selected + batch
            
            # Force add if below min features
            if len(trial_set) <= ALPHA_COUNCIL_MIN_FEATURES:
                selected.extend(batch)
                continue
                
            # Evaluate
            metric = self._evaluate_feature_set(X_train[trial_set], y_train, X_val[trial_set], y_val, class_weight)
            
            # Check improvement
            marginal_gain = metric - best_metric
            
            if marginal_gain > ALPHA_COUNCIL_FEATURE_PENALTY:
                selected.extend(batch)
                best_metric = metric
                # print(f"        Added {batch} | Count: {len(selected)} | Metric: {metric:.4f} (+{marginal_gain:.4f})")
            else:
                # If we fail to improve, we skip this batch. 
                # Optional: early stopping if we fail many times, but for now just skip.
                pass
                
        print(f"    [Alpha Council] Selected {len(selected)} features. Final Proxy Metric: {best_metric:.4f}")
        return selected

    def _evaluate_feature_set(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series, 
        X_val: pd.DataFrame, 
        y_val: pd.Series,
        class_weight: Dict[int, float]
    ) -> float:
        """
        Trains a quick proxy model and returns Expectancy/Sharpe proxy.
        """
        model = LogisticRegression(
            solver='liblinear', 
            class_weight=class_weight, 
            random_state=self.random_state,
            max_iter=100
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        
        # Calculate Expectancy Proxy
        # Win Rate * TP - Loss Rate * SL
        # Assuming fixed TP/SL from config for proxy
        
        prec = precision_score(y_val, preds, zero_division=0)
        rec = recall_score(y_val, preds, zero_division=0)
        
        if sum(preds) < 10: # Too few trades
            return -1.0
            
        # Expectancy = (Win% * TP) - (Loss% * SL)
        # Win% is Precision. Loss% is 1 - Precision.
        
        expectancy = (prec * TP_PCT) - ((1 - prec) * SL_PCT)
        
        # We can also use Sharpe Proxy: Expectancy / StdDev(Returns)
        # But for simple feature selection, Expectancy * sqrt(N_Trades) is a good proxy for Sharpe
        
        n_trades = sum(preds)
        sharpe_proxy = expectancy * np.sqrt(n_trades)
        
        return sharpe_proxy


__all__ = ["AlphaCouncil"]
