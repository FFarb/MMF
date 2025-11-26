"""
Tensor-Flex feature distillation layer.

This module implements:
    * cluster_features helper for grouping highly correlated inputs
    * TensorFlexFeatureRefiner which performs:
        - tensor talk (cross-cluster redundancy removal)
        - per-cluster PCA/whitening
        - sparsifying global selector with stability scoring
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_score

from ..config import (
    TENSOR_FLEX_MIN_LATENTS,
    TENSOR_FLEX_MAX_LATENTS,
    TENSOR_FLEX_VAR_EXPLAINED_MIN,
    TENSOR_FLEX_SHARPE_DELTA_MIN,
    TENSOR_FLEX_ENABLE_DYNAMIC_LATENTS,
    TENSOR_FLEX_SUPERVISED_WEIGHT,
    TENSOR_FLEX_CORR_THRESHOLD,
    TENSOR_FLEX_MODE,
    TP_PCT,
    SL_PCT,
)

logger = logging.getLogger(__name__)


def _ensure_dataframe(X: Union[pd.DataFrame, np.ndarray, Sequence[Sequence[float]]]) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X.copy()
    arr = np.asarray(X)
    if arr.ndim != 2:
        raise ValueError("Expected 2D input when converting to DataFrame.")
    columns = [f"feature_{i}" for i in range(arr.shape[1])]
    return pd.DataFrame(arr, columns=columns)


def _mean_abs_corr(matrix: np.ndarray) -> float:
    if matrix.size == 0:
        return 0.0
    mask = ~np.eye(matrix.shape[0], dtype=bool)
    if mask.sum() == 0:
        return 0.0
    vals = np.abs(matrix[mask])
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return 0.0
    return float(vals.mean())


def cluster_features(
    X: pd.DataFrame,
    max_cluster_size: int = 64,
    corr_threshold: float = 0.85,
    random_state: int = 42,
) -> List[List[str]]:
    """
    Cluster feature names into correlation-based groups.
    Uses recursive splitting to ensure no cluster exceeds max_cluster_size.
    """
    df = _ensure_dataframe(X)
    numeric_df = df.select_dtypes(include=["number"]).copy()
    numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)

    variances = numeric_df.var()
    usable_cols = [col for col in numeric_df.columns if variances.get(col, 0.0) > 0]
    zero_var_cols = [col for col in numeric_df.columns if col not in usable_cols]

    if not usable_cols:
        return [[col] for col in zero_var_cols]

    # Initial Clustering
    clusters = _recursive_cluster(
        numeric_df[usable_cols],
        max_size=max_cluster_size,
        corr_threshold=corr_threshold,
        random_state=random_state
    )

    # Append zero variance features as singletons
    for col in zero_var_cols:
        clusters.append([col])

    return clusters


def _recursive_cluster(
    df: pd.DataFrame,
    max_size: int,
    corr_threshold: float,
    random_state: int
) -> List[List[str]]:
    """
    Recursively split clusters until they fit within max_size.
    """
    n_features = df.shape[1]
    cols = list(df.columns)
    
    # Base case: small enough
    if n_features <= max_size:
        return [cols]

    # Compute correlation matrix
    corr = df.corr().fillna(0.0).abs()
    dist_matrix = 1.0 - corr.to_numpy(copy=True)
    np.fill_diagonal(dist_matrix, 0.0)

    # Determine split count (at least 2, but try to fit max_size)
    n_splits = max(2, int(np.ceil(n_features / max_size)))
    
    # Hierarchical Clustering
    try:
        clustering = AgglomerativeClustering(
            n_clusters=n_splits,
            metric="precomputed",
            linkage="average",
        )
    except TypeError:
        clustering = AgglomerativeClustering(
            n_clusters=n_splits,
            affinity="precomputed",
            linkage="average",
        )
        
    labels = clustering.fit_predict(dist_matrix)
    
    sub_clusters = []
    for label in np.unique(labels):
        members = [cols[i] for i in range(n_features) if labels[i] == label]
        if not members:
            continue
            
        # Check if this sub-cluster needs further splitting
        # We only recurse if it's still too big OR if we want to enforce tighter correlation
        # For now, we strictly enforce max_size.
        if len(members) > max_size:
            sub_df = df[members]
            sub_clusters.extend(_recursive_cluster(sub_df, max_size, corr_threshold, random_state))
        else:
            sub_clusters.append(members)
            
    return sub_clusters


@dataclass
class TensorTalkMapping:
    source_cluster: int
    target_cluster: int
    alpha: float
    model: LinearRegression
    source_columns: List[str]
    target_columns: List[str]


class TensorFlexFeatureRefiner:
    """
    End-to-end feature distillation layer with tensor talk and sparse selection.
    """

    def __init__(
        self,
        max_cluster_size: int = 64,
        max_pairs_per_cluster: int = 5,
        variance_threshold: float = 0.95,
        n_splits_stability: int = 5,
        stability_threshold: float = 0.6,
        selector_coef_threshold: float = 1e-4,
        selector_c: float = 0.1,
        random_state: int = 42,
        artifacts_dir: Optional[Union[str, Path]] = None,
        alpha_grid: Sequence[float] = (0.25, 0.5, 0.75, 1.0),
        supervised_weight: float = 0.2,
        corr_threshold: float = 0.85,
        min_latents: int = TENSOR_FLEX_MIN_LATENTS,
        max_latents: int = TENSOR_FLEX_MAX_LATENTS,
    ) -> None:
        self.max_cluster_size = max_cluster_size
        self.max_pairs_per_cluster = max_pairs_per_cluster
        self.variance_threshold = variance_threshold
        self.n_splits_stability = n_splits_stability
        self.stability_threshold = stability_threshold
        self.selector_coef_threshold = selector_coef_threshold
        self.selector_c = selector_c
        self.random_state = random_state
        self.artifacts_dir = Path(artifacts_dir) if artifacts_dir else None
        self.alpha_grid = tuple(alpha_grid)
        self.supervised_weight = supervised_weight
        self.corr_threshold = corr_threshold

        # Dynamic Latent Config
        self.min_latents = min_latents
        self.max_latents = max_latents
        self.var_explained_min = TENSOR_FLEX_VAR_EXPLAINED_MIN
        self.sharpe_delta_min = TENSOR_FLEX_SHARPE_DELTA_MIN
        self.enable_dynamic_latents = TENSOR_FLEX_ENABLE_DYNAMIC_LATENTS

        self.feature_names_in_: Optional[List[str]] = None
        self.fill_values_: Dict[str, float] = {}
        self.clusters_: List[List[str]] = []
        self.feature_to_cluster_: Dict[str, int] = {}
        self.tensor_talk_pairs_: List[TensorTalkMapping] = []
        self.cluster_pca_: Dict[int, PCA] = {}
        self.cluster_components_: Dict[int, int] = {}
        self.cluster_latent_names_: Dict[int, List[str]] = {}
        self.latent_feature_names_: List[str] = []
        self.global_selector_ = None
        self.selector_type_: Optional[str] = None
        self.feature_stability_: Dict[str, float] = {}
        self.feature_importance_: Dict[str, float] = {}
        self.selected_feature_names_: List[str] = []
        self.report_: Dict[str, object] = {}
        self.fitted_: bool = False

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        sample_weights: Optional[np.ndarray] = None,
    ) -> "TensorFlexFeatureRefiner":
        df = self._prepare_training_frame(X)
        self.feature_names_in_ = list(df.columns)

        clusters = cluster_features(
            df,
            max_cluster_size=self.max_cluster_size,
            corr_threshold=self.corr_threshold,
            random_state=self.random_state,
        )
        self.clusters_ = clusters
        self.feature_to_cluster_ = {
            feature: cluster_id
            for cluster_id, members in enumerate(clusters)
            for feature in members
        }

        cluster_frames = {idx: df[members].copy() for idx, members in enumerate(clusters)}
        intra_before = self._compute_intra_cluster_stats(cluster_frames)
        cross_before = self._compute_cross_cluster_stats(cluster_frames)

        talk_pairs = self._run_tensor_talk(cluster_frames, y)
        self.tensor_talk_pairs_ = talk_pairs

        intra_after = self._compute_intra_cluster_stats(cluster_frames)
        cross_after = self._compute_cross_cluster_stats(cluster_frames)

        latents_df = self._fit_cluster_latents(cluster_frames, y)
        
        if TENSOR_FLEX_MODE == "v2":
            selector_output = self._fit_global_selector_v2(latents_df, y, sample_weights)
        else:
            selector_output = self._fit_global_selector_legacy(latents_df, y, sample_weights)
            
        (
            selector,
            selector_type,
            selected_features,
            stability_scores,
            coef_map,
        ) = selector_output

        self.global_selector_ = selector
        self.selector_type_ = selector_type
        self.latent_feature_names_ = list(latents_df.columns)
        self.selected_feature_names_ = selected_features
        self.feature_stability_ = stability_scores
        self.feature_importance_ = coef_map
        self.fitted_ = True

        self._build_report(
            intra_before=intra_before,
            intra_after=intra_after,
            cross_before=cross_before,
            cross_after=cross_after,
            latents_df=latents_df,
        )

        if self.artifacts_dir:
            self.save(self.artifacts_dir)

        return self

    def transform(self, X: pd.DataFrame, mode: str = "selected") -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("TensorFlexFeatureRefiner must be fitted before calling transform().")
        mode_normalized = mode.lower()
        if mode_normalized not in ("selected", "full_latents"):
            raise ValueError("Tensor-Flex transform mode must be 'selected' or 'full_latents'.")

        df = self._prepare_inference_frame(X)
        cluster_frames = self._build_cluster_frames(df)
        latents = self._transform_clusters(cluster_frames)

        if mode_normalized == "full_latents":
            return latents
        if not self.selected_feature_names_:
            return latents
        missing = [col for col in self.selected_feature_names_ if col not in latents.columns]
        if missing:
            raise RuntimeError(f"Missing latent columns during transform: {missing}")
        return latents[self.selected_feature_names_].copy()

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        sample_weights: Optional[np.ndarray] = None,
        mode: str = "selected",
    ) -> pd.DataFrame:
        return self.fit(X, y, sample_weights).transform(X, mode=mode)

    def save(self, path: Union[str, Path]) -> None:
        target = Path(path)
        if target.suffix:
            target.parent.mkdir(parents=True, exist_ok=True)
            model_path = target
            report_path = target.with_suffix(".report.json")
        else:
            target.mkdir(parents=True, exist_ok=True)
            model_path = target / "tensor_flex.joblib"
            report_path = target / "tensor_flex_report.json"

        joblib.dump(self, model_path)
        if self.report_:
            with report_path.open("w", encoding="utf-8") as fp:
                json.dump(self.report_, fp, indent=2)
        logger.info("Tensor-Flex artifacts saved to %s", model_path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TensorFlexFeatureRefiner":
        source = Path(path)
        model_path = source
        if source.is_dir():
            model_path = source / "tensor_flex.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"No Tensor-Flex artifact found at {model_path}")
        obj = joblib.load(model_path)
        if not isinstance(obj, cls):
            raise TypeError("Loaded artifact is not a TensorFlexFeatureRefiner.")
        return obj

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _prepare_training_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        df = _ensure_dataframe(X)
        numeric = df.select_dtypes(include=["number"]).copy()
        numeric = numeric.replace([np.inf, -np.inf], np.nan)
        medians = numeric.median()
        self.fill_values_ = medians.fillna(0.0).to_dict()
        numeric = numeric.fillna(self.fill_values_)
        numeric = numeric.fillna(0.0)
        return numeric

    def _prepare_inference_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        df = _ensure_dataframe(X)
        df = df.replace([np.inf, -np.inf], np.nan)
        expected = self.feature_names_in_ or []
        for col in expected:
            if col not in df.columns:
                df[col] = self.fill_values_.get(col, 0.0)
        df = df[expected].copy()
        for col, value in self.fill_values_.items():
            if col in df.columns:
                df[col] = df[col].fillna(value)
        df = df.fillna(0.0)
        return df

    def _run_tensor_talk(
        self,
        cluster_frames: Dict[int, pd.DataFrame],
        y: Optional[pd.Series],
    ) -> List[TensorTalkMapping]:
        rng = np.random.default_rng(self.random_state)
        cluster_ids = sorted(cluster_frames.keys())
        similarity_matrix: Dict[Tuple[int, int], float] = {}
        for i, cid in enumerate(cluster_ids):
            Xi = cluster_frames[cid]
            for nj in cluster_ids[i + 1 :]:
                Xj = cluster_frames[nj]
                sim = self._cluster_similarity(Xi, Xj)
                similarity_matrix[(cid, nj)] = sim
                similarity_matrix[(nj, cid)] = sim

        talk_pairs: List[TensorTalkMapping] = []
        for cid in cluster_ids:
            neighbors = [
                (other, similarity_matrix.get((cid, other), 0.0))
                for other in cluster_ids
                if other != cid
            ]
            neighbors.sort(key=lambda item: item[1], reverse=True)
            neighbors = neighbors[: self.max_pairs_per_cluster]
            for neighbor_id, score in neighbors:
                if score <= 0:
                    continue
                mapping = self._build_tensor_talk_pair(
                    source_cluster=cid,
                    target_cluster=neighbor_id,
                    cluster_frames=cluster_frames,
                    y=y,
                )
                if mapping:
                    talk_pairs.append(mapping)

        # Shuffle order to avoid directional bias but keep determinism.
        rng.shuffle(talk_pairs)
        for pair in talk_pairs:
            self._apply_tensor_talk(pair, cluster_frames)

        return talk_pairs

    def _cluster_similarity(self, Xi: pd.DataFrame, Xj: pd.DataFrame) -> float:
        if Xi.shape[1] == 0 or Xj.shape[1] == 0:
            return 0.0
        matrix = np.corrcoef(
            np.hstack([Xi.to_numpy(dtype=float), Xj.to_numpy(dtype=float)]).T
        )
        n_i = Xi.shape[1]
        cross = matrix[:n_i, n_i:]
        return float(np.nanmean(np.abs(cross)))

    def _build_tensor_talk_pair(
        self,
        source_cluster: int,
        target_cluster: int,
        cluster_frames: Dict[int, pd.DataFrame],
        y: Optional[pd.Series],
    ) -> Optional[TensorTalkMapping]:
        Xi = cluster_frames[source_cluster]
        Xj = cluster_frames[target_cluster]
        if Xi.shape[1] == 0 or Xj.shape[1] == 0:
            return None

        reg = LinearRegression()
        reg.fit(Xi.values, Xj.values)
        predictions = reg.predict(Xi.values)

        baseline_label_score = self._label_alignment(Xj, y)
        best_alpha = 0.0
        best_score = float("inf")

        for alpha in self.alpha_grid:
            residual = Xj.values - alpha * predictions
            new_df = pd.DataFrame(residual, columns=Xj.columns, index=Xj.index)
            cross_corr = self._cluster_similarity(Xi, new_df)
            label_score = self._label_alignment(new_df, y)
            penalty = max(0.0, baseline_label_score - label_score)
            objective = cross_corr + 0.5 * penalty
            if objective < best_score:
                best_score = objective
                best_alpha = alpha

        if best_alpha == 0.0:
            return None
        return TensorTalkMapping(
            source_cluster=source_cluster,
            target_cluster=target_cluster,
            alpha=best_alpha,
            model=reg,
            source_columns=list(Xi.columns),
            target_columns=list(Xj.columns),
        )

    def _apply_tensor_talk(
        self,
        mapping: TensorTalkMapping,
        cluster_frames: Dict[int, pd.DataFrame],
    ) -> None:
        Xi = cluster_frames[mapping.source_cluster]
        Xj = cluster_frames[mapping.target_cluster]
        preds = mapping.model.predict(Xi.values)
        residual = Xj.values - mapping.alpha * preds
        cluster_frames[mapping.target_cluster] = pd.DataFrame(
            residual,
            columns=mapping.target_columns,
            index=Xj.index,
        )

    def _fit_cluster_latents(
        self,
        cluster_frames: Dict[int, pd.DataFrame],
        y: Optional[pd.Series],
    ) -> pd.DataFrame:
        latents: List[pd.DataFrame] = []
        rng = np.random.default_rng(self.random_state)

        for cluster_id, df in cluster_frames.items():
            pca = PCA(random_state=self.random_state)
            pca.fit(df.values)
            var_ratio = np.nan_to_num(pca.explained_variance_ratio_, nan=0.0)
            cum = np.cumsum(var_ratio)
            n_components = int(np.searchsorted(cum, self.variance_threshold) + 1)
            n_components = min(n_components, df.shape[1])
            if n_components == 0:
                n_components = min(1, df.shape[1])
            Z = pca.transform(df.values)[:, :n_components]
            latent_cols = [f"tf_cluster{cluster_id}_pc{idx+1}" for idx in range(n_components)]
            latent_df = pd.DataFrame(Z, columns=latent_cols, index=df.index)
            latents.append(latent_df)
            self.cluster_pca_[cluster_id] = pca
            self.cluster_components_[cluster_id] = n_components
            self.cluster_latent_names_[cluster_id] = latent_cols

        rng.shuffle(latents)
        latents_df = pd.concat(latents, axis=1)
        return latents_df

        return selector, selector_type, selected, stability_scores, coef_map

    def _evaluate_latent_set(self, X_train, y_train, X_val, y_val, class_weight):
        """Helper to evaluate a subset of latents."""
        if X_train.shape[1] == 0: return -1.0
        model = LogisticRegression(solver='liblinear', class_weight=class_weight, random_state=self.random_state, max_iter=100)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        prec = precision_score(y_val, preds, zero_division=0)
        if sum(preds) < 5: return -1.0
        expectancy = (prec * TP_PCT) - ((1 - prec) * SL_PCT)
        return expectancy * np.sqrt(sum(preds))

        return selector, selector_type, selected, stability_scores, coef_map

    def _fit_global_selector_v2(
        self,
        latents: pd.DataFrame,
        y: Optional[pd.Series],
        sample_weights: Optional[np.ndarray],
    ):
        """
        V2 Selector: Stability + Supervised Score + Variance Rank.
        """
        if latents.empty:
            return None, None, [], {}, {}

        # 1. Calculate Stability for ALL latents
        stability_scores = self._estimate_stability(latents, y, "logistic", sample_weights) # Re-use estimate stability or custom?
        # Actually _estimate_stability uses Lasso/Logistic. We want purely unsupervised stability of the component direction?
        # The plan said: "run PCA on multiple CV splits... align components".
        # But here 'latents' are already PCA components from the FULL fit.
        # If we want stability of the *selection*, we can use the existing _estimate_stability which checks how often a feature is selected.
        # BUT, the plan also mentioned "stability score as a factor in latent ranking".
        # Let's stick to a simpler proxy for now: How stable is the correlation with target?
        # OR, we can just use the existing stability score which is "frequency of selection by Lasso/Logistic".
        
        # Let's implement a robust scoring metric for each latent:
        # Score = (1 - w) * (VarianceRatio normalized) + w * (Supervised Correlation)
        # Multiplied by Stability (0..1)
        
        # We need variance ratios from the cluster PCAs.
        # Map latent name -> variance ratio
        var_ratios = {}
        for cid, pca in self.cluster_pca_.items():
            names = self.cluster_latent_names_[cid]
            ratios = pca.explained_variance_ratio_[:len(names)]
            for name, ratio in zip(names, ratios):
                var_ratios[name] = ratio
                
        # Normalize variance ratios globally or per cluster? Globally is better for "global" importance.
        # But different clusters have different total variance.
        # Let's just use raw ratio.
        
        # Supervised Score
        supervised_scores = {}
        if y is not None:
            y_aligned = y.reindex(latents.index).fillna(0)
            for col in latents.columns:
                corr = latents[col].corr(y_aligned)
                supervised_scores[col] = abs(corr) if not np.isnan(corr) else 0.0
        else:
            supervised_scores = {col: 0.0 for col in latents.columns}
            
        # Stability Score (using simple subsample correlation stability)
        # If we don't have a good stability measure yet, default to 1.0
        # The existing _estimate_stability is for FEATURE SELECTION stability, not component stability.
        # Let's use a simple bootstrap correlation stability if y is present.
        stability_map = {}
        if y is not None:
            stability_map = self._bootstrap_correlation_stability(latents, y)
        else:
            stability_map = {col: 1.0 for col in latents.columns}

        # Combine Scores
        final_scores = {}
        w = self.supervised_weight
        
        # Max variance for normalization
        max_var = max(var_ratios.values()) if var_ratios else 1.0
        max_sup = max(supervised_scores.values()) if supervised_scores else 1.0
        
        for col in latents.columns:
            v_score = var_ratios.get(col, 0.0) / max_var
            s_score = supervised_scores.get(col, 0.0) / max_sup if max_sup > 0 else 0.0
            stab = stability_map.get(col, 1.0)
            
            # Combined metric
            final_scores[col] = ( (1 - w) * v_score + w * s_score ) * stab

        # Select Top K
        sorted_cols = sorted(final_scores.keys(), key=lambda x: final_scores[x], reverse=True)
        
        # Filter by min/max latents
        # Also ensure we don't pick garbage (score ~ 0)
        selected = []
        for col in sorted_cols:
            if len(selected) >= self.max_latents:
                break
            if final_scores[col] > 1e-6: # minimal threshold
                selected.append(col)
                
        # Enforce min latents if possible
        if len(selected) < self.min_latents:
            # Take more if available
            remaining = [c for c in sorted_cols if c not in selected]
            needed = self.min_latents - len(selected)
            selected.extend(remaining[:needed])

        return None, "v2_weighted_rank", selected, stability_map, final_scores

    def _bootstrap_correlation_stability(self, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> Dict[str, float]:
        """
        Estimate how stable the sign/magnitude of correlation is across splits.
        Returns 1.0 - std(corr) / 2 (approx) or similar.
        Actually, let's use: mean(abs(corr)) / (std(abs(corr)) + epsilon) -> Signal to Noise?
        Or simply: fraction of splits where correlation sign matches global sign?
        Let's use a simple CV consistency: mean correlation across folds.
        """
        scores = {col: [] for col in X.columns}
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        y_aligned = y.reindex(X.index).fillna(0)
        
        for train_idx, _ in tscv.split(X):
            X_fold = X.iloc[train_idx]
            y_fold = y_aligned.iloc[train_idx]
            
            # Skip small folds
            if len(X_fold) < 50:
                continue
                
            for col in X.columns:
                c = X_fold[col].corr(y_fold)
                scores[col].append(c if not np.isnan(c) else 0.0)
                
        stability = {}
        for col, vals in scores.items():
            if not vals:
                stability[col] = 0.0
                continue
            # Stability = 1 - (std / (mean + epsilon)) ? No, too volatile.
            # Let's use: How often is the sign consistent?
            vals = np.array(vals)
            mean_val = np.mean(vals)
            if abs(mean_val) < 1e-9:
                stability[col] = 0.0
            else:
                # CV of the correlation coefficient
                # Lower CV is better. We want a score in [0, 1].
                # Let's use 1 / (1 + CV)
                cv = np.std(vals) / (abs(mean_val) + 1e-6)
                stability[col] = 1.0 / (1.0 + cv)
                
        return stability

    def _build_selector(self, selector_type: str):
        if selector_type == "logistic":
            return LogisticRegression(
                penalty="l1",
                solver="saga",
                C=self.selector_c,
                max_iter=5000,
                random_state=self.random_state,
            )
        return Lasso(alpha=self.selector_c, max_iter=5000)

    def _estimate_stability(
        self,
        latents: pd.DataFrame,
        target: pd.Series,
        selector_type: str,
        weights: Optional[np.ndarray],
    ) -> Dict[str, float]:
        n_samples = latents.shape[0]
        max_splits = max(1, n_samples - 1)
        desired = max(2, self.n_splits_stability)
        n_splits = min(desired, max_splits)
        if n_samples < 5 or n_splits < 2:
            return {col: 1.0 for col in latents.columns}

        tscv = TimeSeriesSplit(n_splits=n_splits)
        counts = {col: 0 for col in latents.columns}

        for split_id, (train_idx, _) in enumerate(tscv.split(latents)):
            selector = self._build_selector(selector_type)
            X_train = latents.iloc[train_idx]
            y_train = target.iloc[train_idx]
            w_train = weights[train_idx] if weights is not None else None
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ConvergenceWarning)
                selector.fit(X_train.values, y_train.values, sample_weight=w_train)
            coef = selector.coef_
            if selector_type == "logistic":
                coef = coef.ravel()
            non_zero = np.abs(coef) > (self.selector_coef_threshold / 5.0)
            for col, flag in zip(latents.columns, non_zero):
                if flag:
                    counts[col] += 1

        stability = {col: counts[col] / float(n_splits) for col in latents.columns}
        return stability

    def _compute_intra_cluster_stats(self, cluster_frames: Dict[int, pd.DataFrame]) -> Dict[str, float]:
        stats = []
        for df in cluster_frames.values():
            if df.shape[1] <= 1:
                continue
            corr = np.corrcoef(df.values.T)
            stats.append(_mean_abs_corr(corr))
        return {
            "mean": float(np.mean(stats)) if stats else 0.0,
            "max": float(np.max(stats)) if stats else 0.0,
        }

    def _compute_cross_cluster_stats(self, cluster_frames: Dict[int, pd.DataFrame]) -> Dict[str, float]:
        stats = []
        ids = sorted(cluster_frames.keys())
        for i, cid in enumerate(ids):
            for nid in ids[i + 1 :]:
                stats.append(self._cluster_similarity(cluster_frames[cid], cluster_frames[nid]))
        return {
            "mean": float(np.mean(stats)) if stats else 0.0,
            "max": float(np.max(stats)) if stats else 0.0,
        }

    def _label_alignment(self, df: pd.DataFrame, y: Optional[pd.Series]) -> float:
        if y is None:
            return 0.0
        y_vec = pd.Series(y).reindex(df.index)
        corr_vals: List[float] = []
        for col in df.columns:
            series = df[col]
            if series.std() == 0 or y_vec.std() == 0:
                continue
            corr = np.corrcoef(series.values, y_vec.values)[0, 1]
            if not np.isnan(corr):
                corr_vals.append(abs(corr))
        if not corr_vals:
            return 0.0
        return float(np.mean(corr_vals))

    def _build_cluster_frames(self, df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
        frames = {}
        for cluster_id, features in enumerate(self.clusters_):
            cols = [col for col in features if col in df.columns]
            missing = len(features) - len(cols)
            if cols:
                frames[cluster_id] = df[cols].copy()
            else:
                frames[cluster_id] = pd.DataFrame(
                    np.zeros((df.shape[0], len(features))),
                    columns=features,
                    index=df.index,
                )
            if missing > 0:
                logger.warning("Cluster %s missing %s columns during transform.", cluster_id, missing)
        for pair in self.tensor_talk_pairs_:
            if pair.target_cluster not in frames:
                continue
            self._apply_tensor_talk(pair, frames)
        return frames

    def _transform_clusters(self, cluster_frames: Dict[int, pd.DataFrame]) -> pd.DataFrame:
        latent_frames = []
        for cluster_id, df in cluster_frames.items():
            if cluster_id not in self.cluster_pca_:
                continue
            pca = self.cluster_pca_[cluster_id]
            n_components = self.cluster_components_.get(cluster_id, df.shape[1])
            transformed = pca.transform(df.values)[:, :n_components]
            cols = self.cluster_latent_names_.get(
                cluster_id,
                [f"tf_cluster{cluster_id}_pc{idx+1}" for idx in range(n_components)],
            )
            latent_frames.append(pd.DataFrame(transformed, columns=cols, index=df.index))
        if not latent_frames:
            return pd.DataFrame(index=cluster_frames[0].index if cluster_frames else None)
        latents = pd.concat(latent_frames, axis=1)
        return latents

    def _build_report(
        self,
        intra_before: Dict[str, float],
        intra_after: Dict[str, float],
        cross_before: Dict[str, float],
        cross_after: Dict[str, float],
        latents_df: pd.DataFrame,
    ) -> None:
        # Collect latent details
        latent_details = []
        for col in latents_df.columns:
            latent_details.append({
                "name": col,
                "selected": col in self.selected_feature_names_,
                "stability": self.feature_stability_.get(col, 0.0),
                "importance_score": self.feature_importance_.get(col, 0.0),
            })
            
        self.report_ = {
            "config": {
                "mode": TENSOR_FLEX_MODE,
                "max_cluster_size": self.max_cluster_size,
                "min_latents": self.min_latents,
                "max_latents": self.max_latents,
                "supervised_weight": self.supervised_weight,
            },
            "clusters": {
                "num_clusters": len(self.clusters_),
                "sizes": [len(c) for c in self.clusters_],
                "intra_corr_before_mean": intra_before.get("mean", 0.0),
                "intra_corr_after_mean": intra_after.get("mean", 0.0),
            },
            "cross_cluster_corr": {
                "before_mean": cross_before.get("mean", 0.0),
                "after_mean": cross_after.get("mean", 0.0),
            },
            "global_selector": {
                "num_latents_total": latents_df.shape[1],
                "num_latents_selected": len(self.selected_feature_names_),
                "selector_type": self.selector_type_,
                "latents": latent_details,
            },
        }


__all__ = ["cluster_features", "TensorFlexFeatureRefiner"]
