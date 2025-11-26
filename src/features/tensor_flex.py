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
    method: str = "corr_mi",
    random_state: int = 42,
) -> List[List[str]]:
    """
    Cluster feature names into correlation-based groups.
    """
    del method  # Plain correlation clustering for v1.

    df = _ensure_dataframe(X)
    numeric_df = df.select_dtypes(include=["number"]).copy()
    numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)

    variances = numeric_df.var()
    usable_cols = [col for col in numeric_df.columns if variances.get(col, 0.0) > 0]
    zero_var_cols = [col for col in numeric_df.columns if col not in usable_cols]

    if usable_cols:
        numeric_df = numeric_df[usable_cols]
    else:
        # Fall back to singleton clusters for zero variance columns.
        return [[col] for col in zero_var_cols]

    corr = numeric_df.corr().fillna(0.0).abs()
    dist_matrix = 1.0 - corr.to_numpy(copy=True)
    np.fill_diagonal(dist_matrix, 0.0)

    n_features = corr.shape[0]
    if n_features == 1:
        clusters = [[usable_cols[0]]]
    else:
        approx_clusters = max(1, int(np.ceil(n_features / max_cluster_size)))
        try:
            clustering = AgglomerativeClustering(
                n_clusters=approx_clusters,
                metric="precomputed",
                linkage="average",
            )
        except TypeError:
            clustering = AgglomerativeClustering(
                n_clusters=approx_clusters,
                affinity="precomputed",
                linkage="average",
            )
        labels = clustering.fit_predict(dist_matrix)
        clusters = []
        for label in np.unique(labels):
            members = corr.columns[labels == label].tolist()
            clusters.append(members)

    rng = np.random.default_rng(random_state)
    final_clusters: List[List[str]] = []
    for members in clusters:
        if len(members) <= max_cluster_size:
            final_clusters.append(members)
            continue
        members = members.copy()
        rng.shuffle(members)
        for start in range(0, len(members), max_cluster_size):
            final_clusters.append(members[start : start + max_cluster_size])

    for col in zero_var_cols:
        final_clusters.append([col])

    return final_clusters


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
        selector_output = self._fit_global_selector(latents_df, y, sample_weights)
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
            report_path = target / "report.json"

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

    def _fit_global_selector(
        self,
        latents: pd.DataFrame,
        y: Optional[pd.Series],
        sample_weights: Optional[np.ndarray],
    ):
        if latents.empty:
            return None, None, [], {}, {}
        if y is None or len(pd.unique(y.dropna())) < 2:
            logger.warning("No valid labels passed to Tensor-Flex selector; returning full latent set.")
            names = list(latents.columns)
            stability = {col: 1.0 for col in names}
            coefs = {col: 0.0 for col in names}
            return None, None, names, stability, coefs

        target = pd.Series(y).reindex(latents.index)
        weights = None
        if sample_weights is not None:
            if isinstance(sample_weights, pd.Series):
                weights_series = sample_weights.reindex(latents.index).fillna(1.0)
                weights = weights_series.to_numpy(dtype=float)
            else:
                weights = np.asarray(sample_weights, dtype=float).reshape(-1)
                if weights.shape[0] != latents.shape[0]:
                    raise ValueError("sample_weights length mismatch.")

        unique_values = pd.Series(target).dropna().unique()
        is_binary = len(unique_values) <= 2
        selector_type = "logistic" if is_binary else "lasso"

        selector = self._build_selector(selector_type)
        stability_scores = self._estimate_stability(latents, target, selector_type, weights)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            selector.fit(latents.values, target.values, sample_weight=weights)
        if selector_type == "logistic":
            coef = selector.coef_.ravel()
        else:
            coef = selector.coef_
        coef_map = {col: float(weight) for col, weight in zip(latents.columns, coef)}

        selected = [
            col
            for col in latents.columns
            if abs(coef_map[col]) >= self.selector_coef_threshold
            and stability_scores.get(col, 0.0) >= self.stability_threshold
        ]
        if not selected:
            sorted_cols = sorted(
                latents.columns,
                key=lambda col: (abs(coef_map[col]), stability_scores.get(col, 0.0)),
                reverse=True,
            )
            top_k = min(32, len(sorted_cols))
            selected = sorted_cols[:top_k]

        return selector, selector_type, selected, stability_scores, coef_map

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
        self.report_ = {
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
                "num_latents": latents_df.shape[1],
                "num_final_features": len(self.selected_feature_names_),
                "stability_stats": self.feature_stability_,
            },
        }


__all__ = ["cluster_features", "TensorFlexFeatureRefiner"]
