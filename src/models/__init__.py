"""
Labeling and modeling utilities (formerly ``feature_selector.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

from ..config import (
    BARRIER_HORIZON,
    FEATURE_STORE,
    RANDOM_SEED,
    SL_PCT,
    TP_PCT,
    TOP_FEATURES,
    TRAINING_SET,
    TRAIN_SPLIT,
    USE_DYNAMIC_TARGETS,
    TP_ATR_MULT,
    SL_ATR_MULT,
    VOLATILITY_LOOKBACK,
    META_PROB_THRESHOLD,
    PRIMARY_RECALL_TARGET,
)
from ..meta_model import MetaModelTrainer
from ..metrics import profit_weighted_confusion_matrix
from src.features.alpha_council import AlphaCouncil
from src.models.moe_ensemble import MixtureOfExpertsEnsemble
from src.training.meta_controller import TrainingScheduler


def get_triple_barrier_labels(
    df: pd.DataFrame, tp: float = TP_PCT, sl: float = SL_PCT, horizon: int = BARRIER_HORIZON
) -> np.ndarray:
    """
    Generate triple-barrier labels where 1 indicates the TP was hit before SL within ``horizon`` bars.
    """
    print(f"[LABELING] Triple Barrier (TP={tp}, SL={sl}, horizon={horizon})")
    labels: List[Optional[int]] = []

    close = df["close"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    n = len(df)

    for i in range(n):
        if i + horizon >= n:
            labels.append(np.nan)
            continue

        entry_price = close[i]
        tp_price = entry_price * (1 + tp)
        sl_price = entry_price * (1 - sl)
        outcome = 0

        for j in range(1, horizon + 1):
            current_high = high[i + j]
            current_low = low[i + j]

            if current_low <= sl_price:
                outcome = 0
                break
            if current_high >= tp_price:
                outcome = 1
                break

        labels.append(outcome)

    return np.array(labels, dtype=float)


def get_dynamic_labels(
    df: pd.DataFrame,
    atr_col: str = "ATR_14",
    tp_mult: float = TP_ATR_MULT,
    sl_mult: float = SL_ATR_MULT,
    horizon: int = BARRIER_HORIZON,
) -> np.ndarray:
    """
    Generate labels using dynamic ATR-based targets.
    """
    print(f"[LABELING] Dynamic ATR (TP={tp_mult}xATR, SL={sl_mult}xATR, horizon={horizon})")
    labels: List[Optional[int]] = []

    close = df["close"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()

    # Handle ATR column
    if atr_col in df.columns:
        atr = df[atr_col].to_numpy()
    else:
        # Calculate rolling volatility proxy if ATR missing
        print(f"[WARNING] {atr_col} missing. Using rolling std dev proxy.")
        atr = df["close"].rolling(VOLATILITY_LOOKBACK).std().fillna(0).to_numpy()

    n = len(df)

    for i in range(n):
        if i + horizon >= n:
            labels.append(np.nan)
            continue

        entry_price = close[i]
        current_atr = atr[i]

        # Sanity check for low volatility
        if current_atr < entry_price * 0.001:
            labels.append(0)  # Do not trade in extremely low vol
            continue

        tp_price = entry_price + (current_atr * tp_mult)
        sl_price = entry_price - (current_atr * sl_mult)
        outcome = 0

        for j in range(1, horizon + 1):
            current_high = high[i + j]
            current_low = low[i + j]

            if current_low <= sl_price:
                outcome = 0
                break
            if current_high >= tp_price:
                outcome = 1
                break

        labels.append(outcome)

    return np.array(labels, dtype=float)


def filter_correlated_features(X: pd.DataFrame, y: pd.Series, threshold: float = 0.95) -> pd.DataFrame:
    """
    Remove highly correlated features, keeping whichever has the higher correlation with the target.
    """
    print(f"[FILTER] Dropping columns with abs corr > {threshold}")
    correlations_with_target = X.corrwith(y).abs()
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop: set[str] = set()
    for column in upper.columns:
        correlated_cols = upper.index[upper[column] > threshold].tolist()
        for other_col in correlated_cols:
            if correlations_with_target[column] < correlations_with_target[other_col]:
                to_drop.add(column)
            else:
                to_drop.add(other_col)

    print(f"[FILTER] Removing {len(to_drop)} columns.")
    return X.drop(columns=list(to_drop), errors="ignore")


@dataclass
class SniperModelTrainer:
    """
    End-to-end workflow for labeling and training the Mixture-of-Experts sniper.
    """

    tp_pct: float = TP_PCT
    sl_pct: float = SL_PCT
    horizon: int = BARRIER_HORIZON
    train_split: float = TRAIN_SPLIT
    top_features: int = TOP_FEATURES
    random_state: int = RANDOM_SEED
    use_dynamic_targets: bool = USE_DYNAMIC_TARGETS
    tp_atr_mult: float = TP_ATR_MULT
    sl_atr_mult: float = SL_ATR_MULT
    volatility_lookback: int = VOLATILITY_LOOKBACK

    def run(
        self,
        feature_store: Path | str = FEATURE_STORE,
        output_path: Path | str = TRAINING_SET,
        corr_threshold: float = 0.95,
        input_df: Optional[pd.DataFrame] = None,
        model_name: str = "sniper_model"
    ) -> Dict[str, object]:
        """
        Execute the labeling, feature ranking, model training, and artifact saving workflow.
        """
        df_labeled, X, y, feature_cols = self.prepare_training_frame(
            feature_store=feature_store,
            input_df=input_df,
        )

        X_corr = filter_correlated_features(X, y, threshold=corr_threshold)

        council = AlphaCouncil(random_state=self.random_state)
        selected_features = council.screen_features(X_corr, y)
        physics_features = self._determine_physics_features(X_corr.columns)
        for feature in physics_features:
            if feature not in selected_features:
                selected_features.append(feature)
        print(f"[COUNCIL] {len(selected_features)} survivor features after voting.")

        metrics, model, training_meta = self._train_model(
            X_corr[selected_features], y, model_name, physics_features
        )

        final_cols = ["open", "high", "low", "close", "volume"] + selected_features + ["target"]
        result_df = df_labeled[final_cols]
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_parquet(out_path)
        print(f"[OUTPUT] Saved labeled dataset to {out_path}")

        return {
            "model": model,
            "metrics": metrics,
            "top_features": selected_features,
            "feature_store": Path(feature_store),
            "training_set": out_path,
            "training_meta": training_meta,
        }

    def prepare_training_frame(
        self,
        feature_store: Path | str = FEATURE_STORE,
        input_df: Optional[pd.DataFrame] = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, List[str]]:
        """
        Load the feature store, apply labeling, and return (df, X, y, feature_cols).
        """
        if input_df is not None:
            df = input_df.copy()
            print(f"[LOAD] Using provided DataFrame with shape {df.shape}")
        else:
            df = self._load_features(feature_store)
        df_labeled = self._label_dataset(df)

        exclude_cols = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "turnover",
            "timestamp",
            "target",
            "regime",
        ]
        feature_cols = [col for col in df_labeled.columns if col not in exclude_cols]
        X = df_labeled[feature_cols]
        y = df_labeled["target"].astype(int)
        return df_labeled, X, y, feature_cols

    def _load_features(self, feature_store: Path | str) -> pd.DataFrame:
        path = Path(feature_store)
        if not path.exists():
            raise FileNotFoundError(f"Feature store not found: {path}")
        df = pd.read_parquet(path)
        print(f"[LOAD] Loaded feature matrix with shape {df.shape} from {path}")
        return df

    def _label_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if self.use_dynamic_targets:
            atr_col = f"ATR_{self.volatility_lookback}"
            df["target"] = get_dynamic_labels(
                df, atr_col=atr_col, tp_mult=self.tp_atr_mult, sl_mult=self.sl_atr_mult, horizon=self.horizon
            )
        else:
            df["target"] = get_triple_barrier_labels(df, self.tp_pct, self.sl_pct, self.horizon)
        df = df.dropna(subset=["target"])

        wins = int((df["target"] == 1).sum())
        losses = int((df["target"] == 0).sum())
        print(f"[LABELS] Wins: {wins} | Losses: {losses} | Win rate: {wins / len(df):.2%}")
        if wins < 10:
            print("[LABELS] Warning: only a handful of positive samples detected.")
        return df

    def train_primary_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        min_recall: float = PRIMARY_RECALL_TARGET,
        class_weight: Optional[Dict[int, float]] = None,
    ) -> Dict[str, object]:
        """
        Train a high-recall RandomForestClassifier to propose trade opportunities.
        """
        split_idx = int(len(X) * self.train_split)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        rf = RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=4,
            max_features="sqrt",
            class_weight=class_weight or {0: 1.0, 1: 2.5},
            random_state=self.random_state,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        metrics = {
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "accuracy": accuracy_score(y_test, y_pred),
        }
        print(
            f"[PRIMARY] Precision={metrics['precision']:.4f} "
            f"Recall={metrics['recall']:.4f} Accuracy={metrics['accuracy']:.4f}"
        )
        if metrics["recall"] < min_recall:
            print(
                f"[PRIMARY] Warning: recall {metrics['recall']:.2f} below target "
                f"{min_recall:.2f}. Consider retuning hyperparameters."
            )

        full_signals = pd.Series(rf.predict(X), index=X.index, name="primary_signal").astype(np.int8)
        probabilities = pd.Series(rf.predict_proba(X)[:, 1], index=X.index, name="primary_probability").astype(
            np.float32
        )

        return {
            "model": rf,
            "metrics": metrics,
            "signals": full_signals,
            "probabilities": probabilities,
        }

    def train_meta_model(
        self,
        X: pd.DataFrame,
        primary_signal: pd.Series,
        y: pd.Series,
        probability_threshold: float = META_PROB_THRESHOLD,
    ) -> Dict[str, object]:
        """
        Train the HistGradientBoosting meta model that filters primary signals.
        """
        meta_trainer = MetaModelTrainer(
            probability_threshold=probability_threshold,
            train_split=self.train_split,
            random_state=self.random_state,
        )
        model, metrics, success_prob = meta_trainer.fit(X, primary_signal, y)
        print(
            f"[META] Precision={metrics['precision']:.4f} "
            f"Recall={metrics['recall']:.4f} Accuracy={metrics['accuracy']:.4f}"
        )

        return {
            "trainer": meta_trainer,
            "model": model,
            "metrics": metrics,
            "train_probabilities": success_prob,
        }

    def _train_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_name: str,
        physics_features: Sequence[str],
    ):
        split_idx = int(len(X) * self.train_split)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        scheduler = TrainingScheduler()
        entropy_signal = self._feature_mean(X, "entropy_200", default=0.8)
        volatility_feature = next((f for f in physics_features if "volatility" in f), "volatility")
        volatility_signal = self._feature_mean(X, volatility_feature, default=1.0)
        depth = scheduler.suggest_training_depth(entropy_signal, max(volatility_signal, 1e-6))

        clf = MixtureOfExpertsEnsemble(
            physics_features=physics_features,
            random_state=self.random_state,
            trend_estimators=depth["n_estimators"],
            gating_epochs=depth["epochs"],
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # Calculate standard metrics
        metrics = {
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "accuracy": accuracy_score(y_test, y_pred),
        }
        print(
            f"[MODEL] Precision={metrics['precision']:.4f} "
            f"Recall={metrics['recall']:.4f} Accuracy={metrics['accuracy']:.4f}"
        )
        
        pnl_matrix = profit_weighted_confusion_matrix(
            y_test.values, 
            y_pred, 
            returns=None,
            tp_pct=self.tp_pct, 
            sl_pct=self.sl_pct
        )
        print(f"\n[FINANCIAL] Profit-Weighted Confusion Matrix ({model_name}):")
        print(pnl_matrix)
        print(f"Total Theoretical PnL: {pnl_matrix.values.sum():.4f}\n")

        return metrics, clf, {"training_depth": depth, "physics_features": physics_features} 

    def _determine_physics_features(self, columns: Sequence[str]) -> List[str]:
        required = []
        for feature in ("hurst_200", "entropy_200"):
            if feature not in columns:
                raise ValueError(f"Required chaos feature '{feature}' missing from feature matrix.")
            required.append(feature)

        volatility_candidates = (
            "volatility_200",
            "volatility",
            "volatility_20",
            "rolling_std_200",
            "rolling_std_20",
        )
        for candidate in volatility_candidates:
            if candidate in columns:
                required.append(candidate)
                break
        else:
            raise ValueError("No volatility proxy available for Mixture-of-Experts gating network.")

        return required

    def _feature_mean(self, X: pd.DataFrame, column: str, default: float) -> float:
        if column not in X.columns:
            return default
        series = pd.to_numeric(X[column], errors="coerce")
        if series.notna().any():
            return float(series.mean())
        return default

    def _save_feature_importance(self, features: List[str], importances: np.ndarray, model_name: str) -> None:
        feat_imp_df = pd.DataFrame({"Feature": features, "Importance": importances}).sort_values(
            "Importance", ascending=True
        )
        fig = px.bar(
            feat_imp_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title=f"Top Features ({model_name}) - Mutual Information",
            color="Importance",
            color_continuous_scale="Viridis",
        )
        filename = f"{model_name}_feature_importance.html"
        fig.write_html(filename)
        print(f"[MODEL] Feature importance chart saved to {filename}")
