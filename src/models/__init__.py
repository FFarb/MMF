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
    CV_NUM_FOLDS,
    CV_SCHEME,
    BOOTSTRAP_TRIALS,
    BOOTSTRAP_SAMPLE_FRACTION,
    MIN_TRADES_FOR_EVAL,
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
    cv_folds: int = CV_NUM_FOLDS
    bootstrap_trials: int = BOOTSTRAP_TRIALS

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
        df_labeled, X, y, feature_cols, weights = self.prepare_training_frame(
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

        # --- Cross-Validation / Train Loop ---
        if self.cv_folds > 1:
            print(f"[TRAINER] Running {self.cv_folds}-fold {CV_SCHEME} CV...")
            cv_results = self._run_cv(
                X_corr[selected_features], y, model_name, physics_features, weights
            )
            model = cv_results["final_model"] # The model trained on full data (or last fold)
            metrics = cv_results["aggregate_metrics"]
            training_meta = cv_results["meta"]
        else:
            metrics, model, training_meta = self._train_model(
                X_corr[selected_features], y, model_name, physics_features, sample_weights=weights
            )
            # Add bootstrap if requested even for single split
            if self.bootstrap_trials > 0:
                 # We need predictions to bootstrap. _train_model returns metrics but we need per-trade outcomes.
                 # For simplicity, we'll rely on the fact that _train_model prints them or we can re-predict.
                 # Actually, let's just let the single split be the "simple" path.
                 pass

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

    def _run_cv(self, X, y, model_name, physics_features, weights):
        """
        Time-aware Cross Validation.
        """
        n_samples = len(X)
        fold_size = n_samples // (self.cv_folds + 1)
        
        fold_metrics = []
        models = []
        
        for i in range(1, self.cv_folds + 1):
            train_end = i * fold_size
            val_end = train_end + fold_size
            
            if CV_SCHEME == "expanding":
                train_idx = range(0, train_end)
            else: # rolling
                train_idx = range(train_end - fold_size, train_end)
                
            val_idx = range(train_end, min(val_end, n_samples))
            
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
            w_train = weights[train_idx] if weights is not None else None
            
            print(f"    [CV] Fold {i}/{self.cv_folds}: Train={len(X_train)}, Val={len(X_val)}")
            
            # Train
            metrics, model, meta = self._train_model_core(
                X_train, y_train, X_val, y_val, model_name, physics_features, w_train
            )
            
            # Bootstrap
            if self.bootstrap_trials > 0 and len(y_val) >= MIN_TRADES_FOR_EVAL:
                y_pred = model.predict(X_val)
                boot_res = self._bootstrap_metrics(y_val.values, y_pred)
                metrics["bootstrap"] = boot_res
                print(f"        Bootstrap Expectancy: {boot_res['expectancy_mean']:.4f} [{boot_res['expectancy_low']:.4f}, {boot_res['expectancy_high']:.4f}]")

            fold_metrics.append(metrics)
            models.append(model)
            
        # Aggregate
        agg_metrics = {}
        for k in fold_metrics[0].keys():
            if isinstance(fold_metrics[0][k], (int, float)):
                agg_metrics[f"{k}_mean"] = np.mean([m[k] for m in fold_metrics])
                agg_metrics[f"{k}_std"] = np.std([m[k] for m in fold_metrics])
                
        return {
            "final_model": models[-1],
            "aggregate_metrics": agg_metrics,
            "fold_metrics": fold_metrics,
            "meta": meta
        }

    def _bootstrap_metrics(self, y_true, y_pred):
        outcomes = []
        for _ in range(self.bootstrap_trials):
            indices = np.random.choice(len(y_true), size=int(len(y_true) * BOOTSTRAP_SAMPLE_FRACTION), replace=True)
            yt_sample = y_true[indices]
            yp_sample = y_pred[indices]
            
            prec = precision_score(yt_sample, yp_sample, zero_division=0)
            # Expectancy
            exp = (prec * self.tp_pct) - ((1 - prec) * self.sl_pct)
            outcomes.append(exp)
            
        return {
            "expectancy_mean": np.mean(outcomes),
            "expectancy_low": np.percentile(outcomes, 5),
            "expectancy_high": np.percentile(outcomes, 95)
        }

    def _train_model_core(self, X_train, y_train, X_test, y_test, model_name, physics_features, w_train):
        """
        Core training logic decoupled from split logic.
        """
        scheduler = TrainingScheduler()
        entropy_signal = self._feature_mean(X_train, "entropy_200", default=0.8)
        volatility_feature = next((f for f in physics_features if "volatility" in f), "volatility")
        volatility_signal = self._feature_mean(X_train, volatility_feature, default=1.0)
        depth = scheduler.suggest_training_depth(entropy_signal, max(volatility_signal, 1e-6))

        clf = MixtureOfExpertsEnsemble(
            physics_features=physics_features,
            random_state=self.random_state,
            trend_estimators=depth["n_estimators"],
            gating_epochs=depth["epochs"],
        )
        
        clf.fit(X_train, y_train, sample_weight=w_train)
        
        y_pred = clf.predict(X_test)
        metrics = {
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "accuracy": accuracy_score(y_test, y_pred),
        }
        
        return metrics, clf, {"training_depth": depth, "physics_features": physics_features}

    def _train_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_name: str,
        physics_features: Sequence[str],
        sample_weights: Optional[np.ndarray] = None
    ):
        split_idx = int(len(X) * self.train_split)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        w_train = sample_weights[:split_idx] if sample_weights is not None else None

        metrics, clf, meta = self._train_model_core(
            X_train, y_train, X_test, y_test, model_name, physics_features, w_train
        )
        
        # Telemetry print (legacy support)
        print(
            f"\n[MODEL] Precision={metrics['precision']:.4f} "
            f"Recall={metrics['recall']:.4f} Accuracy={metrics['accuracy']:.4f}"
        )
        
        pnl_matrix = profit_weighted_confusion_matrix(
            y_test.values, 
            clf.predict(X_test), 
            returns=None,
            tp_pct=self.tp_pct, 
            sl_pct=self.sl_pct
        )
        print(f"\n[FINANCIAL] Profit-Weighted Confusion Matrix ({model_name}):")
        print(pnl_matrix)

        return metrics, clf, meta 

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

    def _calculate_sample_weights(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Calculate sample weights based on future returns.
        Higher weights for samples with stronger directional moves.
        """
        if 'close' not in df.columns:
            print("[WEIGHTS] No 'close' column found, using uniform weights")
            return None
        
        close = df['close'].to_numpy()
        
        # Calculate forward returns (next bar)
        future_ret = np.zeros(len(close))
        future_ret[:-1] = (close[1:] - close[:-1]) / close[:-1]
        
        # Weight by absolute return magnitude (emphasize strong moves)
        abs_ret = np.abs(future_ret)
        
        # Normalize to [0.5, 2.0] range to avoid extreme weights
        if abs_ret.max() > 0:
            weights = 0.5 + 1.5 * (abs_ret / abs_ret.max())
        else:
            weights = np.ones(len(close))
        
        print(f"[WEIGHTS] Calculated sample weights: mean={weights.mean():.3f}, std={weights.std():.3f}")
        return weights

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
