"""
End-to-end experiment runner for the meta-labeling architecture.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.config import FEATURE_STORE, META_PROB_THRESHOLD
from src.features import add_signal_interactions
from src.models import SniperModelTrainer


def compute_precision(signals: pd.Series, target: pd.Series) -> float:
    executed = signals != 0
    if executed.sum() == 0:
        return np.nan
    return (target.loc[executed] == 1).mean()


def build_equity_curve(close: pd.Series, signals: pd.Series) -> pd.Series:
    aligned_signals = signals.reindex(close.index).fillna(0)
    returns = close.pct_change().fillna(0)
    pnl = aligned_signals.shift(1).fillna(0) * returns
    return (1 + pnl).cumprod()


def main() -> None:
    trainer = SniperModelTrainer()
    df_labeled, X, y, _ = trainer.prepare_training_frame(feature_store=FEATURE_STORE)

    # 1) Primary model with high recall
    primary_result = trainer.train_primary_model(X, y)
    primary_signals = primary_result["signals"]

    # 2) Meta-model training with interaction terms
    enriched_features = add_signal_interactions(X.copy(), primary_signals, reference_df=df_labeled)
    meta_result = trainer.train_meta_model(enriched_features, primary_signals, y)
    meta_trainer = meta_result["trainer"]
    meta_model = meta_result["model"]

    success_probability = meta_trainer.predict_success_probability(meta_model, enriched_features, primary_signals)
    filtered_signals = meta_trainer.filter_signals(primary_signals, success_probability)

    split_idx = int(len(X) * trainer.train_split)
    test_slice = slice(split_idx, None)

    primary_test_signal = primary_signals.iloc[test_slice]
    meta_test_signal = filtered_signals.iloc[test_slice]
    target_test = y.iloc[test_slice]
    close_test = df_labeled["close"].iloc[test_slice]

    primary_precision = compute_precision(primary_test_signal, target_test)
    meta_precision = compute_precision(meta_test_signal, target_test)

    primary_curve = build_equity_curve(close_test, primary_test_signal)
    meta_curve = build_equity_curve(close_test, meta_test_signal)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=primary_curve.index,
            y=primary_curve,
            mode="lines",
            name="Primary Only",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=meta_curve.index,
            y=meta_curve,
            mode="lines",
            name=f"Meta-Labeled (>{META_PROB_THRESHOLD:.2f})",
        )
    )
    fig.update_layout(
        title="Primary vs Meta-Labeled Equity Curve",
        yaxis_title="Equity (base = 1.0)",
        xaxis_title="Timestamp",
        template="plotly_dark",
    )
    output_path = Path("meta_vs_primary_equity.html")
    fig.write_html(output_path)

    print("\n========== META-LABELING EXPERIMENT ==========")
    print(f"Primary trades (OOS): {(primary_test_signal != 0).sum()}")
    print(f"Meta trades (OOS): { (meta_test_signal != 0).sum() }")
    print(f"Primary precision (OOS): {primary_precision:.2%}")
    print(f"Meta precision (OOS): {meta_precision:.2%}")
    print(f"Equity chart saved to: {output_path.resolve()}")
    print("==============================================")


if __name__ == "__main__":
    main()
