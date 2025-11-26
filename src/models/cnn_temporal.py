"""
Temporal CNN expert operating on Tensor-Flex latents.
"""
from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:  # pragma: no cover - handled at runtime
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None
    TORCH_IMPORT_ERROR = exc
else:  # pragma: no cover - defined only when torch is available
    TORCH_IMPORT_ERROR = None


logger = logging.getLogger(__name__)


def _check_torch_available() -> None:
    if torch is None:
        raise ImportError(
            "PyTorch is required for CNNExpert but is not available. "
            "Install torch>=2.0.0 to enable the temporal CNN expert."
        ) from TORCH_IMPORT_ERROR


class TemporalConvBlock(nn.Module):
    """Residual temporal convolution block with dilation."""

    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(channels)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation)
        self.norm2 = nn.BatchNorm1d(channels)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.drop1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act2(x)
        x = self.drop2(x)
        return x + residual


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation style module."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.pool(x).squeeze(-1)
        weights = self.mlp(weights).unsqueeze(-1)
        return x * weights


class TemporalConvNet(nn.Module):
    """
    Temporal ConvNet backbone that consumes [batch, channels, length] tensors.
    """

    def __init__(
        self,
        n_channels: int,
        mid_channels: int = 128,
        dilations: Sequence[int] = (1, 2, 4, 8, 16, 32),
        dropout: float = 0.2,
        hidden_dim: int = 64,
        n_outputs: int = 1,
        use_attention: bool = True,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Conv1d(n_channels, mid_channels, kernel_size=1),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [TemporalConvBlock(mid_channels, dilation=d, dropout=dropout) for d in dilations]
        )
        self.attention = ChannelAttention(mid_channels) if use_attention else None
        self.output_head = nn.Sequential(
            nn.Linear(mid_channels * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        if self.attention is not None:
            x = self.attention(x)
        avg_pool = torch.mean(x, dim=-1)
        max_pool, _ = torch.max(x, dim=-1)
        pooled = torch.cat([avg_pool, max_pool], dim=1)
        return self.output_head(pooled)


@dataclass
class CNNExpert(BaseEstimator, ClassifierMixin):
    """
    Sklearn-style wrapper around the TemporalConvNet.
    """

    window_length: int = 64
    mid_channels: int = 128
    hidden_dim: int = 64
    dropout: float = 0.2
    lr: float = 1e-3
    epochs: int = 30
    batch_size: int = 64
    random_state: int = 42
    fill_strategy: str = "nan"
    patience: int = 5
    artifacts_path: Optional[Path | str] = None
    dilations: Sequence[int] = field(default_factory=lambda: (1, 2, 4, 8, 16, 32))
    use_attention: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.artifacts_path, (str, Path)):
            self.artifacts_path = Path(self.artifacts_path)
        else:
            self.artifacts_path = None
        self.scaler_: Optional[StandardScaler] = None
        self.model_: Optional[TemporalConvNet] = None
        self.feature_names_: List[str] = []
        self.device_ = None
        self.loss_history_: List[float] = []

    # ------------------------------------------------------------------ #
    # Sklearn API
    # ------------------------------------------------------------------ #
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "CNNExpert":
        _check_torch_available()
        df, target = self._prepare_frames(X, y)
        if df.shape[0] < self.window_length:
            raise ValueError(
                f"CNNExpert requires at least {self.window_length} samples; received {df.shape[0]}."
            )
        self._set_random_state()
        self.feature_names_ = list(df.columns)
        self.scaler_ = StandardScaler()
        scaled = self.scaler_.fit_transform(df.values).astype(np.float32)
        windows, labels, _ = self._build_windows(scaled, target.values)
        dataset = TensorDataset(
            torch.from_numpy(windows),
            torch.from_numpy(labels.astype(np.float32)).unsqueeze(1),
        )
        val_len = max(1, int(0.2 * len(dataset)))
        train_len = len(dataset) - val_len
        if train_len == 0:
            raise ValueError("Not enough CNN training windows after validation split.")
        generator = torch.Generator().manual_seed(self.random_state)
        train_ds, val_ds = torch.utils.data.random_split(dataset, [train_len, val_len], generator=generator)

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_ = TemporalConvNet(
            n_channels=windows.shape[1],  # Dynamically set from input windows [Batch, Channels, Length]
            mid_channels=self.mid_channels,
            dilations=self.dilations,
            dropout=self.dropout,
            hidden_dim=self.hidden_dim,
            n_outputs=1,
            use_attention=self.use_attention,
        ).to(self.device_)

        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=self.lr)
        criterion = nn.BCEWithLogitsLoss()
        best_state = None
        best_loss = math.inf
        patience_counter = 0
        self.loss_history_.clear()

        logger.info(
            "CNNExpert training with %s windows (%s val), params=%s",
            train_len,
            val_len,
            sum(p.numel() for p in self.model_.parameters()),
        )

        for epoch in range(self.epochs):
            self.model_.train()
            epoch_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device_)
                batch_y = batch_y.to(self.device_)
                optimizer.zero_grad()
                logits = self.model_(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch_x.size(0)
            epoch_loss /= max(1, train_len)
            val_loss = self._evaluate_loss(val_loader, criterion)
            self.loss_history_.append(val_loss)

            if val_loss < best_loss - 1e-4:
                best_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model_.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info("CNNExpert early stopping at epoch %s (val_loss=%.5f).", epoch + 1, val_loss)
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)

        val_probs, val_targets = self._collect_probs(val_loader)
        val_auc = float("nan")
        try:
            val_auc = roc_auc_score(val_targets, val_probs)
        except ValueError:
            logger.warning("CNNExpert AUC undefined (single-class validation set).")
        val_logloss = log_loss(val_targets, np.clip(val_probs, 1e-6, 1 - 1e-6))
        logger.info("CNNExpert validation metrics: AUC=%.4f, logloss=%.4f", val_auc, val_logloss)

        if self.artifacts_path:
            self.save(self.artifacts_path)

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model_ is None or self.scaler_ is None:
            raise RuntimeError("CNNExpert must be fitted before calling predict_proba.")
        df = self._prepare_inference_frame(X)
        scaled = self.scaler_.transform(df.values).astype(np.float32)
        windows, _, indices = self._build_windows(scaled, np.zeros(df.shape[0], dtype=np.float32))
        if windows.size == 0:
            probs = np.full((df.shape[0], 2), np.nan, dtype=np.float32)
            return probs

        loader = DataLoader(torch.from_numpy(windows), batch_size=self.batch_size, shuffle=False)
        self.model_.eval()
        preds: List[np.ndarray] = []
        with torch.no_grad():
            for batch_x in loader:
                batch_x = batch_x.to(self.device_)
                logits = self.model_(batch_x)
                probs_batch = torch.sigmoid(logits).cpu().numpy().ravel()
                preds.append(probs_batch)
        flat_probs = np.concatenate(preds, axis=0)
        aligned = self._align_probabilities(flat_probs, df.shape[0])
        logger.debug(
            "CNNExpert mean probability=%.4f (valid windows=%s)",
            np.nanmean(aligned[:, 1]) if np.isfinite(aligned[:, 1]).any() else float("nan"),
            len(flat_probs),
        )
        return aligned

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: Path | str) -> None:
        if self.model_ is None or self.scaler_ is None:
            raise RuntimeError("CNNExpert must be fitted before saving.")
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        artifact = target / "cnn_expert.pt"
        payload = {
            "state_dict": self.model_.state_dict(),
            "feature_names": self.feature_names_,
            "scaler": self.scaler_,
            "config": {
                "window_length": self.window_length,
                "mid_channels": self.mid_channels,
                "hidden_dim": self.hidden_dim,
                "dropout": self.dropout,
                "lr": self.lr,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "random_state": self.random_state,
                "fill_strategy": self.fill_strategy,
                "patience": self.patience,
                "dilations": tuple(self.dilations),
                "use_attention": self.use_attention,
            },
        }
        torch.save(payload, artifact)
        logger.info("CNNExpert artifacts saved to %s", artifact)

    @classmethod
    def load(cls, path: Path | str) -> "CNNExpert":
        _check_torch_available()
        artifact = Path(path)
        if artifact.is_dir():
            artifact = artifact / "cnn_expert.pt"
        if not artifact.exists():
            raise FileNotFoundError(f"CNNExpert artifact not found at {artifact}")
        payload = torch.load(artifact, map_location="cpu")
        config = payload.get("config", {})
        expert = cls(**config)
        expert.feature_names_ = list(payload["feature_names"])
        expert.scaler_ = payload["scaler"]
        expert.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        expert.model_ = TemporalConvNet(
            n_channels=len(expert.feature_names_),
            mid_channels=expert.mid_channels,
            dilations=expert.dilations,
            dropout=expert.dropout,
            hidden_dim=expert.hidden_dim,
            n_outputs=1,
            use_attention=expert.use_attention,
        )
        expert.model_.load_state_dict(payload["state_dict"])
        expert.model_.to(expert.device_)
        expert.model_.eval()
        return expert

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _prepare_frames(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("CNNExpert expects Tensor-Flex latents as a pandas DataFrame.")
        df = X.sort_index().copy()
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        target = pd.Series(y).reindex(df.index).fillna(method="ffill").fillna(0.0)
        return df, target

    def _prepare_inference_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("CNNExpert expects Tensor-Flex latents as a pandas DataFrame.")
        df = X.sort_index().copy()
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        for col in self.feature_names_:
            if col not in df.columns:
                df[col] = 0.0
        df = df[self.feature_names_]
        return df

    def _build_windows(
        self,
        data: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_samples, n_features = data.shape
        if n_samples < self.window_length:
            return np.empty((0, n_features, 0)), np.empty((0,)), np.empty((0,))

        windows = []
        y_aligned = []
        indices = []
        for end in range(self.window_length - 1, n_samples):
            start = end - self.window_length + 1
            segment = data[start : end + 1].T  # [channels, length]
            windows.append(segment)
            y_aligned.append(labels[end])
            indices.append(end)
        stacked_windows = np.stack(windows, axis=0)
        targets = np.asarray(y_aligned, dtype=np.float32)
        indices = np.asarray(indices, dtype=np.int32)
        return stacked_windows, targets, indices

    def _align_probabilities(self, probs: np.ndarray, n_samples: int) -> np.ndarray:
        aligned = np.full((n_samples, 2), np.nan, dtype=np.float32)
        start = self.window_length - 1
        aligned[start:, 1] = probs
        aligned[start:, 0] = 1.0 - probs
        if self.fill_strategy == "pad_first_valid" and probs.size > 0:
            first = probs[0]
            aligned[:start, 1] = first
            aligned[:start, 0] = 1.0 - first
        return aligned

    def _evaluate_loss(self, loader: DataLoader, criterion: nn.Module) -> float:
        self.model_.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device_)
                batch_y = batch_y.to(self.device_)
                logits = self.model_(batch_x)
                loss = criterion(logits, batch_y)
                total += loss.item() * batch_x.size(0)
                count += batch_x.size(0)
        self.model_.train()
        return total / max(1, count)

    def _collect_probs(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        self.model_.eval()
        probs, targets = [], []
        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device_)
                logits = self.model_(batch_x)
                probs.append(torch.sigmoid(logits).cpu().numpy().ravel())
                targets.append(batch_y.cpu().numpy().ravel())
        return np.concatenate(probs), np.concatenate(targets)

    def _set_random_state(self) -> None:
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)


__all__ = ["TemporalConvNet", "CNNExpert"]
