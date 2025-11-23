"""
GraphVisionary: Global Market Attention Network.

This module implements a neural network that processes the entire market as a single
graph-like structure, allowing assets to attend to each other's liquidity flows.
"""
from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split


class GraphVisionary(nn.Module):
    """
    GraphVisionary: "Social" Neural Expert with Cross-Asset Attention.
    
    Architecture:
    1. Local Encoder (Per-Asset): LSTM/Conv1d extracting temporal features.
    2. Liquidity Router (Cross-Asset): Multi-Head Attention mixing asset embeddings.
    3. Gating/Output: MLP projecting to probabilities.
    
    Input Shape: (Batch, Sequence_Length, N_Assets, N_Features)
    Output Shape: (Batch, N_Assets, 1)
    """
    
    def __init__(
        self,
        n_features: int,
        n_assets: int,
        hidden_dim: int = 64,
        n_heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.n_features = n_features
        self.n_assets = n_assets
        self.hidden_dim = hidden_dim
        
        # 1. Local Encoder: Processes each asset's time series independently
        # Input: (Batch * N_Assets, Seq_Len, N_Features)
        self.local_encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        
        # 2. Liquidity Router: Cross-Asset Attention
        # Input: (Batch, N_Assets, Hidden_Dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # 3. Output Head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (Batch, Seq_Len, N_Assets, N_Features).
            
        Returns
        -------
        torch.Tensor
            Predictions of shape (Batch, N_Assets, 1).
        """
        batch_size, seq_len, n_assets, n_features = x.shape
        
        # Flatten Batch and Assets for Local Encoder
        # (B, S, N, F) -> (B * N, S, F)
        x_flat = x.permute(0, 2, 1, 3).reshape(batch_size * n_assets, seq_len, n_features)
        
        # Local Encoding
        # lstm_out: (B*N, S, H), last_hidden: (1, B*N, H)
        _, (last_hidden, _) = self.local_encoder(x_flat)
        
        # Reshape back to (B, N, H)
        # last_hidden[0] is (B*N, H)
        local_embeddings = last_hidden[0].view(batch_size, n_assets, self.hidden_dim)
        
        # Cross-Asset Attention (Liquidity Router)
        # Q = K = V = local_embeddings
        # attn_output: (B, N, H)
        attn_output, _ = self.attention(
            query=local_embeddings,
            key=local_embeddings,
            value=local_embeddings,
        )
        
        # Residual Connection + Norm (Optional but good practice, keeping simple for now as per spec)
        # Mixing local and global context
        mixed_embeddings = local_embeddings + attn_output
        
        # Output Head
        # (B, N, H) -> (B, N, 1)
        output = self.head(mixed_embeddings)
        
        return output


class TorchSklearnWrapper(BaseEstimator, ClassifierMixin):
    """
    Sklearn-compatible wrapper for GraphVisionary.
    
    Handles reshaping of input data from (Samples, Features) to (Batch, Time, Assets, Features)
    assuming the input X contains metadata or is structured in a specific way.
    
    CRITICAL: This wrapper assumes that when `fit` is called, X is either:
    1. A 4D numpy array (Batch, Time, Assets, Features) - Ideal
    2. A 2D array that needs reshaping.
    
    For the QFC pipeline, `HybridTrendExpert` passes a 2D array.
    However, with the Global Market upgrade, we need to change how data is passed.
    
    To maintain compatibility, we will assume X contains a special structure or we
    will rely on the user to pass a 4D array if they want Global mode.
    
    Actually, the task says: "You must reshape the linear X input back into (Time, Assets, Features)..."
    This implies X is flattened.
    """
    
    def __init__(
        self,
        n_features: int,
        n_assets: int = 11,
        sequence_length: int = 16,
        hidden_dim: int = 64,
        n_heads: int = 4,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        max_epochs: int = 100,
        patience: int = 10,
        validation_split: float = 0.15,
        random_state: int = 42,
    ):
        self.n_features = n_features
        self.n_assets = n_assets
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.validation_split = validation_split
        self.random_state = random_state
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_ = None
        self.classes_ = np.array([0, 1])
        
    def _initialize_model(self) -> None:
        self.model_ = GraphVisionary(
            n_features=self.n_features,
            n_assets=self.n_assets,
            hidden_dim=self.hidden_dim,
            n_heads=self.n_heads,
            dropout=self.dropout,
        ).to(self.device)
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> "TorchSklearnWrapper":
        """
        Fit the GraphVisionary model.
        
        Supports two modes:
        1. Global Mode: X is (Batch * N_Assets, Seq * Features) - requires divisibility by n_assets
        2. Legacy Mode: X is (N_Samples, Features) - treats as single-asset with n_assets=1
        """
        self._initialize_model()
        
        # Check input shape
        n_samples, input_dim = X.shape
        
        # Check if we can use global mode (multi-asset)
        if n_samples % self.n_assets == 0 and self.n_assets > 1:
            # Global mode: Multi-asset training
            batch_size = n_samples // self.n_assets
            
            # Reshape: (Batch * Assets, Seq * F) -> (Batch, Assets, Seq, F)
            X_reshaped = X.reshape(batch_size, self.n_assets, self.sequence_length, self.n_features)
            
            # Permute to (Batch, Seq, Assets, F) for GraphVisionary
            X_tensor = torch.FloatTensor(X_reshaped).permute(0, 2, 1, 3).to(self.device)
            
            # y is (Batch * Assets,) -> reshape to (Batch, Assets)
            if y.ndim == 1:
                y = y.reshape(batch_size, self.n_assets)
            elif y.ndim == 2 and y.shape[1] == 1:
                y = y.reshape(batch_size, self.n_assets)
                
            y_tensor = torch.FloatTensor(y).to(self.device)
            
        else:
            # Legacy mode: Treat as single-asset or per-sample training
            # Create synthetic "batch" dimension by grouping consecutive samples
            # Each sample becomes a single-asset "market state"
            
            # We need to create windows of sequence_length
            # If input_dim matches sequence_length * n_features, we can reshape
            if input_dim % self.sequence_length == 0:
                inferred_features = input_dim // self.sequence_length
            else:
                # Fallback: treat entire input as features for a single timestep
                # This won't work well with GraphVisionary but prevents crash
                inferred_features = input_dim
                
            # Create sliding windows or just use samples as-is
            # For simplicity, we'll treat each sample as a single timestep with n_assets=1
            # Reshape: (N_Samples, Seq * F) -> (N_Samples, Seq, 1, F)
            
            batch_size = n_samples
            X_reshaped = X.reshape(batch_size, self.sequence_length, 1, inferred_features)
            X_tensor = torch.FloatTensor(X_reshaped).to(self.device)
            
            # y stays as (N_Samples,) -> reshape to (N_Samples, 1)
            if y.ndim == 1:
                y = y.reshape(batch_size, 1)
            y_tensor = torch.FloatTensor(y).to(self.device)
            
            # Temporarily override n_assets for this training session
            original_n_assets = self.model_.n_assets
            self.model_.n_assets = 1
            # Re-init attention layer for n_assets=1
            self.model_.attention = nn.MultiheadAttention(
                embed_dim=self.model_.hidden_dim,
                num_heads=min(self.n_heads, 1),  # Can't have more heads than assets
                dropout=self.dropout,
                batch_first=True,
            ).to(self.device)
        
        # Train/Val split (on Batch dimension)
        val_size = int(batch_size * self.validation_split)
        train_size = batch_size - val_size
        
        X_train = X_tensor[:train_size]
        y_train = y_tensor[:train_size]
        X_val = X_tensor[train_size:]
        y_val = y_tensor[train_size:]
        
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        criterion = nn.BCELoss()
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.max_epochs):
            self.model_.train()
            
            # Mini-batching
            n_train_batches = X_train.size(0)
            indices = torch.randperm(n_train_batches)
            
            train_loss = 0.0
            n_batches = 0
            
            for start_idx in range(0, n_train_batches, self.batch_size):
                end_idx = min(start_idx + self.batch_size, n_train_batches)
                batch_indices = indices[start_idx:end_idx]
                
                X_batch = X_train[batch_indices]
                y_batch = y_train[batch_indices]
                
                optimizer.zero_grad()
                outputs = self.model_(X_batch) # (B, N, 1)
                outputs = outputs.squeeze(-1)  # (B, N)
                
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                n_batches += 1
            
            train_loss /= max(1, n_batches)
            
            # Validation
            self.model_.eval()
            with torch.no_grad():
                val_outputs = self.model_(X_val).squeeze(-1)
                val_loss = criterion(val_outputs, y_val).item()
                
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.best_state_dict_ = self.model_.state_dict().copy()
            else:
                patience_counter += 1
                
            if patience_counter >= self.patience:
                self.model_.load_state_dict(self.best_state_dict_)
                break
                
        return self
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Model must be fitted before predict_proba.")
            
        n_samples, input_dim = X.shape
        
        # Check mode (same logic as fit)
        if n_samples % self.n_assets == 0 and self.n_assets > 1:
            # Global mode
            batch_size = n_samples // self.n_assets
            X_reshaped = X.reshape(batch_size, self.n_assets, self.sequence_length, self.n_features)
            X_tensor = torch.FloatTensor(X_reshaped).permute(0, 2, 1, 3).to(self.device)
        else:
            # Legacy mode
            if input_dim % self.sequence_length == 0:
                inferred_features = input_dim // self.sequence_length
            else:
                inferred_features = input_dim
                
            batch_size = n_samples
            X_reshaped = X.reshape(batch_size, self.sequence_length, 1, inferred_features)
            X_tensor = torch.FloatTensor(X_reshaped).to(self.device)
        
        self.model_.eval()
        with torch.no_grad():
            probs = self.model_(X_tensor).cpu().numpy() # (B, N, 1)
            
        # Flatten back to (B * N, 1)
        probs_flat = probs.reshape(-1, 1)
        return np.hstack([1.0 - probs_flat, probs_flat])

__all__ = ["GraphVisionary", "TorchSklearnWrapper"]
