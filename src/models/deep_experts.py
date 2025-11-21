"""
Seven-Eye Visionary: Adaptive Deep Learning Expert for Topological Pattern Recognition.

This module implements a multi-scale convolutional neural network that learns to
identify market patterns across 7 different time horizons (3 hours to ~5 days).
The attention mechanism dynamically weights each scale based on market conditions,
learning to suppress macro-kernels during choppy markets and micro-kernels during
strong trends.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split


class AdaptiveConvExpert(nn.Module):
    """
    Seven-Eye Visionary: Multi-scale convolutional network with attention.
    
    Architecture:
    1. Latent Projection: Linear + BatchNorm1d + ReLU (tabular -> 3D manifold)
    2. Multi-Scale Conv1d: 7 parallel branches with kernels [3, 5, 9, 17, 33, 65, 129]
    3. Squeeze-and-Excitation: Attention mechanism to weight each scale
    4. LSTM: Temporal integration of fused features
    5. Head: Linear -> Sigmoid for binary classification
    
    Parameters
    ----------
    n_features : int
        Number of input features (tabular data dimension).
    hidden_dim : int, optional
        Hidden dimension for latent projection (default: 32).
    sequence_length : int, optional
        Sequence length after reshaping (default: 16).
    lstm_hidden : int, optional
        LSTM hidden size (default: 64).
    dropout : float, optional
        Dropout probability (default: 0.2).
    """
    
    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 32,
        sequence_length: int = 16,
        lstm_hidden: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.lstm_hidden = lstm_hidden
        
        # Latent Projection: (Batch, N_Features) -> (Batch, Hidden_Dim * Seq_Len)
        self.projection = nn.Sequential(
            nn.Linear(n_features, hidden_dim * sequence_length),
            nn.BatchNorm1d(hidden_dim * sequence_length),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Multi-Scale Convolution: 7 parallel branches
        # Kernels: [3, 5, 9, 17, 33, 65, 129] capture patterns from hours to days
        self.kernel_sizes = [3, 5, 9, 17, 33, 65, 129]
        self.conv_branches = nn.ModuleList([
            nn.Conv1d(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=k,
                padding='same',  # Maintain time-alignment
            )
            for k in self.kernel_sizes
        ])
        
        # Squeeze-and-Excitation Attention: Learn to weight each scale
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # Global pooling: (B, 7*H, S) -> (B, 7*H, 1)
            nn.Flatten(),
            nn.Linear(len(self.kernel_sizes) * hidden_dim, len(self.kernel_sizes)),
            nn.ReLU(),
            nn.Linear(len(self.kernel_sizes), len(self.kernel_sizes)),
            nn.Softmax(dim=1),  # Weights sum to 1
        )
        
        # LSTM: Temporal integration
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            dropout=0.0,  # Single layer, no dropout
        )
        
        # Classification Head
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, 1),
            nn.Sigmoid(),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Seven-Eye Visionary.
        
        Parameters
        ----------
        x : torch.Tensor
            Input features of shape (Batch, N_Features).
            
        Returns
        -------
        torch.Tensor
            Predicted probabilities of shape (Batch, 1).
        """
        batch_size = x.size(0)
        
        # 1. Latent Projection: (B, N_Features) -> (B, Hidden * Seq)
        projected = self.projection(x)
        
        # 2. Reshape to 3D: (B, Hidden * Seq) -> (B, Hidden, Seq)
        reshaped = projected.view(batch_size, self.hidden_dim, self.sequence_length)
        
        # 3. Multi-Scale Convolution: Apply 7 parallel branches
        conv_outputs = []
        for conv_branch in self.conv_branches:
            conv_out = F.relu(conv_branch(reshaped))  # (B, Hidden, Seq)
            conv_outputs.append(conv_out)
        
        # 4. Stack outputs: (B, 7*Hidden, Seq)
        stacked = torch.cat(conv_outputs, dim=1)
        
        # 5. Attention Aggregation: Learn weights for each scale
        attention_weights = self.attention(stacked)  # (B, 7)
        
        # 6. Weighted Fusion: Sum weighted conv outputs
        fused = torch.zeros_like(conv_outputs[0])  # (B, Hidden, Seq)
        for idx, conv_out in enumerate(conv_outputs):
            weight = attention_weights[:, idx:idx+1, None]  # (B, 1, 1)
            fused += weight * conv_out
        
        # 7. LSTM: Temporal integration
        # Transpose for LSTM: (B, Hidden, Seq) -> (B, Seq, Hidden)
        lstm_input = fused.transpose(1, 2)
        lstm_out, _ = self.lstm(lstm_input)  # (B, Seq, LSTM_Hidden)
        
        # 8. Take last timestep
        last_hidden = lstm_out[:, -1, :]  # (B, LSTM_Hidden)
        
        # 9. Classification Head
        output = self.head(last_hidden)  # (B, 1)
        
        return output


class TorchSklearnWrapper(BaseEstimator, ClassifierMixin):
    """
    Sklearn-compatible wrapper for PyTorch models.
    
    Handles training loop, early stopping, device management, and provides
    sklearn-compatible fit/predict_proba interface.
    
    Parameters
    ----------
    n_features : int
        Number of input features.
    hidden_dim : int, optional
        Hidden dimension for the neural network (default: 32).
    sequence_length : int, optional
        Sequence length for reshaping (default: 16).
    lstm_hidden : int, optional
        LSTM hidden size (default: 64).
    dropout : float, optional
        Dropout probability (default: 0.2).
    learning_rate : float, optional
        Learning rate for Adam optimizer (default: 0.001).
    batch_size : int, optional
        Training batch size (default: 64).
    max_epochs : int, optional
        Maximum training epochs (default: 100).
    patience : int, optional
        Early stopping patience (default: 10).
    validation_split : float, optional
        Fraction of data for validation (default: 0.15).
    random_state : int, optional
        Random seed (default: 42).
    """
    
    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 32,
        sequence_length: int = 16,
        lstm_hidden: int = 64,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        batch_size: int = 64,
        max_epochs: int = 100,
        patience: int = 10,
        validation_split: float = 0.15,
        random_state: int = 42,
    ):
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.lstm_hidden = lstm_hidden
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.validation_split = validation_split
        self.random_state = random_state
        
        # Device management: Auto-detect GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model_ = None
        self.classes_ = np.array([0, 1])
        
    def _initialize_model(self) -> None:
        """Initialize the neural network model."""
        self.model_ = AdaptiveConvExpert(
            n_features=self.n_features,
            hidden_dim=self.hidden_dim,
            sequence_length=self.sequence_length,
            lstm_hidden=self.lstm_hidden,
            dropout=self.dropout,
        ).to(self.device)
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> "TorchSklearnWrapper":
        """
        Train the neural network with early stopping.
        
        Parameters
        ----------
        X : np.ndarray
            Training features of shape (n_samples, n_features).
        y : np.ndarray
            Training labels of shape (n_samples,).
            
        Returns
        -------
        self : TorchSklearnWrapper
            Fitted estimator.
        """
        # Initialize model
        self._initialize_model()
        
        # Train/validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=self.validation_split,
            random_state=self.random_state,
            stratify=y,
        )
        
        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.FloatTensor(y_val).unsqueeze(1).to(self.device)
        
        # Optimizer and loss
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        criterion = nn.BCELoss()
        
        # Training loop with early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.max_epochs):
            self.model_.train()
            
            # Mini-batch training
            n_samples = X_train_t.size(0)
            indices = torch.randperm(n_samples)
            
            train_loss = 0.0
            n_batches = 0
            
            for start_idx in range(0, n_samples, self.batch_size):
                end_idx = min(start_idx + self.batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]
                
                X_batch = X_train_t[batch_indices]
                y_batch = y_train_t[batch_indices]
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.model_(X_batch)
                loss = criterion(outputs, y_batch)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                n_batches += 1
            
            train_loss /= n_batches
            
            # Validation
            self.model_.eval()
            with torch.no_grad():
                val_outputs = self.model_(X_val_t)
                val_loss = criterion(val_outputs, y_val_t).item()
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model state
                self.best_state_dict_ = self.model_.state_dict().copy()
            else:
                patience_counter += 1
                
            if patience_counter >= self.patience:
                # Restore best model
                self.model_.load_state_dict(self.best_state_dict_)
                break
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Parameters
        ----------
        X : np.ndarray
            Features of shape (n_samples, n_features).
            
        Returns
        -------
        np.ndarray
            Probabilities of shape (n_samples, 2) for [class_0, class_1].
        """
        if self.model_ is None:
            raise RuntimeError("Model must be fitted before predict_proba.")
        
        self.model_.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            probs_class_1 = self.model_(X_t).cpu().numpy()
        
        # Return probabilities for both classes
        probs_class_0 = 1.0 - probs_class_1
        return np.hstack([probs_class_0, probs_class_1])
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Parameters
        ----------
        X : np.ndarray
            Features of shape (n_samples, n_features).
            
        Returns
        -------
        np.ndarray
            Predicted labels of shape (n_samples,).
        """
        probs = self.predict_proba(X)
        return (probs[:, 1] > 0.5).astype(int)


__all__ = ["AdaptiveConvExpert", "TorchSklearnWrapper"]
