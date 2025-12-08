"""
Scenario Dataset for Conditional Diffusion Training.

Builds (past_window, future_window) pairs from time series data.

Author: QFC System - Diffusion Architecture
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ScenarioDataset(Dataset):
    """
    Dataset yielding (past_window, future_window) pairs for scenario model training.
    
    Args:
        past_features: [N, C_past] past feature array (multiple channels)
        future_target: [N, C_target] future target array (usually returns)
        L_past: Length of past conditioning window
        H_future: Forecast horizon
        stride: Step between samples (default: 1)
        normalize: Whether to normalize features
    """
    
    def __init__(
        self,
        past_features: np.ndarray,
        future_target: np.ndarray,
        L_past: int = 96,
        H_future: int = 12,
        stride: int = 1,
        normalize: bool = True,
    ):
        self.L_past = L_past
        self.H_future = H_future
        self.stride = stride
        
        # Ensure 2D
        if past_features.ndim == 1:
            past_features = past_features[:, np.newaxis]
        if future_target.ndim == 1:
            future_target = future_target[:, np.newaxis]
        
        self.past_features = past_features
        self.future_target = future_target
        
        # Store normalization stats
        self.normalize = normalize
        if normalize:
            self.past_mean = past_features.mean(axis=0, keepdims=True)
            self.past_std = past_features.std(axis=0, keepdims=True) + 1e-8
            self.future_mean = future_target.mean(axis=0, keepdims=True)
            self.future_std = future_target.std(axis=0, keepdims=True) + 1e-8
        else:
            self.past_mean = np.zeros((1, past_features.shape[1]))
            self.past_std = np.ones((1, past_features.shape[1]))
            self.future_mean = np.zeros((1, future_target.shape[1]))
            self.future_std = np.ones((1, future_target.shape[1]))
        
        # Compute valid indices
        # Need L_past bars of past + H_future bars of future
        n_samples = len(past_features)
        self.valid_indices = list(range(L_past, n_samples - H_future, stride))
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get (past_window, future_window) pair.
        
        Returns:
            past_window: [C_past, L_past]
            future_window: [C_target, H_future]
        """
        t = self.valid_indices[idx]
        
        # Extract windows
        past = self.past_features[t - self.L_past:t]  # [L_past, C_past]
        future = self.future_target[t:t + self.H_future]  # [H_future, C_target]
        
        # Normalize
        if self.normalize:
            past = (past - self.past_mean) / self.past_std
            future = (future - self.future_mean) / self.future_std
        
        # Transpose to [C, L] format
        past = past.T  # [C_past, L_past]
        future = future.T  # [C_target, H_future]
        
        return (
            torch.tensor(past, dtype=torch.float32),
            torch.tensor(future, dtype=torch.float32),
        )
    
    def get_normalization_stats(self) -> Dict[str, np.ndarray]:
        """Get normalization statistics for inference."""
        return {
            'past_mean': self.past_mean,
            'past_std': self.past_std,
            'future_mean': self.future_mean,
            'future_std': self.future_std,
        }
    
    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        past_columns: List[str],
        future_columns: List[str],
        L_past: int = 96,
        H_future: int = 12,
        stride: int = 1,
        normalize: bool = True,
    ) -> 'ScenarioDataset':
        """
        Create dataset from DataFrame.
        
        Args:
            df: DataFrame with time-indexed data
            past_columns: Column names for past features
            future_columns: Column names for future target
            L_past: Past window length
            H_future: Future horizon
            stride: Sample stride
            normalize: Whether to normalize
            
        Returns:
            ScenarioDataset instance
        """
        past_features = df[past_columns].values
        future_target = df[future_columns].values
        
        return cls(
            past_features=past_features,
            future_target=future_target,
            L_past=L_past,
            H_future=H_future,
            stride=stride,
            normalize=normalize,
        )
    
    @classmethod
    def from_returns(
        cls,
        returns: np.ndarray,
        L_past: int = 96,
        H_future: int = 12,
        stride: int = 1,
        add_features: bool = True,
        normalize: bool = True,
    ) -> 'ScenarioDataset':
        """
        Create dataset from returns array with automatically computed features.
        
        Args:
            returns: [N] or [N, 1] log returns
            L_past: Past window length
            H_future: Future horizon
            stride: Sample stride
            add_features: Add derived features (volatility, momentum, etc.)
            normalize: Whether to normalize
            
        Returns:
            ScenarioDataset instance
        """
        returns = returns.flatten()
        n = len(returns)
        
        if add_features:
            # Compute additional features
            features = []
            
            # 1. Returns (primary)
            features.append(returns)
            
            # 2. Realized volatility (rolling std)
            for window in [5, 10, 20]:
                vol = pd.Series(returns).rolling(window).std().fillna(0).values
                features.append(vol)
            
            # 3. Momentum (cumulative returns)
            for window in [5, 10, 20]:
                cum_ret = pd.Series(returns).rolling(window).sum().fillna(0).values
                features.append(cum_ret)
            
            # 4. Abs returns (activity)
            abs_ret = np.abs(returns)
            features.append(abs_ret)
            
            # 5. Rolling mean
            mean_10 = pd.Series(returns).rolling(10).mean().fillna(0).values
            features.append(mean_10)
            
            past_features = np.column_stack(features)
        else:
            past_features = returns[:, np.newaxis]
        
        future_target = returns[:, np.newaxis]
        
        return cls(
            past_features=past_features,
            future_target=future_target,
            L_past=L_past,
            H_future=H_future,
            stride=stride,
            normalize=normalize,
        )


if __name__ == "__main__":
    print("[ScenarioDataset Test]")
    
    # Create synthetic data
    np.random.seed(42)
    n = 1000
    returns = np.random.randn(n) * 0.01
    
    # Create dataset
    dataset = ScenarioDataset.from_returns(
        returns,
        L_past=64,
        H_future=12,
        stride=1,
        add_features=True,
    )
    
    print(f"  Dataset size: {len(dataset)}")
    
    past, future = dataset[0]
    print(f"  Past shape: {past.shape}")
    print(f"  Future shape: {future.shape}")
    
    stats = dataset.get_normalization_stats()
    print(f"  Norm stats: past_mean shape = {stats['past_mean'].shape}")
    
    print("[OK] ScenarioDataset test passed!")
