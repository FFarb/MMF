"""
Diffusion Feature Denoiser.

High-level wrapper for integrating diffusion denoising into the feature pipeline.
Provides a clean API for denoising feature tensors before downstream model consumption.

Modes:
    - 'off': Pass through unchanged (no denoising)
    - 'inference': Run shortened reverse diffusion to denoise

Author: QFC System - Diffusion Architecture
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


class DiffusionFeatureDenoiser:
    """
    Feature-level diffusion denoiser.
    
    Wraps DiffusionDenoiserModel for easy integration into feature pipelines.
    Can denoise individual feature channels or full multi-channel tensors.
    
    Args:
        model: Trained DiffusionDenoiserModel (or None for 'off' mode)
        device: Torch device
        config: Configuration dict with keys:
            - enabled: bool
            - mode: 'off' | 'inference'
            - inference_steps: int (default 20)
            - noise_level: float (default 0.3)
            - clip_output: bool (default True)
    """
    
    def __init__(
        self,
        model=None,
        device: str = 'cpu',
        config: Optional[Dict] = None,
    ):
        self.model = model
        self.device = device
        self.config = config or {}
        
        # Extract config with defaults
        self.enabled = self.config.get('enabled', False)
        self.mode = self.config.get('mode', 'off')
        self.inference_steps = self.config.get('inference_steps', 20)
        self.noise_level = self.config.get('noise_level', 0.3)
        self.clip_output = self.config.get('clip_output', True)
        
        # Normalization stats (fit during first use if not provided)
        self.mean_ = self.config.get('mean', None)
        self.std_ = self.config.get('std', None)
        
        if self.model is not None:
            self.model.to(device)
            self.model.eval()
    
    def fit_normalization(self, features: np.ndarray):
        """
        Fit normalization statistics from training data.
        
        Args:
            features: [N, C, L] or [N, L] feature array
        """
        if features.ndim == 2:
            features = features[:, np.newaxis, :]
        
        self.mean_ = features.mean(axis=(0, 2), keepdims=True)
        self.std_ = features.std(axis=(0, 2), keepdims=True) + 1e-8
        
        logger.info(f"[DiffusionDenoiser] Fit normalization: mean={self.mean_.mean():.4f}, std={self.std_.mean():.4f}")
    
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize to [-1, 1] range for diffusion."""
        if self.mean_ is not None and self.std_ is not None:
            mean = torch.tensor(self.mean_, device=x.device, dtype=x.dtype)
            std = torch.tensor(self.std_, device=x.device, dtype=x.dtype)
            return (x - mean) / std
        return x
    
    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Reverse normalization."""
        if self.mean_ is not None and self.std_ is not None:
            mean = torch.tensor(self.mean_, device=x.device, dtype=x.dtype)
            std = torch.tensor(self.std_, device=x.device, dtype=x.dtype)
            return x * std + mean
        return x
    
    def denoise_batch(
        self,
        features: Union[np.ndarray, torch.Tensor],
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Denoise a batch of feature sequences.
        
        Args:
            features: [B, C, L] or [B, L] feature tensor/array
            
        Returns:
            denoised: Same shape as input, denoised features
        """
        # Return unchanged if disabled or off mode
        if not self.enabled or self.mode == 'off' or self.model is None:
            return features
        
        # Track input type
        is_numpy = isinstance(features, np.ndarray)
        
        # Convert to tensor
        if is_numpy:
            features_t = torch.tensor(features, dtype=torch.float32, device=self.device)
        else:
            features_t = features.to(self.device)
        
        # Add channel dimension if needed
        squeeze_channel = False
        if features_t.ndim == 2:
            features_t = features_t.unsqueeze(1)
            squeeze_channel = True
        
        # Normalize
        features_norm = self._normalize(features_t)
        
        # Denoise
        with torch.no_grad():
            denoised_norm = self.model.denoise(
                features_norm,
                num_inference_steps=self.inference_steps,
                noise_level=self.noise_level,
                clip_output=self.clip_output,
            )
        
        # Denormalize
        denoised = self._denormalize(denoised_norm)
        
        # Remove channel dimension if added
        if squeeze_channel:
            denoised = denoised.squeeze(1)
        
        # Convert back to numpy if input was numpy
        if is_numpy:
            denoised = denoised.cpu().numpy()
        
        return denoised
    
    def denoise_dataframe_column(
        self,
        df,
        column: str,
        seq_len: int = 128,
        output_column: Optional[str] = None,
    ):
        """
        Denoise a specific column in a DataFrame using sliding windows.
        
        Args:
            df: pandas DataFrame
            column: Column name to denoise
            seq_len: Window length for diffusion
            output_column: Output column name (default: column + '_denoised')
            
        Returns:
            df: DataFrame with new denoised column
        """
        import pandas as pd
        
        if not self.enabled or self.mode == 'off' or self.model is None:
            if output_column:
                df[output_column] = df[column]
            return df
        
        output_column = output_column or f"{column}_denoised"
        values = df[column].values
        
        # Pad if needed
        n = len(values)
        if n < seq_len:
            # Too short, just copy
            df[output_column] = values
            return df
        
        # Process in overlapping windows
        denoised_values = np.zeros_like(values)
        counts = np.zeros(n)
        
        step = seq_len // 2  # 50% overlap
        
        for start in range(0, n - seq_len + 1, step):
            end = start + seq_len
            window = values[start:end]
            
            # Denoise window
            window_t = torch.tensor(window, dtype=torch.float32, device=self.device)
            window_t = window_t.unsqueeze(0).unsqueeze(0)  # [1, 1, L]
            
            window_norm = self._normalize(window_t)
            
            with torch.no_grad():
                denoised_norm = self.model.denoise(
                    window_norm,
                    num_inference_steps=self.inference_steps,
                    noise_level=self.noise_level,
                )
            
            denoised_window = self._denormalize(denoised_norm)
            denoised_window = denoised_window.squeeze().cpu().numpy()
            
            # Accumulate with overlap weighting
            denoised_values[start:end] += denoised_window
            counts[start:end] += 1
        
        # Handle edge that might not be covered
        if counts[-1] == 0:
            # Process final window
            window = values[-seq_len:]
            window_t = torch.tensor(window, dtype=torch.float32, device=self.device)
            window_t = window_t.unsqueeze(0).unsqueeze(0)
            window_norm = self._normalize(window_t)
            
            with torch.no_grad():
                denoised_norm = self.model.denoise(
                    window_norm,
                    num_inference_steps=self.inference_steps,
                    noise_level=self.noise_level,
                )
            
            denoised_window = self._denormalize(denoised_norm)
            denoised_window = denoised_window.squeeze().cpu().numpy()
            
            start = n - seq_len
            denoised_values[start:] += denoised_window
            counts[start:] += 1
        
        # Average overlapping regions
        counts = np.maximum(counts, 1)
        denoised_values = denoised_values / counts
        
        df[output_column] = denoised_values
        
        return df
    
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        config: Optional[Dict] = None,
        device: str = 'cpu',
    ) -> 'DiffusionFeatureDenoiser':
        """
        Load denoiser from checkpoint.
        
        Args:
            checkpoint_path: Path to model checkpoint
            config: Configuration dict
            device: Torch device
            
        Returns:
            DiffusionFeatureDenoiser instance
        """
        from .diffusion.time_series_diffusion import DiffusionDenoiserModel
        
        path = Path(checkpoint_path)
        if not path.exists():
            logger.warning(f"[DiffusionDenoiser] Checkpoint not found: {path}")
            return cls(model=None, device=device, config=config)
        
        model = DiffusionDenoiserModel.load(str(path), device=device)
        
        config = config or {}
        config['enabled'] = True
        config['mode'] = 'inference'
        
        return cls(model=model, device=device, config=config)
    
    def save_config(self, path: str):
        """Save denoiser configuration."""
        import json
        
        config = {
            'enabled': self.enabled,
            'mode': self.mode,
            'inference_steps': self.inference_steps,
            'noise_level': self.noise_level,
            'clip_output': self.clip_output,
        }
        
        if self.mean_ is not None:
            config['mean'] = self.mean_.tolist() if hasattr(self.mean_, 'tolist') else self.mean_
        if self.std_ is not None:
            config['std'] = self.std_.tolist() if hasattr(self.std_, 'tolist') else self.std_
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)


def create_denoiser_from_config(config: Dict, device: str = 'cpu') -> DiffusionFeatureDenoiser:
    """
    Factory function to create denoiser from global config.
    
    Args:
        config: DIFFUSION_DENOISER config block
        device: Torch device
        
    Returns:
        DiffusionFeatureDenoiser instance
    """
    if not config.get('enabled', False):
        return DiffusionFeatureDenoiser(model=None, device=device, config={'enabled': False})
    
    checkpoint_path = config.get('checkpoint_path', 'artifacts/diffusion_denoiser/latest.ckpt')
    
    return DiffusionFeatureDenoiser.from_checkpoint(
        checkpoint_path=checkpoint_path,
        config=config,
        device=device,
    )
