"""
Training Script for Diffusion Denoiser.

Trains a 1D diffusion model to denoise time series features (frac_diff, log_returns, etc.)
for improved downstream model performance.

Usage:
    python -m src.training.train_diffusion_denoiser --symbol BTCUSDT --epochs 100

Author: QFC System - Diffusion Architecture
"""

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Local imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src import config as cfg
from src.data_loader import MarketDataLoader
from src.features import SignalFactory
from src.models.diffusion.time_series_diffusion import DiffusionDenoiserModel
from src.preprocessing.frac_diff import FractionalDifferentiator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TimeSeriesWindowDataset(Dataset):
    """
    Dataset of sliding windows from time series.
    
    Args:
        data: [N] or [N, C] time series data
        seq_len: Window length
        stride: Stride between windows (default: seq_len // 2 for 50% overlap)
    """
    
    def __init__(
        self,
        data: np.ndarray,
        seq_len: int = 128,
        stride: Optional[int] = None,
    ):
        self.data = data
        self.seq_len = seq_len
        self.stride = stride or seq_len // 2
        
        # Ensure 2D: [N, C]
        if self.data.ndim == 1:
            self.data = self.data[:, np.newaxis]
        
        # Compute windows
        n_samples = self.data.shape[0]
        self.n_windows = max(1, (n_samples - seq_len) // self.stride + 1)
    
    def __len__(self):
        return self.n_windows
    
    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.seq_len
        
        window = self.data[start:end]  # [L, C]
        
        # Transpose to [C, L] for conv layers
        window = window.T
        
        return torch.tensor(window, dtype=torch.float32)


class DiffusionTrainer:
    """
    Trainer for diffusion denoiser model.
    
    Handles:
        - Data loading and preprocessing
        - Training loop with loss logging
        - Validation with reconstruction metrics
        - Checkpointing
    """
    
    def __init__(
        self,
        config: Dict,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        self.config = config
        self.device = device
        
        # Extract config
        self.seq_len = config.get('seq_len', 128)
        self.num_timesteps = config.get('num_timesteps', 1000)
        self.model_channels = config.get('model_channels', 64)
        self.beta_schedule = config.get('beta_schedule', 'cosine')
        self.dropout = config.get('dropout', 0.1)
        self.artifacts_dir = Path(config.get('artifacts_dir', 'artifacts/diffusion_denoiser'))
        
        # Training config
        self.epochs = config.get('epochs', 100)
        self.batch_size = config.get('batch_size', 32)
        self.lr = config.get('lr', 1e-4)
        self.weight_decay = config.get('weight_decay', 1e-4)
        
        # Create directories
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        (self.artifacts_dir / 'checkpoints').mkdir(exist_ok=True)
        
        # Initialize model
        self.model = None
        self.optimizer = None
        self.scheduler = None
        
        # Metrics history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_mse': [],
        }
        
        # Normalization stats
        self.mean_ = None
        self.std_ = None
    
    def prepare_data(
        self,
        symbol: str,
        target: str = 'frac_diff',
        train_ratio: float = 0.8,
        history_days: int = 60,
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Load and prepare data for training.
        
        Args:
            symbol: Trading pair symbol
            target: Target feature ('frac_diff', 'log_returns')
            train_ratio: Train/val split ratio
            history_days: Days of history to load
            
        Returns:
            train_loader, val_loader
        """
        logger.info(f"[Data] Loading {symbol} @ 60m for {history_days} days...")
        
        # Load data
        loader = MarketDataLoader()
        df = loader.fetch_ohlcv(symbol, interval='60', days_back=history_days)
        
        if df is None or len(df) < self.seq_len * 2:
            raise ValueError(f"Insufficient data for {symbol}")
        
        logger.info(f"[Data] Loaded {len(df)} candles")
        
        # Compute target feature
        if target == 'frac_diff':
            fracdiff = FractionalDifferentiator(max_d=0.7)
            fracdiff.fit(df['close'])
            values = fracdiff.transform(df['close'])
            # Handle NaN
            values = np.nan_to_num(values, nan=0.0)
        elif target == 'log_returns':
            values = np.log(df['close'] / df['close'].shift(1)).fillna(0).values
        else:
            raise ValueError(f"Unknown target: {target}")
        
        # Remove warm-up period
        warmup = min(200, len(values) // 10)
        values = values[warmup:]
        
        logger.info(f"[Data] Target '{target}' shape: {values.shape}")
        
        # Compute normalization
        self.mean_ = values.mean()
        self.std_ = values.std() + 1e-8
        values_norm = (values - self.mean_) / self.std_
        
        logger.info(f"[Data] Normalized: mean={self.mean_:.6f}, std={self.std_:.6f}")
        
        # Split
        split_idx = int(len(values_norm) * train_ratio)
        train_data = values_norm[:split_idx]
        val_data = values_norm[split_idx:]
        
        # Create datasets
        train_dataset = TimeSeriesWindowDataset(train_data, self.seq_len)
        val_dataset = TimeSeriesWindowDataset(val_data, self.seq_len, stride=self.seq_len)
        
        logger.info(f"[Data] Train windows: {len(train_dataset)}, Val windows: {len(val_dataset)}")
        
        # Create loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )
        
        return train_loader, val_loader
    
    def build_model(self, in_channels: int = 1):
        """Initialize model, optimizer, and scheduler."""
        self.model = DiffusionDenoiserModel(
            in_channels=in_channels,
            model_channels=self.model_channels,
            num_timesteps=self.num_timesteps,
            beta_schedule=self.beta_schedule,
            dropout=self.dropout,
        ).to(self.device)
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.epochs,
            eta_min=self.lr * 0.01,
        )
        
        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"[Model] Parameters: {n_params:,}")
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        for batch in train_loader:
            batch = batch.to(self.device)
            
            self.optimizer.zero_grad()
            
            losses = self.model.compute_loss(batch)
            loss = losses['loss']
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / max(n_batches, 1)
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()
        total_loss = 0.0
        total_mse = 0.0
        n_batches = 0
        
        for batch in val_loader:
            batch = batch.to(self.device)
            
            # Noise prediction loss
            losses = self.model.compute_loss(batch)
            total_loss += losses['loss'].item()
            
            # Reconstruction MSE (denoise and compare)
            denoised = self.model.denoise(batch, num_inference_steps=20, noise_level=0.3)
            mse = ((batch - denoised) ** 2).mean().item()
            total_mse += mse
            
            n_batches += 1
        
        return {
            'val_loss': total_loss / max(n_batches, 1),
            'val_mse': total_mse / max(n_batches, 1),
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        log_every: int = 10,
    ):
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            log_every: Log every N epochs
        """
        best_val_loss = float('inf')
        start_time = time.time()
        
        logger.info(f"[Train] Starting {self.epochs} epochs on {self.device}")
        
        for epoch in range(1, self.epochs + 1):
            # Train
            train_loss = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            
            # Validate
            val_metrics = self.validate(val_loader)
            self.history['val_loss'].append(val_metrics['val_loss'])
            self.history['val_mse'].append(val_metrics['val_mse'])
            
            # LR schedule
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            
            # Log
            if epoch % log_every == 0 or epoch == 1:
                elapsed = time.time() - start_time
                logger.info(
                    f"[Epoch {epoch:3d}/{self.epochs}] "
                    f"Loss: {train_loss:.4f} | Val: {val_metrics['val_loss']:.4f} | "
                    f"MSE: {val_metrics['val_mse']:.6f} | LR: {current_lr:.2e} | "
                    f"Time: {elapsed:.1f}s"
                )
            
            # Save best
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                self.save_checkpoint('best.pt')
            
            # Periodic checkpoint
            if epoch % 20 == 0:
                self.save_checkpoint(f'epoch_{epoch}.pt')
        
        # Save final
        self.save_checkpoint('latest.ckpt')
        self.save_metrics()
        
        total_time = time.time() - start_time
        logger.info(f"[Train] Complete in {total_time:.1f}s. Best val loss: {best_val_loss:.4f}")
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = self.artifacts_dir / 'checkpoints' / filename
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'mean': self.mean_,
            'std': self.std_,
            'history': self.history,
        }, path)
        
        # Also save to latest.ckpt at root
        if filename == 'latest.ckpt':
            latest_path = self.artifacts_dir / 'latest.ckpt'
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'config': {
                    'in_channels': 1,
                    'out_channels': 1,
                    'num_timesteps': self.num_timesteps,
                    'model_channels': self.model_channels,
                    'beta_schedule': self.beta_schedule,
                },
                'mean': self.mean_,
                'std': self.std_,
            }, latest_path)
    
    def save_metrics(self):
        """Save training metrics."""
        metrics_path = self.artifacts_dir / 'metrics.json'
        
        with open(metrics_path, 'w') as f:
            json.dump({
                'history': self.history,
                'config': {k: str(v) if isinstance(v, Path) else v for k, v in self.config.items()},
                'timestamp': datetime.now().isoformat(),
            }, f, indent=2)
        
        logger.info(f"[Metrics] Saved to {metrics_path}")


def main():
    parser = argparse.ArgumentParser(description='Train Diffusion Denoiser')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Trading pair')
    parser.add_argument('--target', type=str, default='frac_diff', choices=['frac_diff', 'log_returns'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seq-len', type=int, default=128)
    parser.add_argument('--history-days', type=int, default=60)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 72)
    print("DIFFUSION DENOISER TRAINING")
    print("=" * 72)
    print(f"Symbol:       {args.symbol}")
    print(f"Target:       {args.target}")
    print(f"Epochs:       {args.epochs}")
    print(f"Batch Size:   {args.batch_size}")
    print(f"Seq Length:   {args.seq_len}")
    print(f"History:      {args.history_days} days")
    print(f"Device:       {args.device}")
    print("=" * 72 + "\n")
    
    # Build config
    config = {
        **cfg.DIFFUSION_DENOISER,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'seq_len': args.seq_len,
    }
    
    # Create trainer
    trainer = DiffusionTrainer(config, device=args.device)
    
    # Prepare data
    train_loader, val_loader = trainer.prepare_data(
        symbol=args.symbol,
        target=args.target,
        history_days=args.history_days,
    )
    
    # Build model
    trainer.build_model(in_channels=1)
    
    # Train
    trainer.train(train_loader, val_loader)
    
    print("\n[SUCCESS] Training complete!")
    print(f"  Checkpoint: artifacts/diffusion_denoiser/latest.ckpt")
    print(f"  Metrics: artifacts/diffusion_denoiser/metrics.json")


if __name__ == '__main__':
    main()
