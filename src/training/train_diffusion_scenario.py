"""
Training Script for Diffusion Scenario Model.

Trains a conditional diffusion model to generate future return scenarios
given past feature windows. Used by DiffusionExpert for trading signals.

Usage:
    python -m src.training.train_diffusion_scenario --symbol BTCUSDT --epochs 100

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
from torch.utils.data import DataLoader

# Local imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src import config as cfg
from src.data_loader import MarketDataLoader
from src.models.diffusion.diffusion_scenario import DiffusionScenarioModel
from src.datasets.scenario_dataset import ScenarioDataset
from src.preprocessing.frac_diff import FractionalDifferentiator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ScenarioTrainer:
    """
    Trainer for diffusion scenario model.
    
    Handles:
        - Data loading and feature computation
        - Training loop with conditional diffusion loss
        - Scenario quality evaluation
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
        self.L_past = config.get('L_past', 96)
        self.H_future = config.get('H_future', 12)
        self.num_timesteps = config.get('num_timesteps', 1000)
        self.model_channels = config.get('model_channels', 64)
        self.cond_dim = config.get('cond_dim', 128)
        self.beta_schedule = config.get('beta_schedule', 'cosine')
        
        self.artifacts_dir = Path(config.get('artifacts_dir', 'artifacts/diffusion_scenario'))
        
        # Training config
        self.epochs = config.get('epochs', 100)
        self.batch_size = config.get('batch_size', 32)
        self.lr = config.get('lr', 1e-4)
        self.weight_decay = config.get('weight_decay', 1e-4)
        
        # Create directories
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        (self.artifacts_dir / 'checkpoints').mkdir(exist_ok=True)
        
        # Model and optimizer
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.in_channels_past = None
        
        # Metrics
        self.history = {
            'train_loss': [],
            'val_loss': [],
        }
        
        # Normalization
        self.norm_stats = None
    
    def prepare_data(
        self,
        symbol: str,
        train_ratio: float = 0.8,
        history_days: int = 60,
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Load and prepare scenario training data.
        """
        logger.info(f"[Data] Loading {symbol} @ 60m for {history_days} days...")
        
        loader = MarketDataLoader()
        df = loader.fetch_ohlcv(symbol, interval='60', days_back=history_days)
        
        if df is None or len(df) < self.L_past + self.H_future + 100:
            raise ValueError(f"Insufficient data for {symbol}")
        
        logger.info(f"[Data] Loaded {len(df)} candles")
        
        # Compute log returns
        log_returns = np.log(df['close'] / df['close'].shift(1)).fillna(0).values
        
        # Build feature set for past window
        features = []
        feature_names = []
        
        # 1. Log returns
        features.append(log_returns)
        feature_names.append('log_return')
        
        # 2. Frac diff
        fracdiff = FractionalDifferentiator(max_d=0.7)
        fracdiff.fit(df['close'])
        frac_diff = fracdiff.transform(df['close'])
        frac_diff = np.nan_to_num(frac_diff, nan=0.0)
        features.append(frac_diff)
        feature_names.append('frac_diff')
        
        # 3. Volatility features
        import pandas as pd
        for window in [5, 10, 20]:
            vol = pd.Series(log_returns).rolling(window).std().fillna(0).values
            features.append(vol)
            feature_names.append(f'vol_{window}')
        
        # 4. Momentum features
        for window in [5, 10, 20]:
            mom = pd.Series(log_returns).rolling(window).sum().fillna(0).values
            features.append(mom)
            feature_names.append(f'mom_{window}')
        
        # 5. Abs returns
        abs_ret = np.abs(log_returns)
        features.append(abs_ret)
        feature_names.append('abs_return')
        
        # 6. Rolling mean
        mean_20 = pd.Series(log_returns).rolling(20).mean().fillna(0).values
        features.append(mean_20)
        feature_names.append('mean_20')
        
        # Stack features
        past_features = np.column_stack(features)
        future_target = log_returns[:, np.newaxis]
        
        self.in_channels_past = past_features.shape[1]
        logger.info(f"[Data] Past features: {self.in_channels_past} channels ({feature_names})")
        
        # Remove warmup
        warmup = max(50, self.L_past)
        past_features = past_features[warmup:]
        future_target = future_target[warmup:]
        
        # Split
        n = len(past_features)
        split_idx = int(n * train_ratio)
        
        train_dataset = ScenarioDataset(
            past_features[:split_idx],
            future_target[:split_idx],
            L_past=self.L_past,
            H_future=self.H_future,
            stride=1,
            normalize=True,
        )
        
        val_dataset = ScenarioDataset(
            past_features[split_idx:],
            future_target[split_idx:],
            L_past=self.L_past,
            H_future=self.H_future,
            stride=self.H_future,  # Non-overlapping for val
            normalize=True,
        )
        
        # Store normalization stats from training set
        self.norm_stats = train_dataset.get_normalization_stats()
        
        logger.info(f"[Data] Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
        
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
    
    def build_model(self):
        """Initialize model, optimizer, scheduler."""
        self.model = DiffusionScenarioModel(
            in_channels_past=self.in_channels_past,
            in_channels_future=1,
            L_past=self.L_past,
            H_future=self.H_future,
            model_channels=self.model_channels,
            cond_dim=self.cond_dim,
            num_timesteps=self.num_timesteps,
            beta_schedule=self.beta_schedule,
        ).to(self.device)
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
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
        
        for past, future in train_loader:
            past = past.to(self.device)
            future = future.to(self.device)
            
            self.optimizer.zero_grad()
            
            losses = self.model.compute_loss(future, past)
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
        n_batches = 0
        
        for past, future in val_loader:
            past = past.to(self.device)
            future = future.to(self.device)
            
            losses = self.model.compute_loss(future, past)
            total_loss += losses['loss'].item()
            n_batches += 1
        
        return {'val_loss': total_loss / max(n_batches, 1)}
    
    @torch.no_grad()
    def evaluate_scenarios(self, val_loader: DataLoader, num_samples: int = 32) -> Dict[str, float]:
        """
        Evaluate scenario quality on validation set.
        
        Computes distributional metrics comparing generated vs real futures.
        """
        self.model.eval()
        
        all_real = []
        all_generated = []
        
        for past, future in val_loader:
            past = past.to(self.device)
            future = future.to(self.device)
            
            # Generate scenarios
            scenarios = self.model.generate_scenarios(
                past, num_samples=num_samples, num_inference_steps=20
            )
            
            # Get mean scenario
            mean_scenario = scenarios.view(past.shape[0], num_samples, -1, self.H_future).mean(dim=1)
            
            all_real.append(future.cpu())
            all_generated.append(mean_scenario.cpu())
        
        real = torch.cat(all_real, dim=0)
        gen = torch.cat(all_generated, dim=0)
        
        # Compute metrics
        mse = ((real - gen) ** 2).mean().item()
        mae = (real - gen).abs().mean().item()
        
        # Correlation
        real_flat = real.view(-1).numpy()
        gen_flat = gen.view(-1).numpy()
        corr = np.corrcoef(real_flat, gen_flat)[0, 1]
        
        return {
            'scenario_mse': mse,
            'scenario_mae': mae,
            'scenario_corr': corr,
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        log_every: int = 10,
    ):
        """Full training loop."""
        best_val_loss = float('inf')
        start_time = time.time()
        
        logger.info(f"[Train] Starting {self.epochs} epochs on {self.device}")
        
        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            
            val_metrics = self.validate(val_loader)
            self.history['val_loss'].append(val_metrics['val_loss'])
            
            self.lr_scheduler.step()
            current_lr = self.lr_scheduler.get_last_lr()[0]
            
            if epoch % log_every == 0 or epoch == 1:
                elapsed = time.time() - start_time
                logger.info(
                    f"[Epoch {epoch:3d}/{self.epochs}] "
                    f"Loss: {train_loss:.4f} | Val: {val_metrics['val_loss']:.4f} | "
                    f"LR: {current_lr:.2e} | Time: {elapsed:.1f}s"
                )
            
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                self.save_checkpoint('best.pt')
            
            if epoch % 20 == 0:
                self.save_checkpoint(f'epoch_{epoch}.pt')
        
        # Final evaluation
        logger.info("[Train] Evaluating scenario quality...")
        scenario_metrics = self.evaluate_scenarios(val_loader)
        logger.info(f"  Scenario MSE: {scenario_metrics['scenario_mse']:.6f}")
        logger.info(f"  Scenario Corr: {scenario_metrics['scenario_corr']:.4f}")
        
        self.save_checkpoint('latest.ckpt')
        self.save_metrics()
        
        total_time = time.time() - start_time
        logger.info(f"[Train] Complete in {total_time:.1f}s. Best val loss: {best_val_loss:.4f}")
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = self.artifacts_dir / 'checkpoints' / filename
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': {
                'in_channels_past': self.in_channels_past,
                'in_channels_future': 1,
                'L_past': self.L_past,
                'H_future': self.H_future,
                'model_channels': self.model_channels,
                'cond_dim': self.cond_dim,
                'num_timesteps': self.num_timesteps,
            },
            'history': self.history,
        }
        
        if self.norm_stats:
            checkpoint['past_mean'] = self.norm_stats['past_mean'].tolist()
            checkpoint['past_std'] = self.norm_stats['past_std'].tolist()
            checkpoint['future_mean'] = self.norm_stats['future_mean'].tolist()
            checkpoint['future_std'] = self.norm_stats['future_std'].tolist()
        
        torch.save(checkpoint, path)
        
        if filename == 'latest.ckpt':
            torch.save(checkpoint, self.artifacts_dir / 'latest.ckpt')
    
    def save_metrics(self):
        """Save training metrics."""
        metrics_path = self.artifacts_dir / 'metrics.json'
        
        with open(metrics_path, 'w') as f:
            json.dump({
                'history': self.history,
                'config': {k: str(v) if isinstance(v, Path) else v for k, v in self.config.items()},
                'timestamp': datetime.now().isoformat(),
            }, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Train Diffusion Scenario Model')
    parser.add_argument('--symbol', type=str, default='BTCUSDT')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--L-past', type=int, default=96)
    parser.add_argument('--H-future', type=int, default=12)
    parser.add_argument('--history-days', type=int, default=60)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 72)
    print("DIFFUSION SCENARIO MODEL TRAINING")
    print("=" * 72)
    print(f"Symbol:       {args.symbol}")
    print(f"Epochs:       {args.epochs}")
    print(f"L_past:       {args.L_past}")
    print(f"H_future:     {args.H_future}")
    print(f"Device:       {args.device}")
    print("=" * 72 + "\n")
    
    config = {
        **cfg.DIFFUSION_SCENARIO,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'L_past': args.L_past,
        'H_future': args.H_future,
    }
    
    trainer = ScenarioTrainer(config, device=args.device)
    train_loader, val_loader = trainer.prepare_data(
        symbol=args.symbol,
        history_days=args.history_days,
    )
    
    trainer.build_model()
    trainer.train(train_loader, val_loader)
    
    print("\n[SUCCESS] Training complete!")
    print(f"  Checkpoint: artifacts/diffusion_scenario/latest.ckpt")


if __name__ == '__main__':
    main()
