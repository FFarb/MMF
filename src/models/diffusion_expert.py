"""
Diffusion Expert - 6th Expert in Mixture of Experts.

Uses DiffusionScenarioModel to generate K future scenarios and compute:
- P_up_diff: Probability of positive future return
- E_ret_diff: Expected future return
- tail_risk: 5th percentile return (downside risk)

Compatible with MoE interface (fit/predict_proba).

Author: QFC System - Diffusion Architecture
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.models.diffusion.scenario_generator import ScenarioGenerator

logger = logging.getLogger(__name__)


class CalibrationHead(nn.Module):
    """
    Small MLP to calibrate scenario-based probabilities.
    
    Takes raw scenario statistics and produces calibrated P(up).
    """
    
    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, input_dim] scenario statistics
            
        Returns:
            p_up: [B, 1] calibrated probability
        """
        return self.net(x)


class DiffusionExpert:
    """
    Diffusion-based expert for Mixture of Experts ensemble.
    
    Uses ScenarioGenerator to compute probabilistic trading signals
    from diffusion-generated future scenarios.
    
    Args:
        scenario_generator: ScenarioGenerator instance
        config: Expert configuration dict
        device: Torch device
    """
    
    def __init__(
        self,
        scenario_generator: Optional[ScenarioGenerator] = None,
        config: Optional[Dict] = None,
        device: str = 'cpu',
    ):
        self.scenario_generator = scenario_generator
        self.config = config or {}
        self.device = device
        
        # Config with defaults
        self.num_scenarios = self.config.get('num_scenarios', 32)
        self.horizon = self.config.get('horizon', 12)
        self.use_calibration_head = self.config.get('use_calibration_head', True)
        self.calibration_hidden_dim = self.config.get('calibration_hidden_dim', 32)
        self.tail_risk_quantile = self.config.get('tail_risk_quantile', 0.05)
        
        # Feature construction for past window
        self.feature_columns: List[str] = []
        self.L_past = self.config.get('L_past', 96)
        
        # Calibration head
        self.calibration_head = None
        if self.use_calibration_head:
            self.calibration_head = CalibrationHead(
                input_dim=5,  # P_up_raw, E_ret, volatility, tail_risk, max_dd
                hidden_dim=self.calibration_hidden_dim,
            ).to(device)
        
        # Normalization stats (set during fit)
        self.feature_mean_ = None
        self.feature_std_ = None
        
        # Flag for enabled state
        self.enabled = self.config.get('enabled', True)
        self._fitted = False
    
    def _compute_past_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Compute features for past window from DataFrame.
        
        Returns: [N, C_past] feature array
        """
        features = []
        
        # Log returns
        if 'log_return' in df.columns:
            features.append(df['log_return'].values)
        elif 'close' in df.columns:
            log_ret = np.log(df['close'] / df['close'].shift(1)).fillna(0).values
            features.append(log_ret)
        
        # Frac diff
        if 'frac_diff' in df.columns:
            features.append(df['frac_diff'].fillna(0).values)
        
        # Volatility
        for col in ['vol_5', 'vol_10', 'vol_20', 'volatility_14']:
            if col in df.columns:
                features.append(df[col].fillna(0).values)
        
        # Momentum
        for col in ['mom_5', 'mom_10', 'mom_20', 'returns_5', 'returns_10']:
            if col in df.columns:
                features.append(df[col].fillna(0).values)
        
        # If not enough features, add synthetic ones
        while len(features) < 4:
            if len(features) > 0:
                features.append(features[0])  # Duplicate first feature
            else:
                features.append(np.zeros(len(df)))
        
        return np.column_stack(features)
    
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ):
        """
        Fit the diffusion expert.
        
        For now, just stores normalization stats and optionally trains calibration head.
        The scenario model itself should be pre-trained separately.
        
        Args:
            X: Feature matrix or DataFrame
            y: Target labels (0/1 for down/up)
            sample_weight: Sample weights (unused)
        """
        if not self.enabled:
            self._fitted = True
            return
        
        logger.info("[DiffusionExpert] Fitting calibration...")
        
        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        
        # Compute features
        past_features = self._compute_past_features(X)
        
        # Store normalization
        self.feature_mean_ = past_features.mean(axis=0, keepdims=True)
        self.feature_std_ = past_features.std(axis=0, keepdims=True) + 1e-8
        
        # If we have a scenario generator and calibration head, train calibration
        if self.scenario_generator is not None and self.calibration_head is not None:
            self._train_calibration(past_features, y)
        
        self._fitted = True
        logger.info("[DiffusionExpert] [OK] Fit complete")
    
    def _train_calibration(
        self,
        past_features: np.ndarray,
        y: np.ndarray,
        epochs: int = 50,
        batch_size: int = 64,
    ):
        """Train calibration head on historical data."""
        logger.info("  Training calibration head...")
        
        n_samples = len(y)
        if n_samples < self.L_past + 100:
            logger.warning("  Not enough samples for calibration training")
            return
        
        # Compute scenario statistics for samples
        scenario_stats = []
        labels = []
        
        # Sample subset for efficiency
        sample_indices = np.random.choice(
            range(self.L_past, n_samples - self.horizon),
            size=min(1000, n_samples - self.L_past - self.horizon),
            replace=False,
        )
        
        for idx in sample_indices:
            # Get past window
            window = past_features[idx - self.L_past:idx]
            window_norm = (window - self.feature_mean_) / self.feature_std_
            
            # Generate scenarios
            window_t = torch.tensor(window_norm.T, dtype=torch.float32, device=self.device)
            scenarios_t, stats = self.scenario_generator.generate_and_analyze(
                window_t.unsqueeze(0),
                num_samples=self.num_scenarios,
            )
            
            # Extract statistics
            stats_vec = [
                stats['P_up'].item(),
                stats['E_return'].item(),
                stats['volatility'].item(),
                stats['tail_risk'].item(),
                stats['max_drawdown'].item(),
            ]
            scenario_stats.append(stats_vec)
            
            # Get true label (future return direction)
            future_returns = y[idx:idx + self.horizon]
            if len(future_returns) >= self.horizon:
                true_up = 1.0 if future_returns.sum() > 0 else 0.0
            else:
                true_up = y[idx]
            labels.append(true_up)
        
        if len(scenario_stats) == 0:
            return
        
        # Train calibration head
        X_train = torch.tensor(scenario_stats, dtype=torch.float32, device=self.device)
        y_train = torch.tensor(labels, dtype=torch.float32, device=self.device).unsqueeze(1)
        
        optimizer = torch.optim.Adam(self.calibration_head.parameters(), lr=1e-3)
        criterion = nn.BCELoss()
        
        self.calibration_head.train()
        for epoch in range(epochs):
            # Mini-batch training
            perm = torch.randperm(len(X_train))
            total_loss = 0.0
            n_batches = 0
            
            for i in range(0, len(X_train), batch_size):
                batch_idx = perm[i:i + batch_size]
                x_batch = X_train[batch_idx]
                y_batch = y_train[batch_idx]
                
                optimizer.zero_grad()
                pred = self.calibration_head(x_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
            
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / n_batches
                logger.info(f"    Calibration epoch {epoch + 1}: loss = {avg_loss:.4f}")
        
        self.calibration_head.eval()
    
    def _compute_scenario_stats(
        self,
        past_window: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate scenarios and compute statistics.
        
        Args:
            past_window: [B, C, L] past feature window
            
        Returns:
            Dict with P_up, E_return, volatility, tail_risk, max_drawdown
        """
        if self.scenario_generator is None:
            batch_size = past_window.shape[0]
            return {
                'P_up': torch.full((batch_size,), 0.5),
                'E_return': torch.zeros(batch_size),
                'volatility': torch.ones(batch_size) * 0.01,
                'tail_risk': torch.full((batch_size,), -0.02),
                'max_drawdown': torch.full((batch_size,), 0.01),
            }
        
        scenarios, stats = self.scenario_generator.generate_and_analyze(
            past_window,
            num_samples=self.num_scenarios,
        )
        
        return stats
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Feature matrix or DataFrame
            
        Returns:
            proba: [N, 2] array with [P(down), P(up)]
        """
        if not self.enabled or not self._fitted:
            # Return neutral predictions
            n = len(X) if hasattr(X, '__len__') else 1
            return np.full((n, 2), 0.5)
        
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        
        n_samples = len(X)
        proba = np.zeros((n_samples, 2))
        
        # Process in batches
        batch_size = 32
        
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_df = X.iloc[start:end]
            
            # For each sample, we need L_past previous rows
            # This is tricky with DataFrames - we'll use simple approach
            p_up_batch = []
            
            for i in range(len(batch_df)):
                global_idx = start + i
                
                if global_idx < self.L_past:
                    # Not enough history
                    p_up_batch.append(0.5)
                    continue
                
                # Get past window from original X
                past_df = X.iloc[global_idx - self.L_past:global_idx]
                past_features = self._compute_past_features(past_df)
                
                # Normalize
                if self.feature_mean_ is not None:
                    past_features = (past_features - self.feature_mean_) / self.feature_std_
                
                # Convert to tensor [1, C, L]
                past_t = torch.tensor(
                    past_features.T,
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(0)
                
                # Get scenario statistics
                with torch.no_grad():
                    stats = self._compute_scenario_stats(past_t)
                
                # Apply calibration if available
                if self.calibration_head is not None:
                    stats_vec = torch.tensor([
                        [stats['P_up'].item(),
                         stats['E_return'].item(),
                         stats['volatility'].item(),
                         stats['tail_risk'].item(),
                         stats['max_drawdown'].item()]
                    ], dtype=torch.float32, device=self.device)
                    
                    p_up = self.calibration_head(stats_vec).item()
                else:
                    p_up = stats['P_up'].item()
                
                p_up_batch.append(p_up)
            
            # Fill in probabilities
            for i, p_up in enumerate(p_up_batch):
                proba[start + i, 0] = 1.0 - p_up
                proba[start + i, 1] = p_up
        
        return proba
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict class labels."""
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)
    
    def predict_risk(self, X: Union[pd.DataFrame, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Predict with extended risk metrics.
        
        Returns:
            Dict with P_up, expected_return, tail_risk, volatility
        """
        if not self.enabled or not self._fitted:
            n = len(X) if hasattr(X, '__len__') else 1
            return {
                'P_up': np.full(n, 0.5),
                'expected_return': np.zeros(n),
                'tail_risk': np.full(n, -0.02),
                'volatility': np.full(n, 0.01),
            }
        
        # Simplified version - uses predict_proba
        proba = self.predict_proba(X)
        
        return {
            'P_up': proba[:, 1],
            'expected_return': (proba[:, 1] - 0.5) * 0.02,  # Rough estimate
            'tail_risk': np.where(proba[:, 1] > 0.5, -0.01, -0.03),
            'volatility': np.full(len(proba), 0.01),
        }
    
    @classmethod
    def from_config(
        cls,
        config: Dict,
        device: str = 'cpu',
        scenario_checkpoint: Optional[str] = None,
    ) -> 'DiffusionExpert':
        """
        Create DiffusionExpert from configuration.
        
        Args:
            config: DIFFUSION_EXPERT config block
            device: Torch device
            scenario_checkpoint: Path to scenario model checkpoint
            
        Returns:
            DiffusionExpert instance
        """
        # Load scenario generator if checkpoint provided
        scenario_generator = None
        
        if scenario_checkpoint and Path(scenario_checkpoint).exists():
            scenario_generator = ScenarioGenerator.from_checkpoint(
                scenario_checkpoint,
                config=config,
                device=device,
            )
        
        return cls(
            scenario_generator=scenario_generator,
            config=config,
            device=device,
        )


if __name__ == "__main__":
    print("[DiffusionExpert Test]")
    
    # Create expert without scenario generator (will use fallback)
    expert = DiffusionExpert(
        scenario_generator=None,
        config={'enabled': True, 'use_calibration_head': False},
        device='cpu',
    )
    
    # Create dummy data
    n = 200
    df = pd.DataFrame({
        'close': np.cumsum(np.random.randn(n) * 0.01) + 100,
        'log_return': np.random.randn(n) * 0.01,
        'frac_diff': np.random.randn(n) * 0.01,
        'vol_5': np.abs(np.random.randn(n) * 0.01),
        'vol_10': np.abs(np.random.randn(n) * 0.01),
    })
    y = (np.random.randn(n) > 0).astype(float)
    
    # Fit
    expert.fit(df, y)
    
    # Predict
    proba = expert.predict_proba(df)
    print(f"  Proba shape: {proba.shape}")
    print(f"  Proba range: [{proba[:, 1].min():.4f}, {proba[:, 1].max():.4f}]")
    
    # Predict risk
    risk = expert.predict_risk(df)
    print(f"  Risk metrics: {list(risk.keys())}")
    
    print("[OK] DiffusionExpert test passed!")
