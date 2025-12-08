"""
Scenario Generator API.

High-level wrapper for generating and analyzing future scenarios.
Used by DiffusionExpert and analysis tools.

Author: QFC System - Diffusion Architecture
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


class ScenarioGenerator:
    """
    High-level API for generating future scenarios from past data.
    
    Wraps DiffusionScenarioModel with normalization, caching, and statistics.
    
    Args:
        model: Trained DiffusionScenarioModel
        config: Configuration dict
        device: Torch device
    """
    
    def __init__(
        self,
        model=None,
        config: Optional[Dict] = None,
        device: str = 'cpu',
    ):
        self.model = model
        self.config = config or {}
        self.device = device
        
        # Config with defaults
        self.num_scenarios = self.config.get('num_scenarios', 32)
        self.horizon = self.config.get('horizon', 12)
        self.inference_steps = self.config.get('inference_steps', 50)
        
        # Normalization stats
        self.past_mean_ = self.config.get('past_mean')
        self.past_std_ = self.config.get('past_std')
        self.future_mean_ = self.config.get('future_mean')
        self.future_std_ = self.config.get('future_std')
        
        if self.model is not None:
            self.model.to(device)
            self.model.eval()
    
    def generate(
        self,
        past_window: Union[np.ndarray, torch.Tensor],
        num_samples: Optional[int] = None,
        horizon: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate future scenarios given past conditioning.
        
        Args:
            past_window: [C_past, L_past] or [1, C_past, L_past] past features
            num_samples: Number of scenarios (default: self.num_scenarios)
            horizon: Not used (model has fixed horizon)
            
        Returns:
            scenarios: [K, C_target, H] future scenarios
        """
        if self.model is None:
            raise RuntimeError("No model loaded")
        
        num_samples = num_samples or self.num_scenarios
        
        # Convert to tensor
        if isinstance(past_window, np.ndarray):
            past_window = torch.tensor(past_window, dtype=torch.float32)
        
        past_window = past_window.to(self.device)
        
        # Ensure batch dimension
        if past_window.dim() == 2:
            past_window = past_window.unsqueeze(0)
        
        # Normalize if stats available
        if self.past_mean_ is not None and self.past_std_ is not None:
            mean = torch.tensor(self.past_mean_, device=self.device, dtype=torch.float32)
            std = torch.tensor(self.past_std_, device=self.device, dtype=torch.float32)
            if mean.dim() == 1:
                mean = mean[:, None]
                std = std[:, None]
            past_window = (past_window - mean) / (std + 1e-8)
        
        # Generate
        with torch.no_grad():
            scenarios = self.model.generate_scenarios(
                past_window,
                num_samples=num_samples,
                num_inference_steps=self.inference_steps,
            )
        
        # Denormalize output if stats available
        if self.future_mean_ is not None and self.future_std_ is not None:
            future_mean = torch.tensor(self.future_mean_, device=self.device)
            future_std = torch.tensor(self.future_std_, device=self.device)
            scenarios = scenarios * future_std + future_mean
        
        return scenarios
    
    def compute_statistics(
        self,
        scenarios: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute summary statistics from generated scenarios.
        
        Args:
            scenarios: [K, C, H] generated scenarios
            
        Returns:
            stats dict with P_up, expected_return, volatility, tail_risk, etc.
        """
        # Sum of returns across horizon
        total_returns = scenarios[:, 0, :].sum(dim=1)  # [K]
        
        # Probability of positive return
        P_up = (total_returns > 0).float().mean()
        
        # Expected return
        E_return = total_returns.mean()
        
        # Volatility (std of total returns)
        volatility = total_returns.std()
        
        # Tail risk (5th percentile)
        tail_risk = torch.quantile(total_returns, 0.05)
        
        # Max drawdown per scenario
        cum_returns = scenarios[:, 0, :].cumsum(dim=1)  # [K, H]
        running_max = cum_returns.cummax(dim=1)[0]
        drawdowns = running_max - cum_returns
        max_drawdown = drawdowns.max(dim=1)[0].mean()
        
        return {
            'P_up': P_up,
            'E_return': E_return,
            'volatility': volatility,
            'tail_risk': tail_risk,
            'max_drawdown': max_drawdown,
            'total_returns': total_returns,
        }
    
    def generate_and_analyze(
        self,
        past_window: Union[np.ndarray, torch.Tensor],
        num_samples: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Generate scenarios and compute statistics in one call.
        
        Args:
            past_window: Past feature window
            num_samples: Number of scenarios
            
        Returns:
            scenarios: [K, C, H] generated scenarios
            stats: Statistics dict
        """
        scenarios = self.generate(past_window, num_samples)
        stats = self.compute_statistics(scenarios)
        return scenarios, stats
    
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        config: Optional[Dict] = None,
        device: str = 'cpu',
    ) -> 'ScenarioGenerator':
        """
        Load from checkpoint.
        
        Args:
            checkpoint_path: Path to model checkpoint
            config: Configuration overrides
            device: Torch device
            
        Returns:
            ScenarioGenerator instance
        """
        from .diffusion_scenario import DiffusionScenarioModel
        
        path = Path(checkpoint_path)
        if not path.exists():
            logger.warning(f"[ScenarioGenerator] Checkpoint not found: {path}")
            return cls(model=None, config=config, device=device)
        
        checkpoint = torch.load(path, map_location=device)
        model_config = checkpoint.get('config', {})
        
        model = DiffusionScenarioModel(
            in_channels_past=model_config.get('in_channels_past', 16),
            in_channels_future=model_config.get('in_channels_future', 1),
            L_past=model_config.get('L_past', 96),
            H_future=model_config.get('H_future', 12),
            model_channels=model_config.get('model_channels', 64),
            cond_dim=model_config.get('cond_dim', 128),
            num_timesteps=model_config.get('num_timesteps', 1000),
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        # Merge configs
        full_config = {
            **model_config,
            **(config or {}),
            'past_mean': checkpoint.get('past_mean'),
            'past_std': checkpoint.get('past_std'),
            'future_mean': checkpoint.get('future_mean'),
            'future_std': checkpoint.get('future_std'),
        }
        
        return cls(model=model, config=full_config, device=device)


class ScenarioOracle:
    """
    Evaluate expert strategies on generated scenarios.
    
    Useful for offline policy evaluation and regime analysis.
    """
    
    def __init__(
        self,
        scenario_generator: ScenarioGenerator,
        config: Optional[Dict] = None,
    ):
        self.generator = scenario_generator
        self.config = config or {}
    
    def evaluate_expert(
        self,
        expert_fn,
        past_window: Union[np.ndarray, torch.Tensor],
        num_samples: int = 100,
    ) -> Dict[str, float]:
        """
        Evaluate an expert's expected performance on scenarios.
        
        Args:
            expert_fn: Callable that takes scenarios [K, C, H] and returns 
                       positions [K] (allocation per scenario, -1 to 1)
            past_window: Past conditioning
            num_samples: Number of scenarios
            
        Returns:
            Evaluation metrics
        """
        # Generate scenarios
        scenarios, stats = self.generator.generate_and_analyze(past_window, num_samples)
        
        # Get positions from expert
        positions = expert_fn(scenarios)
        
        if isinstance(positions, torch.Tensor):
            positions = positions.cpu().numpy()
        
        # Compute PnL per scenario
        total_returns = stats['total_returns'].cpu().numpy()
        pnl = positions * total_returns
        
        # Metrics
        expected_pnl = pnl.mean()
        sharpe = pnl.mean() / (pnl.std() + 1e-8)
        hit_rate = (pnl > 0).mean()
        tail_risk = np.percentile(pnl, 5)
        
        return {
            'expected_pnl': float(expected_pnl),
            'sharpe': float(sharpe),
            'hit_rate': float(hit_rate),
            'tail_risk': float(tail_risk),
            'n_scenarios': num_samples,
            'avg_position': float(np.abs(positions).mean()),
        }
    
    def compare_experts(
        self,
        expert_fns: Dict[str, callable],
        past_window: Union[np.ndarray, torch.Tensor],
        num_samples: int = 100,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple experts on same scenarios.
        
        Args:
            expert_fns: Dict of {name: expert_fn}
            past_window: Past conditioning
            num_samples: Number of scenarios
            
        Returns:
            Dict of {name: metrics}
        """
        # Generate scenarios once
        scenarios = self.generator.generate(past_window, num_samples)
        stats = self.generator.compute_statistics(scenarios)
        total_returns = stats['total_returns'].cpu().numpy()
        
        results = {}
        for name, expert_fn in expert_fns.items():
            positions = expert_fn(scenarios)
            if isinstance(positions, torch.Tensor):
                positions = positions.cpu().numpy()
            
            pnl = positions * total_returns
            
            results[name] = {
                'expected_pnl': float(pnl.mean()),
                'sharpe': float(pnl.mean() / (pnl.std() + 1e-8)),
                'hit_rate': float((pnl > 0).mean()),
            }
        
        return results
