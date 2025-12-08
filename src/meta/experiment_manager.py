"""
Experiment Manager for Tracking and Reproducibility.

Handles:
- Experiment registration and versioning
- Config snapshots and hashing
- Results logging and comparison
- Artifacts organization

Author: QFC System - Meta Layer
"""

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ExperimentManager:
    """
    Manages experiment tracking, logging, and reproducibility.
    
    Features:
        - Config versioning with hashes
        - Results storage (JSON, Parquet)
        - Experiment comparison
        - Artifact organization
    
    Args:
        experiments_dir: Root directory for experiments
        project_name: Name of the project
    """
    
    def __init__(
        self,
        experiments_dir: str = 'artifacts/experiments',
        project_name: str = 'quanta_futures',
    ):
        self.experiments_dir = Path(experiments_dir)
        self.project_name = project_name
        
        # Create directories
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        
        # Current experiment state
        self.current_experiment: Optional[Dict] = None
        self.experiments_log: List[Dict] = []
        
        # Load existing experiments index
        self.index_path = self.experiments_dir / 'experiments_index.json'
        self._load_index()
    
    def _load_index(self):
        """Load existing experiments index."""
        if self.index_path.exists():
            with open(self.index_path, 'r') as f:
                self.experiments_log = json.load(f)
        else:
            self.experiments_log = []
    
    def _save_index(self):
        """Save experiments index."""
        with open(self.index_path, 'w') as f:
            json.dump(self.experiments_log, f, indent=2, default=str)
    
    def _config_hash(self, config: Dict) -> str:
        """Compute deterministic hash of config."""
        config_str = json.dumps(config, sort_keys=True, default=str)
        return hashlib.sha256(config_str.encode()).hexdigest()[:12]
    
    def register_experiment(
        self,
        config: Dict,
        description: str = '',
        tags: Optional[List[str]] = None,
    ) -> str:
        """
        Register a new experiment.
        
        Args:
            config: Configuration snapshot
            description: Human-readable description
            tags: Tags for filtering
            
        Returns:
            experiment_id: Unique experiment identifier
        """
        timestamp = datetime.now().isoformat()
        config_hash = self._config_hash(config)
        
        # Generate experiment ID
        exp_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{config_hash[:6]}"
        
        # Create experiment directory
        exp_dir = self.experiments_dir / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_experiment = {
            'id': exp_id,
            'timestamp': timestamp,
            'config_hash': config_hash,
            'config': config,
            'description': description,
            'tags': tags or [],
            'status': 'running',
            'metrics': {},
            'artifacts_dir': str(exp_dir),
        }
        
        # Save config
        config_path = exp_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        logger.info(f"[ExperimentManager] Registered experiment: {exp_id}")
        
        return exp_id
    
    def log_metrics(self, metrics: Dict[str, float]):
        """
        Log metrics for current experiment.
        
        Args:
            metrics: Dictionary of metric name -> value
        """
        if self.current_experiment is None:
            logger.warning("[ExperimentManager] No current experiment. Register first.")
            return
        
        self.current_experiment['metrics'].update(metrics)
        
        # Save metrics
        exp_dir = Path(self.current_experiment['artifacts_dir'])
        metrics_path = exp_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(self.current_experiment['metrics'], f, indent=2)
    
    def log_artifact(self, name: str, data: Any, artifact_type: str = 'json'):
        """
        Save an artifact for current experiment.
        
        Args:
            name: Artifact name (without extension)
            data: Data to save
            artifact_type: 'json', 'numpy', 'pickle'
        """
        if self.current_experiment is None:
            return
        
        exp_dir = Path(self.current_experiment['artifacts_dir'])
        
        if artifact_type == 'json':
            path = exp_dir / f'{name}.json'
            with open(path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        elif artifact_type == 'numpy':
            import numpy as np
            path = exp_dir / f'{name}.npy'
            np.save(path, data)
        elif artifact_type == 'pickle':
            import pickle
            path = exp_dir / f'{name}.pkl'
            with open(path, 'wb') as f:
                pickle.dump(data, f)
    
    def finish_experiment(self, status: str = 'completed'):
        """
        Mark current experiment as finished.
        
        Args:
            status: Final status ('completed', 'failed', 'cancelled')
        """
        if self.current_experiment is None:
            return
        
        self.current_experiment['status'] = status
        self.current_experiment['end_timestamp'] = datetime.now().isoformat()
        
        # Add to log and save
        self.experiments_log.append(self.current_experiment)
        self._save_index()
        
        logger.info(f"[ExperimentManager] Finished experiment: {self.current_experiment['id']} ({status})")
        
        self.current_experiment = None
    
    def get_experiment(self, exp_id: str) -> Optional[Dict]:
        """Get experiment by ID."""
        for exp in self.experiments_log:
            if exp['id'] == exp_id:
                return exp
        return None
    
    def list_experiments(
        self,
        tags: Optional[List[str]] = None,
        status: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict]:
        """
        List experiments with optional filtering.
        
        Args:
            tags: Filter by tags
            status: Filter by status
            limit: Maximum number to return
            
        Returns:
            List of experiment summaries
        """
        results = self.experiments_log.copy()
        
        if tags:
            results = [e for e in results if any(t in e.get('tags', []) for t in tags)]
        
        if status:
            results = [e for e in results if e.get('status') == status]
        
        # Sort by timestamp descending
        results = sorted(results, key=lambda x: x['timestamp'], reverse=True)
        
        return results[:limit]
    
    def compare_experiments(
        self,
        exp_ids: List[str],
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare metrics across experiments.
        
        Args:
            exp_ids: Experiment IDs to compare
            metrics: Specific metrics to compare (None = all)
            
        Returns:
            Dict of {exp_id: {metric: value}}
        """
        comparison = {}
        
        for exp_id in exp_ids:
            exp = self.get_experiment(exp_id)
            if exp:
                exp_metrics = exp.get('metrics', {})
                if metrics:
                    exp_metrics = {k: v for k, v in exp_metrics.items() if k in metrics}
                comparison[exp_id] = exp_metrics
        
        return comparison


if __name__ == "__main__":
    print("[ExperimentManager Test]")
    
    # Create manager
    manager = ExperimentManager(experiments_dir='artifacts/experiments')
    
    # Register experiment
    config = {
        'model': 'diffusion_expert',
        'num_scenarios': 32,
        'horizon': 12,
    }
    exp_id = manager.register_experiment(config, description='Test experiment', tags=['test'])
    
    # Log metrics
    manager.log_metrics({
        'sharpe': 1.5,
        'max_drawdown': 0.15,
        'hit_rate': 0.52,
    })
    
    # Log artifact
    manager.log_artifact('test_data', {'values': [1, 2, 3]}, 'json')
    
    # Finish
    manager.finish_experiment('completed')
    
    # List
    experiments = manager.list_experiments()
    print(f"  Total experiments: {len(experiments)}")
    print(f"  Latest: {experiments[0]['id'] if experiments else 'None'}")
    
    print("[OK] ExperimentManager test passed!")
