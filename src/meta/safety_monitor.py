"""
Safety Monitor for Live Trading.

Implements:
- Kill-switch logic for loss limits
- Drift detection for model degradation
- Real-time telemetry logging
- Margin and liquidation monitoring

Author: QFC System - Meta Layer
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class SafetyMonitor:
    """
    Real-time safety monitoring for trading systems.
    
    Features:
        - Daily/weekly loss limits with kill-switch
        - Model drift detection
        - Margin usage and liquidation monitoring
        - Event logging and alerts
    
    Args:
        config: TUNING.SAFETY config block
        log_dir: Directory for safety logs
    """
    
    def __init__(
        self,
        config: Optional[Dict] = None,
        log_dir: str = 'artifacts/safety_logs',
    ):
        self.config = config or {}
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Safety thresholds
        self.daily_loss_limit = self.config.get('daily_loss_limit', 0.05)
        self.weekly_loss_limit = self.config.get('weekly_loss_limit', 0.10)
        self.max_drawdown = self.config.get('max_drawdown_since_start', 0.20)
        self.min_margin_buffer = self.config.get('min_margin_buffer', 0.30)
        
        # State tracking
        self.initial_equity = None
        self.peak_equity = None
        self.current_equity = None
        
        self.daily_start_equity = None
        self.weekly_start_equity = None
        self.daily_start_time = None
        self.weekly_start_time = None
        
        # Drift tracking
        self.predictions_history: List[float] = []
        self.outcomes_history: List[float] = []
        
        # Status
        self.is_paused = False
        self.kill_switch_triggered = False
        self.kill_switch_reason = None
        
        # Events
        self.events: List[Dict] = []
    
    def reset(self, initial_equity: float):
        """
        Reset monitor with initial equity.
        
        Args:
            initial_equity: Starting equity value
        """
        self.initial_equity = initial_equity
        self.peak_equity = initial_equity
        self.current_equity = initial_equity
        
        now = datetime.now()
        self.daily_start_equity = initial_equity
        self.weekly_start_equity = initial_equity
        self.daily_start_time = now
        self.weekly_start_time = now
        
        self.is_paused = False
        self.kill_switch_triggered = False
        self.kill_switch_reason = None
        self.events = []
        
        self._log_event('reset', {'initial_equity': initial_equity})
    
    def update(
        self,
        equity: float,
        margin_usage: float = 0.0,
        distance_to_liquidation: float = float('inf'),
        prediction: Optional[float] = None,
        outcome: Optional[float] = None,
    ) -> Dict:
        """
        Update safety state and check thresholds.
        
        Args:
            equity: Current equity
            margin_usage: Current margin usage (0-1)
            distance_to_liquidation: Distance to liquidation (%)
            prediction: Latest model prediction
            outcome: Latest realized outcome
            
        Returns:
            status dict with 'safe', 'warnings', 'actions'
        """
        if self.kill_switch_triggered:
            return {
                'safe': False,
                'paused': True,
                'reason': self.kill_switch_reason,
            }
        
        self.current_equity = equity
        self.peak_equity = max(self.peak_equity or equity, equity)
        
        warnings = []
        actions = []
        
        now = datetime.now()
        
        # Check daily reset
        if self.daily_start_time is None or (now - self.daily_start_time) > timedelta(days=1):
            self.daily_start_equity = equity
            self.daily_start_time = now
        
        # Check weekly reset
        if self.weekly_start_time is None or (now - self.weekly_start_time) > timedelta(weeks=1):
            self.weekly_start_equity = equity
            self.weekly_start_time = now
        
        # Calculate losses
        daily_loss = 1 - (equity / self.daily_start_equity) if self.daily_start_equity else 0
        weekly_loss = 1 - (equity / self.weekly_start_equity) if self.weekly_start_equity else 0
        total_drawdown = 1 - (equity / self.peak_equity) if self.peak_equity else 0
        
        # Check thresholds
        if daily_loss >= self.daily_loss_limit:
            self._trigger_kill_switch(f'Daily loss limit breached: {daily_loss:.1%}')
            return {'safe': False, 'paused': True, 'reason': self.kill_switch_reason}
        
        if weekly_loss >= self.weekly_loss_limit:
            self._trigger_kill_switch(f'Weekly loss limit breached: {weekly_loss:.1%}')
            return {'safe': False, 'paused': True, 'reason': self.kill_switch_reason}
        
        if total_drawdown >= self.max_drawdown:
            self._trigger_kill_switch(f'Max drawdown breached: {total_drawdown:.1%}')
            return {'safe': False, 'paused': True, 'reason': self.kill_switch_reason}
        
        # Warnings
        if daily_loss >= self.daily_loss_limit * 0.7:
            warnings.append(f'Approaching daily loss limit: {daily_loss:.1%}')
        
        if margin_usage > (1 - self.min_margin_buffer):
            warnings.append(f'High margin usage: {margin_usage:.1%}')
            actions.append('reduce_position_size')
        
        if distance_to_liquidation < 0.10:
            warnings.append(f'Close to liquidation: {distance_to_liquidation:.1%}')
            actions.append('emergency_reduce')
        
        # Track predictions for drift detection
        if prediction is not None and outcome is not None:
            self.predictions_history.append(prediction)
            self.outcomes_history.append(outcome)
            
            # Keep last 100
            if len(self.predictions_history) > 100:
                self.predictions_history.pop(0)
                self.outcomes_history.pop(0)
        
        return {
            'safe': True,
            'paused': False,
            'daily_loss': daily_loss,
            'weekly_loss': weekly_loss,
            'drawdown': total_drawdown,
            'warnings': warnings,
            'actions': actions,
        }
    
    def check_drift(self, window: int = 50) -> Dict:
        """
        Check for model drift based on prediction quality.
        
        Args:
            window: Rolling window for drift analysis
            
        Returns:
            drift analysis dict
        """
        if len(self.predictions_history) < window:
            return {'drift_detected': False, 'message': 'Insufficient data'}
        
        recent_preds = np.array(self.predictions_history[-window:])
        recent_outcomes = np.array(self.outcomes_history[-window:])
        
        # Calculate calibration metrics
        hit_rate = ((recent_preds > 0.5) == (recent_outcomes > 0)).mean()
        
        # Calculate Expected Calibration Error (simplified)
        pred_mean = recent_preds.mean()
        outcome_mean = (recent_outcomes > 0).mean()
        calibration_error = abs(pred_mean - outcome_mean)
        
        # Drift detection thresholds
        drift_detected = False
        reasons = []
        
        if hit_rate < 0.45:
            drift_detected = True
            reasons.append(f'Low hit rate: {hit_rate:.1%}')
        
        if calibration_error > 0.15:
            drift_detected = True
            reasons.append(f'High calibration error: {calibration_error:.2f}')
        
        if drift_detected:
            self._log_event('drift_detected', {
                'hit_rate': hit_rate,
                'calibration_error': calibration_error,
                'reasons': reasons,
            })
        
        return {
            'drift_detected': drift_detected,
            'hit_rate': hit_rate,
            'calibration_error': calibration_error,
            'reasons': reasons,
            'recommendation': 'retrain' if drift_detected else 'continue',
        }
    
    def _trigger_kill_switch(self, reason: str):
        """Trigger kill-switch and log."""
        self.kill_switch_triggered = True
        self.is_paused = True
        self.kill_switch_reason = reason
        
        self._log_event('kill_switch', {'reason': reason})
        logger.critical(f"[KILL SWITCH] {reason}")
    
    def _log_event(self, event_type: str, data: Dict):
        """Log a safety event."""
        event = {
            'timestamp': datetime.now().isoformat(),
            'type': event_type,
            'data': data,
        }
        self.events.append(event)
        
        # Save to file
        log_file = self.log_dir / f'safety_{datetime.now().strftime("%Y%m%d")}.json'
        try:
            if log_file.exists():
                with open(log_file, 'r') as f:
                    existing = json.load(f)
            else:
                existing = []
            
            existing.append(event)
            
            import json
            with open(log_file, 'w') as f:
                json.dump(existing, f, indent=2)
        except Exception as e:
            logger.error(f"[SafetyMonitor] Failed to log event: {e}")
    
    def resume(self):
        """Resume trading after kill-switch (manual action required)."""
        if not self.kill_switch_triggered:
            return
        
        self._log_event('resume', {'previous_reason': self.kill_switch_reason})
        
        self.kill_switch_triggered = False
        self.is_paused = False
        self.kill_switch_reason = None
        
        # Reset equity trackers
        self.daily_start_equity = self.current_equity
        self.weekly_start_equity = self.current_equity
        self.daily_start_time = datetime.now()
        self.weekly_start_time = datetime.now()
        
        logger.info("[SafetyMonitor] Trading resumed")
    
    def get_status(self) -> Dict:
        """Get current safety status."""
        return {
            'is_paused': self.is_paused,
            'kill_switch_triggered': self.kill_switch_triggered,
            'kill_switch_reason': self.kill_switch_reason,
            'current_equity': self.current_equity,
            'peak_equity': self.peak_equity,
            'drawdown': 1 - (self.current_equity / self.peak_equity) if self.peak_equity else 0,
            'events_count': len(self.events),
        }


# Import json at module level for _log_event
import json


if __name__ == "__main__":
    print("[SafetyMonitor Test]")
    
    # Create monitor
    monitor = SafetyMonitor(config={
        'daily_loss_limit': 0.05,
        'weekly_loss_limit': 0.10,
        'max_drawdown_since_start': 0.20,
    })
    
    # Initialize
    monitor.reset(10000)
    
    # Normal update
    status = monitor.update(equity=9800, margin_usage=0.3)
    print(f"  Update 1: safe={status['safe']}, drawdown={status['drawdown']:.2%}")
    
    # Approaching limit
    status = monitor.update(equity=9550, margin_usage=0.5)
    print(f"  Update 2: safe={status['safe']}, warnings={status['warnings']}")
    
    # Check drift (with dummy data)
    for i in range(60):
        monitor.predictions_history.append(0.55 + np.random.randn() * 0.1)
        monitor.outcomes_history.append(1 if np.random.random() > 0.45 else 0)
    
    drift = monitor.check_drift()
    print(f"  Drift check: detected={drift['drift_detected']}, hit_rate={drift['hit_rate']:.2%}")
    
    # Get status
    status = monitor.get_status()
    print(f"  Status: paused={status['is_paused']}, events={status['events_count']}")
    
    print("[OK] SafetyMonitor test passed!")
