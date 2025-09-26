import json
import random
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np

class AIStrategyOptimizer:
    """
    AI Strategy Optimizer that analyzes backtest results and generates improved strategies
    """
    
    def __init__(self):
        self.optimization_history = []
        self.best_strategy = None
        self.best_performance = None
        self.max_iterations = 10
        self.target_return_threshold = 2.0  # Target minimum return percentage
        
    def analyze_performance_issues(self, backtest_results: Dict[str, Any]) -> List[str]:
        """
        Analyze backtest results to identify performance issues
        
        Args:
            backtest_results: Results from AI backtester
            
        Returns:
            List of identified issues
        """
        issues = []
        
        performance = backtest_results.get('performance_metrics', {})
        trading = backtest_results.get('trading_statistics', {})
        daily_analysis = backtest_results.get('daily_analysis', {})
        
        # Check overall performance
        total_return = performance.get('total_return_pct', 0)
        if total_return < 0:
            issues.append("negative_returns")
            
        # Check win rate
        win_rate = trading.get('win_rate_pct', 0)
        if win_rate < 50:
            issues.append("low_win_rate")
            
        # Check trading frequency
        trading_frequency = daily_analysis.get('daily_trading_frequency_pct', 0)
        if trading_frequency < 20:
            issues.append("low_trading_frequency")
        elif trading_frequency > 80:
            issues.append("high_trading_frequency")
            
        # Check drawdown
        max_drawdown = performance.get('max_drawdown_pct', 0)
        if max_drawdown > 10:
            issues.append("high_drawdown")
            
        # Check average trade return
        avg_trade_return = trading.get('avg_trade_return_pct', 0)
        if avg_trade_return < 0:
            issues.append("negative_avg_trade_return")
            
        # Check holding period
        avg_holding_period = trading.get('avg_holding_period_days', 0)
        if avg_holding_period < 1:
            issues.append("short_holding_period")
        elif avg_holding_period > 10:
            issues.append("long_holding_period")
            
        return issues
        
    def generate_optimization_rules(self, issues: List[str], current_strategy: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate optimization rules based on identified issues
        
        Args:
            issues: List of performance issues
            current_strategy: Current strategy configuration
            
        Returns:
            Optimized strategy configuration
        """
        optimized_strategy = current_strategy.copy()
        
        # Get current parameters
        risk_mgmt = optimized_strategy.get('risk_management', {})
        entry_conditions = optimized_strategy.get('entry_conditions', [])
        exit_conditions = optimized_strategy.get('exit_conditions', [])
        
        # Optimization based on issues
        if "negative_returns" in issues or "negative_avg_trade_return" in issues:
            # Tighten entry conditions and improve exit strategy
            risk_mgmt['stop_loss_pct'] = max(0.01, risk_mgmt.get('stop_loss_pct', 0.02) * 0.8)
            risk_mgmt['take_profit_pct'] = min(0.08, risk_mgmt.get('take_profit_pct', 0.04) * 1.2)
            
            # Add more selective entry conditions
            if not any('rsi' in str(condition).lower() for condition in entry_conditions):
                entry_conditions.append({
                    'condition': 'RSI oversold with momentum confirmation',
                    'parameters': {
                        'rsi_threshold': 35,
                        'momentum_confirmation': True
                    },
                    'weight': 0.8
                })
                
        if "low_win_rate" in issues:
            # Make entry conditions more selective
            for condition in entry_conditions:
                if 'weight' in condition:
                    condition['weight'] = min(1.0, condition['weight'] * 1.1)
                    
            # Add trend confirmation
            entry_conditions.append({
                'condition': 'Trend confirmation with moving average',
                'parameters': {
                    'ma_period': 20,
                    'trend_strength': 0.7
                },
                'weight': 0.9
            })
            
        if "low_trading_frequency" in issues:
            # Relax entry conditions slightly
            risk_mgmt['max_position_size'] = min(0.15, risk_mgmt.get('max_position_size', 0.1) * 1.2)
            
            # Add more entry signals
            entry_conditions.append({
                'condition': 'Volume breakout with price momentum',
                'parameters': {
                    'volume_multiplier': 1.5,
                    'price_momentum_threshold': 0.02
                },
                'weight': 0.7
            })
            
        if "high_trading_frequency" in issues:
            # Make entry conditions more restrictive
            for condition in entry_conditions:
                if 'weight' in condition:
                    condition['weight'] = max(0.5, condition['weight'] * 0.9)
                    
        if "high_drawdown" in issues:
            # Implement stricter risk management
            risk_mgmt['stop_loss_pct'] = max(0.005, risk_mgmt.get('stop_loss_pct', 0.02) * 0.7)
            risk_mgmt['max_position_size'] = max(0.05, risk_mgmt.get('max_position_size', 0.1) * 0.8)
            
        if "short_holding_period" in issues:
            # Encourage longer holds with better exit conditions
            risk_mgmt['take_profit_pct'] = min(0.1, risk_mgmt.get('take_profit_pct', 0.04) * 1.3)
            
            # Add time-based exit condition
            exit_conditions.append({
                'condition': 'Minimum holding period',
                'parameters': {
                    'min_days': 2
                },
                'weight': 0.6
            })
            
        if "long_holding_period" in issues:
            # Add time-based exit to prevent overholding
            exit_conditions.append({
                'condition': 'Maximum holding period',
                'parameters': {
                    'max_days': 7
                },
                'weight': 0.8
            })
            
        # Update strategy
        optimized_strategy['risk_management'] = risk_mgmt
        optimized_strategy['entry_conditions'] = entry_conditions
        optimized_strategy['exit_conditions'] = exit_conditions
        
        # Add optimization metadata
        optimized_strategy['optimization_iteration'] = len(self.optimization_history) + 1
        optimized_strategy['optimization_timestamp'] = datetime.now().isoformat()
        optimized_strategy['addressed_issues'] = issues
        
        return optimized_strategy
        
    def optimize_strategy(self, current_strategy: Dict[str, Any], backtest_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main optimization method that analyzes results and generates improved strategy
        
        Args:
            current_strategy: Current strategy configuration
            backtest_results: Backtest results to analyze
            
        Returns:
            Optimized strategy configuration
        """
        # Analyze performance issues
        issues = self.analyze_performance_issues(backtest_results)
        
        # Generate optimized strategy
        optimized_strategy = self.generate_optimization_rules(issues, current_strategy)
        
        # Record optimization history
        optimization_record = {
            'iteration': len(self.optimization_history) + 1,
            'timestamp': datetime.now().isoformat(),
            'original_performance': backtest_results.get('performance_metrics', {}),
            'identified_issues': issues,
            'optimization_changes': self._get_optimization_changes(current_strategy, optimized_strategy)
        }
        
        self.optimization_history.append(optimization_record)
        
        return optimized_strategy
        
    def _get_optimization_changes(self, original: Dict[str, Any], optimized: Dict[str, Any]) -> List[str]:
        """
        Identify what changes were made during optimization
        
        Args:
            original: Original strategy
            optimized: Optimized strategy
            
        Returns:
            List of changes made
        """
        changes = []
        
        # Check risk management changes
        orig_risk = original.get('risk_management', {})
        opt_risk = optimized.get('risk_management', {})
        
        if orig_risk.get('stop_loss_pct') != opt_risk.get('stop_loss_pct'):
            changes.append(f"Adjusted stop loss from {orig_risk.get('stop_loss_pct', 'N/A')} to {opt_risk.get('stop_loss_pct', 'N/A')}")
            
        if orig_risk.get('take_profit_pct') != opt_risk.get('take_profit_pct'):
            changes.append(f"Adjusted take profit from {orig_risk.get('take_profit_pct', 'N/A')} to {opt_risk.get('take_profit_pct', 'N/A')}")
            
        if orig_risk.get('max_position_size') != opt_risk.get('max_position_size'):
            changes.append(f"Adjusted position size from {orig_risk.get('max_position_size', 'N/A')} to {opt_risk.get('max_position_size', 'N/A')}")
            
        # Check if new conditions were added
        orig_entry_count = len(original.get('entry_conditions', []))
        opt_entry_count = len(optimized.get('entry_conditions', []))
        
        if opt_entry_count > orig_entry_count:
            changes.append(f"Added {opt_entry_count - orig_entry_count} new entry conditions")
            
        orig_exit_count = len(original.get('exit_conditions', []))
        opt_exit_count = len(optimized.get('exit_conditions', []))
        
        if opt_exit_count > orig_exit_count:
            changes.append(f"Added {opt_exit_count - orig_exit_count} new exit conditions")
            
        return changes
        
    def should_continue_optimization(self, backtest_results: Dict[str, Any]) -> bool:
        """
        Determine if optimization should continue based on results
        
        Args:
            backtest_results: Latest backtest results
            
        Returns:
            True if optimization should continue, False otherwise
        """
        # Check if we've reached max iterations
        if len(self.optimization_history) >= self.max_iterations:
            return False
            
        # Check if we've achieved target performance
        performance = backtest_results.get('performance_metrics', {})
        total_return = performance.get('total_return_pct', 0)
        
        if total_return >= self.target_return_threshold:
            return False
            
        # Check if we're making progress
        if len(self.optimization_history) >= 3:
            recent_returns = []
            for record in self.optimization_history[-3:]:
                recent_returns.append(record['original_performance'].get('total_return_pct', 0))
                
            # If no improvement in last 3 iterations, stop
            if len(set(recent_returns)) == 1:  # All same values
                return False
                
        return True
        
    def get_optimization_summary(self) -> Dict[str, Any]:
        """
        Get summary of optimization process
        
        Returns:
            Optimization summary
        """
        if not self.optimization_history:
            return {'status': 'No optimization performed'}
            
        return {
            'total_iterations': len(self.optimization_history),
            'optimization_history': self.optimization_history,
            'best_performance': self.best_performance,
            'final_status': 'Optimization completed' if self.best_performance else 'Optimization in progress'
        }

# Global instance
strategy_optimizer = AIStrategyOptimizer()