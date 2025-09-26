"""
Trading Rules Template
Provides template structures for trading rules and strategies
"""

from typing import Dict, List, Any, Optional
from datetime import datetime

def get_trading_rules_prompt(symbol: str, market_condition: str = "neutral") -> str:
    """Generate trading rules prompt for AI models"""
    return f"""
    Generate a comprehensive trading strategy for {symbol} considering the current market condition: {market_condition}.
    
    Please provide a detailed JSON response with the following structure:
    {{
        "strategy_name": "Descriptive strategy name",
        "description": "Detailed strategy description",
        "entry_conditions": [
            {{"condition": "condition_name", "parameters": {{"key": "value"}}, "confidence": 0.8}}
        ],
        "exit_conditions": [
            {{"condition": "condition_name", "parameters": {{"key": "value"}}, "confidence": 0.9}}
        ],
        "risk_management": {{
            "max_position_size": 0.1,
            "stop_loss": 0.02,
            "take_profit": 0.05,
            "max_holding_days": 5
        }},
        "technical_indicators": ["RSI", "MA", "Volume"],
        "market_conditions": ["trending", "volatile"],
        "confidence_score": 0.8,
        "expected_win_rate": 0.6
    }}
    """

def validate_strategy_performance(backtest_results: Dict[str, Any]) -> Dict[str, Any]:
    """Validate strategy performance based on backtest results"""
    total_return = backtest_results.get('total_return', 0)
    win_rate = backtest_results.get('win_rate', 0)
    max_drawdown = backtest_results.get('max_drawdown', 0)
    total_trades = backtest_results.get('total_trades', 0)
    
    validation = {
        'is_profitable': total_return > 0,
        'acceptable_win_rate': win_rate >= 45,
        'acceptable_drawdown': max_drawdown <= 20,
        'sufficient_trades': total_trades >= 5,
        'overall_score': 0
    }
    
    # Calculate overall score
    score = 0
    if validation['is_profitable']:
        score += 40
    if validation['acceptable_win_rate']:
        score += 30
    if validation['acceptable_drawdown']:
        score += 20
    if validation['sufficient_trades']:
        score += 10
    
    validation['overall_score'] = score
    validation['needs_improvement'] = score < 70
    
    return validation

def generate_strategy_improvement_prompt(strategy: Dict[str, Any], backtest_results: Dict[str, Any]) -> str:
    """Generate prompt for strategy improvement based on backtest results"""
    total_return = backtest_results.get('total_return', 0)
    win_rate = backtest_results.get('win_rate', 0)
    max_drawdown = backtest_results.get('max_drawdown', 0)
    
    return f"""
    Improve the following trading strategy based on poor backtest performance:
    
    Current Strategy: {strategy}
    
    Performance Issues:
    - Total Return: {total_return:.2f}% (Target: >0%)
    - Win Rate: {win_rate:.1f}% (Target: >45%)
    - Max Drawdown: {max_drawdown:.2f}% (Target: <20%)
    
    Please provide an improved strategy that addresses these performance issues.
    Focus on:
    1. Improving entry/exit conditions
    2. Better risk management
    3. More selective trade criteria
    
    Return the improved strategy in the same JSON format.
    """

class TradingRulesTemplate:
    """Template for creating trading rules and strategies"""
    
    @staticmethod
    def get_basic_strategy_template() -> Dict[str, Any]:
        """Get basic strategy template"""
        return {
            'strategy_name': 'Basic Strategy Template',
            'description': 'Template for creating basic trading strategies',
            'entry_conditions': [
                {
                    'condition': 'rsi_oversold',
                    'parameters': {'threshold': 30},
                    'confidence': 0.7
                },
                {
                    'condition': 'volume_spike',
                    'parameters': {'threshold': 1.5},
                    'confidence': 0.6
                }
            ],
            'exit_conditions': [
                {
                    'condition': 'profit_target',
                    'parameters': {'threshold': 0.05},
                    'confidence': 0.9
                },
                {
                    'condition': 'stop_loss',
                    'parameters': {'threshold': -0.02},
                    'confidence': 1.0
                }
            ],
            'risk_management': {
                'max_position_size': 0.1,
                'stop_loss': 0.02,
                'take_profit': 0.05,
                'max_holding_days': 5
            },
            'technical_indicators': ['RSI', 'Volume', 'MA'],
            'market_conditions': ['oversold', 'normal_volume'],
            'confidence_score': 0.7,
            'expected_win_rate': 0.6,
            'created_at': datetime.now().isoformat()
        }
    
    @staticmethod
    def get_multi_timeframe_template() -> Dict[str, Any]:
        """Get multi-timeframe strategy template"""
        return {
            'strategy_name': 'Multi-Timeframe Strategy Template',
            'description': 'Template for multi-timeframe trading strategies',
            'use_multi_timeframe': True,
            'entry_conditions': [
                {'condition': 'daily_momentum', 'parameters': {'threshold': -0.5}, 'confidence': 0.7},
                {'condition': '3day_swing', 'parameters': {'threshold': -1.0}, 'confidence': 0.75},
                {'condition': 'weekly_trend', 'parameters': {'threshold': 1.5}, 'confidence': 0.8}
            ],
            'exit_conditions': [
                {'condition': 'daily_profit', 'parameters': {'threshold': 0.02}, 'confidence': 0.9},
                {'condition': 'weekly_profit', 'parameters': {'threshold': 0.08}, 'confidence': 0.85}
            ],
            'risk_management': {
                'max_position_size': 0.3,
                'stop_loss': 0.02,
                'take_profit': 0.1,
                'max_holding_days': 30
            },
            'timeframe_allocations': {
                'daily': 0.2,
                '3day': 0.25,
                'weekly': 0.25,
                'biweekly': 0.15,
                'monthly': 0.15
            },
            'technical_indicators': ['RSI', 'MA', 'Volume', 'Momentum', 'MACD'],
            'market_conditions': ['trending', 'volatile', 'sideways'],
            'confidence_score': 0.8,
            'expected_win_rate': 0.65,
            'model_used': 'Multi-Timeframe',
            'created_at': datetime.now().isoformat()
        }
    
    @staticmethod
    def get_condition_templates() -> Dict[str, Dict[str, Any]]:
        """Get templates for different trading conditions"""
        return {
            'rsi_oversold': {
                'condition': 'rsi_oversold',
                'parameters': {'threshold': 30},
                'confidence': 0.7,
                'description': 'RSI below threshold indicates oversold condition'
            },
            'rsi_overbought': {
                'condition': 'rsi_overbought',
                'parameters': {'threshold': 70},
                'confidence': 0.7,
                'description': 'RSI above threshold indicates overbought condition'
            },
            'volume_spike': {
                'condition': 'volume_spike',
                'parameters': {'threshold': 1.5},
                'confidence': 0.6,
                'description': 'Volume above average indicates increased interest'
            },
            'price_breakout': {
                'condition': 'price_breakout',
                'parameters': {'threshold': 0.02},
                'confidence': 0.8,
                'description': 'Price breaks above resistance level'
            },
            'moving_average_cross': {
                'condition': 'ma_cross',
                'parameters': {'fast_period': 5, 'slow_period': 20},
                'confidence': 0.75,
                'description': 'Fast MA crosses above slow MA'
            }
        }
    
    @staticmethod
    def validate_strategy(strategy: Dict[str, Any]) -> List[str]:
        """Validate strategy structure and return list of issues"""
        issues = []
        
        # Check required fields
        required_fields = ['strategy_name', 'entry_conditions', 'exit_conditions', 'risk_management']
        for field in required_fields:
            if field not in strategy:
                issues.append(f"Missing required field: {field}")
        
        # Validate entry conditions
        if 'entry_conditions' in strategy:
            for i, condition in enumerate(strategy['entry_conditions']):
                if 'condition' not in condition:
                    issues.append(f"Entry condition {i} missing 'condition' field")
                if 'confidence' not in condition:
                    issues.append(f"Entry condition {i} missing 'confidence' field")
                elif not 0 <= condition['confidence'] <= 1:
                    issues.append(f"Entry condition {i} confidence must be between 0 and 1")
        
        # Validate exit conditions
        if 'exit_conditions' in strategy:
            for i, condition in enumerate(strategy['exit_conditions']):
                if 'condition' not in condition:
                    issues.append(f"Exit condition {i} missing 'condition' field")
                if 'confidence' not in condition:
                    issues.append(f"Exit condition {i} missing 'confidence' field")
        
        # Validate risk management
        if 'risk_management' in strategy:
            rm = strategy['risk_management']
            if 'max_position_size' in rm and not 0 < rm['max_position_size'] <= 1:
                issues.append("max_position_size must be between 0 and 1")
            if 'stop_loss' in rm and rm['stop_loss'] <= 0:
                issues.append("stop_loss must be positive")
            if 'take_profit' in rm and rm['take_profit'] <= 0:
                issues.append("take_profit must be positive")
        
        return issues

# Global template instance
trading_rules_template = TradingRulesTemplate()