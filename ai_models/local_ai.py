"""
Local AI Model for Trading Strategy Generation
"""

import json
import requests
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

class LocalAIModel:
    """Local AI model using Ollama for strategy generation"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.logger = logging.getLogger(__name__)
        
    def generate_strategy(self, symbol: str, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate trading strategy using local AI model"""
        try:
            # Create prompt for strategy generation
            prompt = self._create_strategy_prompt(symbol, market_data)
            
            # Call Ollama API
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": "llama3.1:8b",
                    "prompt": prompt,
                    "stream": False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return self._parse_strategy_response(result.get('response', ''))
            else:
                self.logger.error(f"Ollama API error: {response.status_code}")
                return self._generate_fallback_strategy(symbol)
                
        except Exception as e:
            self.logger.error(f"Local AI generation failed: {e}")
            return self._generate_fallback_strategy(symbol)
    
    def _create_strategy_prompt(self, symbol: str, market_data: Dict[str, Any]) -> str:
        """Create prompt for AI strategy generation"""
        return f"""
        Generate a trading strategy for {symbol} based on the following market data:
        
        Current Price: ${market_data.get('current_price', 'N/A')}
        RSI: {market_data.get('rsi', 'N/A')}
        Volume Ratio: {market_data.get('volume_ratio', 'N/A')}
        
        Please provide a JSON response with the following structure:
        {{
            "strategy_name": "Strategy Name",
            "description": "Strategy description",
            "entry_conditions": [
                {{"condition": "rsi_oversold", "parameters": {{"threshold": 30}}, "confidence": 0.8}}
            ],
            "exit_conditions": [
                {{"condition": "profit_target", "parameters": {{"threshold": 0.05}}, "confidence": 0.9}}
            ],
            "risk_management": {{
                "max_position_size": 0.1,
                "stop_loss": 0.02,
                "take_profit": 0.05
            }}
        }}
        """
    
    def _parse_strategy_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse AI response into strategy format"""
        try:
            # Try to extract JSON from response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                strategy = json.loads(json_str)
                
                # Add metadata
                strategy['generated_at'] = datetime.now().isoformat()
                strategy['model_used'] = 'Local-Ollama'
                
                return strategy
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to parse AI response: {e}")
            return None
    
    def _generate_fallback_strategy(self, symbol: str) -> Dict[str, Any]:
        """Generate fallback strategy when AI fails"""
        return {
            'strategy_name': f'Fallback Strategy for {symbol}',
            'description': 'Simple RSI-based strategy generated as fallback',
            'entry_conditions': [
                {'condition': 'rsi_oversold', 'parameters': {'threshold': 30}, 'confidence': 0.7},
                {'condition': 'volume_spike', 'parameters': {'threshold': 1.5}, 'confidence': 0.6}
            ],
            'exit_conditions': [
                {'condition': 'profit_target', 'parameters': {'threshold': 0.03}, 'confidence': 0.8},
                {'condition': 'stop_loss', 'parameters': {'threshold': -0.02}, 'confidence': 1.0}
            ],
            'risk_management': {
                'max_position_size': 0.1,
                'stop_loss': 0.02,
                'take_profit': 0.03,
                'max_holding_days': 5
            },
            'technical_indicators': ['RSI', 'Volume', 'MA'],
            'market_conditions': ['oversold', 'normal_volume'],
            'confidence_score': 0.6,
            'expected_win_rate': 0.55,
            'model_used': 'Fallback',
            'generated_at': datetime.now().isoformat()
        }
    
    def optimize_strategy(self, strategy: Dict[str, Any], backtest_results: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize strategy based on backtest results"""
        try:
            # Create optimization prompt
            prompt = f"""
            Optimize the following trading strategy based on backtest results:
            
            Original Strategy: {json.dumps(strategy, indent=2)}
            
            Backtest Results:
            - Total Return: {backtest_results.get('total_return', 0)}%
            - Win Rate: {backtest_results.get('win_rate', 0)}%
            - Max Drawdown: {backtest_results.get('max_drawdown', 0)}%
            - Total Trades: {backtest_results.get('total_trades', 0)}
            
            Please provide an optimized strategy in the same JSON format.
            """
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": "llama3.1:8b",
                    "prompt": prompt,
                    "stream": False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                optimized = self._parse_strategy_response(result.get('response', ''))
                if optimized:
                    return optimized
            
            # Fallback: simple parameter adjustment
            return self._simple_optimization(strategy, backtest_results)
            
        except Exception as e:
            self.logger.error(f"Strategy optimization failed: {e}")
            return self._simple_optimization(strategy, backtest_results)
    
    def _simple_optimization(self, strategy: Dict[str, Any], backtest_results: Dict[str, Any]) -> Dict[str, Any]:
        """Simple rule-based optimization"""
        optimized = strategy.copy()
        
        # Adjust based on performance
        total_return = backtest_results.get('total_return', 0)
        win_rate = backtest_results.get('win_rate', 0)
        
        if total_return < 0:
            # Tighten stop loss if losing money
            if 'risk_management' in optimized:
                current_stop = optimized['risk_management'].get('stop_loss', 0.02)
                optimized['risk_management']['stop_loss'] = max(current_stop * 0.8, 0.01)
        
        if win_rate < 0.4:
            # Adjust entry conditions to be more selective
            for condition in optimized.get('entry_conditions', []):
                if 'confidence' in condition:
                    condition['confidence'] = min(condition['confidence'] * 1.1, 1.0)
        
        optimized['optimized_at'] = datetime.now().isoformat()
        optimized['optimization_reason'] = f"Adjusted based on {total_return:.2f}% return and {win_rate:.1f}% win rate"
        
        return optimized