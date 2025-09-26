"""
Google AI Model for Trading Strategy Generation
"""

import json
import requests
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

class GoogleAIModel:
    """Google AI model for strategy generation using Gemini API"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
        self.logger = logging.getLogger(__name__)
        
    def generate_strategy(self, symbol: str, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate trading strategy using Google AI model"""
        try:
            # Create prompt for strategy generation
            prompt = self._create_strategy_prompt(symbol, market_data)
            
            # Call Google AI API
            response = requests.post(
                f"{self.base_url}?key={self.api_key}",
                json={
                    "contents": [{
                        "parts": [{"text": prompt}]
                    }]
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'candidates' in result and len(result['candidates']) > 0:
                    text_response = result['candidates'][0]['content']['parts'][0]['text']
                    return self._parse_strategy_response(text_response)
            else:
                self.logger.error(f"Google AI API error: {response.status_code}")
                return self._generate_fallback_strategy(symbol)
                
        except Exception as e:
            self.logger.error(f"Google AI generation failed: {e}")
            return self._generate_fallback_strategy(symbol)
    
    def _create_strategy_prompt(self, symbol: str, market_data: Dict[str, Any]) -> str:
        """Create prompt for AI strategy generation"""
        return f"""
        Generate a comprehensive trading strategy for {symbol} based on the following market data:
        
        Current Price: ${market_data.get('current_price', 'N/A')}
        RSI: {market_data.get('rsi', 'N/A')}
        Volume Ratio: {market_data.get('volume_ratio', 'N/A')}
        Market Trend: {market_data.get('trend', 'N/A')}
        
        Please provide a detailed JSON response with the following structure:
        {{
            "strategy_name": "Strategy Name",
            "description": "Detailed strategy description",
            "entry_conditions": [
                {{"condition": "rsi_oversold", "parameters": {{"threshold": 30}}, "confidence": 0.8}}
            ],
            "exit_conditions": [
                {{"condition": "profit_target", "parameters": {{"threshold": 0.05}}, "confidence": 0.9}}
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
        
        Focus on creating a strategy that:
        1. Has clear entry and exit rules
        2. Includes proper risk management
        3. Is suitable for the current market conditions
        4. Has realistic profit targets
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
                strategy['model_used'] = 'Google-Gemini'
                
                return strategy
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to parse Google AI response: {e}")
            return None
    
    def _generate_fallback_strategy(self, symbol: str) -> Dict[str, Any]:
        """Generate fallback strategy when AI fails"""
        return {
            'strategy_name': f'Google AI Fallback Strategy for {symbol}',
            'description': 'Conservative RSI-based strategy generated as fallback',
            'entry_conditions': [
                {'condition': 'rsi_oversold', 'parameters': {'threshold': 25}, 'confidence': 0.8},
                {'condition': 'volume_confirmation', 'parameters': {'threshold': 1.2}, 'confidence': 0.7}
            ],
            'exit_conditions': [
                {'condition': 'profit_target', 'parameters': {'threshold': 0.04}, 'confidence': 0.9},
                {'condition': 'stop_loss', 'parameters': {'threshold': -0.015}, 'confidence': 1.0}
            ],
            'risk_management': {
                'max_position_size': 0.08,
                'stop_loss': 0.015,
                'take_profit': 0.04,
                'max_holding_days': 7
            },
            'technical_indicators': ['RSI', 'Volume', 'MA', 'MACD'],
            'market_conditions': ['oversold', 'volume_confirmation'],
            'confidence_score': 0.75,
            'expected_win_rate': 0.65,
            'model_used': 'Google-Fallback',
            'generated_at': datetime.now().isoformat()
        }
    
    def optimize_strategy(self, strategy: Dict[str, Any], backtest_results: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize strategy based on backtest results using Google AI"""
        try:
            # Create optimization prompt
            prompt = f"""
            Optimize the following trading strategy based on backtest performance:
            
            Original Strategy: {json.dumps(strategy, indent=2)}
            
            Backtest Performance:
            - Total Return: {backtest_results.get('total_return', 0)}%
            - Win Rate: {backtest_results.get('win_rate', 0)}%
            - Max Drawdown: {backtest_results.get('max_drawdown', 0)}%
            - Sharpe Ratio: {backtest_results.get('sharpe_ratio', 0)}
            - Total Trades: {backtest_results.get('total_trades', 0)}
            
            Please analyze the performance and provide an optimized strategy that:
            1. Improves profitability if return is negative
            2. Reduces drawdown if it's too high (>15%)
            3. Improves win rate if it's low (<50%)
            4. Maintains or improves risk-adjusted returns
            
            Return the optimized strategy in the same JSON format with explanations for changes made.
            """
            
            response = requests.post(
                f"{self.base_url}?key={self.api_key}",
                json={
                    "contents": [{
                        "parts": [{"text": prompt}]
                    }]
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'candidates' in result and len(result['candidates']) > 0:
                    text_response = result['candidates'][0]['content']['parts'][0]['text']
                    optimized = self._parse_strategy_response(text_response)
                    if optimized:
                        return optimized
            
            # Fallback: rule-based optimization
            return self._rule_based_optimization(strategy, backtest_results)
            
        except Exception as e:
            self.logger.error(f"Google AI optimization failed: {e}")
            return self._rule_based_optimization(strategy, backtest_results)
    
    def _rule_based_optimization(self, strategy: Dict[str, Any], backtest_results: Dict[str, Any]) -> Dict[str, Any]:
        """Rule-based optimization as fallback"""
        optimized = strategy.copy()
        
        total_return = backtest_results.get('total_return', 0)
        win_rate = backtest_results.get('win_rate', 0)
        max_drawdown = backtest_results.get('max_drawdown', 0)
        
        optimization_notes = []
        
        # Adjust based on performance
        if total_return < 0:
            # Tighten entry conditions
            for condition in optimized.get('entry_conditions', []):
                if 'confidence' in condition:
                    condition['confidence'] = min(condition['confidence'] * 1.15, 1.0)
            optimization_notes.append("Tightened entry conditions due to negative returns")
        
        if win_rate < 50:
            # More conservative risk management
            if 'risk_management' in optimized:
                rm = optimized['risk_management']
                rm['stop_loss'] = max(rm.get('stop_loss', 0.02) * 0.8, 0.01)
                rm['take_profit'] = min(rm.get('take_profit', 0.05) * 1.2, 0.1)
            optimization_notes.append("Adjusted risk management for better win rate")
        
        if max_drawdown > 15:
            # Reduce position size
            if 'risk_management' in optimized:
                current_size = optimized['risk_management'].get('max_position_size', 0.1)
                optimized['risk_management']['max_position_size'] = max(current_size * 0.7, 0.05)
            optimization_notes.append("Reduced position size to limit drawdown")
        
        optimized['optimized_at'] = datetime.now().isoformat()
        optimized['optimization_notes'] = optimization_notes
        optimized['model_used'] = 'Google-Rule-Based-Optimization'
        
        return optimized