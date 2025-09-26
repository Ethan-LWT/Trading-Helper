#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Strategy Generator
Uses AI models to generate trading strategies based on market analysis
"""

import os
import sys
import json
import requests
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from data.data_fetcher import get_daily_adjusted
import google.generativeai as genai
from typing import Dict, List, Optional, Any

# Try to import scrapers, create fallback if not available
try:
    from scrapers.stock_scraper import StockDataScraper
except ImportError:
    # Create a simple fallback scraper
    class StockDataScraper:
        def get_stock_data_with_fallback(self, symbol: str, period: str = "1y"):
            try:
                import yfinance as yf
                stock = yf.Ticker(symbol)
                return stock.history(period=period)
            except:
                return None

class AIStrategyGenerator:
    def __init__(self, google_api_key: str = None, ollama_url: str = "http://localhost:11434"):
        """
        Initialize AI Strategy Generator with support for Google AI and Ollama
        
        Args:
            google_api_key: Google AI API key
            ollama_url: Ollama server URL (default: http://localhost:11434)
        """
        self.google_api_key = google_api_key
        self.ollama_url = ollama_url
        self.logger = logging.getLogger(__name__)
        
        # Configure Google AI if API key is provided
        if self.google_api_key:
            genai.configure(api_key=self.google_api_key)
            self.google_model = genai.GenerativeModel('gemini-pro')
        else:
            self.google_model = None
            
    def get_stock_data_summary(self, symbol: str, days: int = 30) -> Dict[str, Any]:
        """
        Get comprehensive stock data summary for AI analysis
        
        Args:
            symbol: Stock symbol
            days: Number of days to analyze
            
        Returns:
            Dictionary containing stock data summary
        """
        try:
            data = get_daily_adjusted(symbol)
            if 'Time Series (Daily)' not in data:
                return None
                
            time_series = data['Time Series (Daily)']
            dates = list(time_series.keys())[:days]
            
            # Extract price data
            prices = []
            volumes = []
            highs = []
            lows = []
            
            for date in dates:
                day_data = time_series[date]
                prices.append(float(day_data['4. close']))
                volumes.append(int(day_data['6. volume']))
                highs.append(float(day_data['2. high']))
                lows.append(float(day_data['3. low']))
            
            # Calculate technical indicators
            prices_series = pd.Series(prices[::-1])  # Reverse for chronological order
            
            # Moving averages
            ma_5 = prices_series.rolling(5).mean().iloc[-1] if len(prices) >= 5 else None
            ma_20 = prices_series.rolling(20).mean().iloc[-1] if len(prices) >= 20 else None
            ma_50 = prices_series.rolling(50).mean().iloc[-1] if len(prices) >= 50 else None
            
            # RSI calculation
            delta = prices_series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1] if len(rsi) > 0 else None
            
            # Price changes
            current_price = prices[0]
            price_change_1d = ((prices[0] - prices[1]) / prices[1] * 100) if len(prices) > 1 else 0
            price_change_5d = ((prices[0] - prices[4]) / prices[4] * 100) if len(prices) > 4 else 0
            price_change_20d = ((prices[0] - prices[19]) / prices[19] * 100) if len(prices) > 19 else 0
            
            # Volatility
            volatility = np.std(prices[:20]) if len(prices) >= 20 else np.std(prices)
            
            # Volume analysis
            avg_volume = np.mean(volumes[:20]) if len(volumes) >= 20 else np.mean(volumes)
            volume_ratio = volumes[0] / avg_volume if avg_volume > 0 else 1
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'price_changes': {
                    '1_day': price_change_1d,
                    '5_day': price_change_5d,
                    '20_day': price_change_20d
                },
                'technical_indicators': {
                    'ma_5': ma_5,
                    'ma_20': ma_20,
                    'ma_50': ma_50,
                    'rsi': current_rsi
                },
                'volatility': volatility,
                'volume_analysis': {
                    'current_volume': volumes[0],
                    'avg_volume': avg_volume,
                    'volume_ratio': volume_ratio
                },
                'price_range': {
                    'high_20d': max(highs[:20]) if len(highs) >= 20 else max(highs),
                    'low_20d': min(lows[:20]) if len(lows) >= 20 else min(lows)
                }
            }
            
        except Exception as e:
            print(f"Error getting stock data for {symbol}: {e}")
            return None
    
    def create_enhanced_strategy_prompt(self, stock_data: Dict[str, Any], previous_results: Optional[Dict[str, Any]] = None) -> str:
        """
        Create enhanced strategy generation prompt with optimization feedback
        
        Args:
            stock_data: Stock data summary
            previous_results: Previous backtest results for optimization
            
        Returns:
            Enhanced prompt string
        """
        base_analysis = f"""
STOCK ANALYSIS FOR {stock_data['symbol']}:
Current Price: ${stock_data['current_price']:.2f}
Price Changes: 1D: {stock_data['price_changes']['1_day']:.2f}%, 5D: {stock_data['price_changes']['5_day']:.2f}%, 20D: {stock_data['price_changes']['20_day']:.2f}%

TECHNICAL INDICATORS:
- RSI: {stock_data['technical_indicators']['rsi']:.1f} ({'Oversold' if stock_data['technical_indicators']['rsi'] < 30 else 'Overbought' if stock_data['technical_indicators']['rsi'] > 70 else 'Neutral'})
- MA5: ${stock_data['technical_indicators']['ma_5']:.2f}, MA20: ${stock_data['technical_indicators']['ma_20']:.2f}
- Volatility: {stock_data['volatility']:.2f}%
- Volume Ratio: {stock_data['volume_analysis']['volume_ratio']:.2f}x
"""

        # Add previous results analysis if available
        optimization_guidance = ""
        if previous_results:
            optimization_guidance = f"""
PREVIOUS STRATEGY PERFORMANCE ANALYSIS:
- Total Return: {previous_results.get('total_return', 0):.2f}%
- Win Rate: {previous_results.get('win_rate', 0):.1f}%
- Max Drawdown: {previous_results.get('max_drawdown', 0):.2f}%
- Sharpe Ratio: {previous_results.get('sharpe_ratio', 0):.3f}
- Total Trades: {previous_results.get('total_trades', 0)}
- Daily Trading Frequency: {previous_results.get('daily_trading_frequency', 0):.1f}%

OPTIMIZATION REQUIREMENTS BASED ON PREVIOUS RESULTS:
"""
            
            # Specific optimization guidance based on results
            if previous_results.get('total_return', 0) < 0:
                optimization_guidance += """
1. IMPROVE ENTRY TIMING: Previous strategy had negative returns
   - Tighten entry conditions to reduce false signals
   - Add multiple confirmation indicators
   - Increase minimum confidence threshold to 0.8+
   
2. ENHANCE RISK MANAGEMENT:
   - Reduce position sizes to 2-3% of portfolio
   - Implement tighter stop losses (1-1.5%)
   - Add trailing stops for profit protection
"""
            
            if previous_results.get('win_rate', 0) < 50:
                optimization_guidance += """
3. IMPROVE WIN RATE:
   - Focus on high-probability setups only
   - Add trend confirmation (only trade with trend)
   - Implement multi-timeframe analysis
   - Require volume confirmation for all entries
"""
            
            if previous_results.get('max_drawdown', 0) > 3:
                optimization_guidance += """
4. REDUCE DRAWDOWN:
   - Implement position sizing based on recent performance
   - Add maximum consecutive loss limits
   - Reduce correlation between trades
"""

        enhanced_prompt = base_analysis + optimization_guidance + """

ENHANCED STRATEGY GENERATION RULES:

MANDATORY HIGH-PROBABILITY ENTRY CONDITIONS (Must include ALL):
1. TREND CONFIRMATION: Only trade in direction of 20-day MA trend
2. MOMENTUM FILTER: RSI between 40-60 for entries (avoid extremes)
3. VOLUME CONFIRMATION: Current volume > 1.2x average volume
4. PRICE ACTION: Clear breakout or bounce pattern
5. MULTIPLE TIMEFRAME ALIGNMENT: 5-min and 1-hour trends aligned

STRICT EXIT CONDITIONS (Implement ALL):
1. PROFIT TARGET: 2-3% gain with partial profit taking at 1.5%
2. STOP LOSS: Maximum 1% loss per trade
3. TRAILING STOP: Activate after 1% profit, trail by 0.5%
4. TIME STOP: Exit after 2 days if no movement
5. TECHNICAL EXIT: Exit on trend reversal signals

ENHANCED RISK MANAGEMENT:
- Position Size: Maximum 2% of portfolio per trade
- Daily Loss Limit: 0.5% of portfolio
- Maximum Trades: 2 per day to avoid overtrading
- Correlation Check: No more than 1 position in same sector

PROFITABILITY OPTIMIZATION RULES:
1. QUALITY OVER QUANTITY: Fewer, higher-quality trades
2. ASYMMETRIC RISK-REWARD: Minimum 2:1 reward-to-risk ratio
3. MARKET CONDITION ADAPTATION: Different rules for trending vs ranging markets
4. VOLATILITY ADJUSTMENT: Adjust position sizes based on current volatility

TARGET PERFORMANCE METRICS:
- Win Rate: 60-75% (higher than previous)
- Average Return per Trade: 1.5-2%
- Maximum Drawdown: <2%
- Sharpe Ratio: >1.0
- Daily Trading Frequency: 10-20% (selective trading)

Generate a JSON strategy with these enhanced requirements:
{
    "strategy_name": "Enhanced High-Probability Strategy",
    "description": "Optimized strategy focusing on quality over quantity",
    "optimization_notes": "Improvements made based on previous performance",
    "entry_conditions": [
        {
            "condition": "Detailed condition with strict parameters",
            "parameters": {
                "indicator": "specific_indicator",
                "threshold": "conservative_value",
                "confirmation": "multiple_confirmations"
            },
            "confidence": 0.85,
            "expected_frequency": "1-2 per day"
        }
    ],
    "exit_conditions": [
        {
            "condition": "Conservative profit target",
            "parameters": {
                "target_pct": 0.02,
                "stop_pct": 0.01,
                "trailing_pct": 0.005
            },
            "type": "profit_target",
            "priority": 1
        }
    ],
    "risk_management": {
        "max_position_size": 0.02,
        "stop_loss_pct": 0.01,
        "take_profit_pct": 0.025,
        "max_daily_trades": 2,
        "max_daily_loss": 0.005,
        "position_sizing_method": "volatility_adjusted",
        "trailing_stop_pct": 0.005
    },
    "signal_frequency": "1-2 per day",
    "expected_win_rate": 0.70,
    "risk_reward_ratio": 2.5,
    "holding_period": "intraday to 1 day",
    "market_conditions": "trending markets with clear direction"
}

CRITICAL: Focus on CONSERVATIVE, HIGH-PROBABILITY trades. Better to miss opportunities than take losses.
"""
        return enhanced_prompt

    def create_strategy_prompt(self, stock_data: Dict[str, Any]) -> str:
        """
        Create a comprehensive prompt for AI strategy generation with enhanced conditions
        
        Args:
            stock_data: Dictionary containing stock analysis data
            
        Returns:
            Formatted prompt string for AI model
        """
        return self.create_enhanced_strategy_prompt(stock_data)
        prompt = f"""
You are an expert quantitative trading strategist with 15+ years of experience in algorithmic trading. Based on the following comprehensive stock analysis, generate a highly specific, actionable trading strategy that can produce consistent daily signals.

COMPREHENSIVE STOCK ANALYSIS FOR {stock_data['symbol']}:

PRICE ACTION:
- Current Price: ${stock_data['current_price']:.2f}
- Price Changes: 1D: {stock_data['price_changes']['1_day']:.2f}%, 5D: {stock_data['price_changes']['5_day']:.2f}%, 20D: {stock_data['price_changes']['20_day']:.2f}%
- 20-day Range: ${stock_data['price_range']['low_20d']:.2f} - ${stock_data['price_range']['high_20d']:.2f}
- Price Position in Range: {((stock_data['current_price'] - stock_data['price_range']['low_20d']) / (stock_data['price_range']['high_20d'] - stock_data['price_range']['low_20d']) * 100):.1f}%

TECHNICAL INDICATORS:
- Moving Averages:
  * MA5: {stock_data['technical_indicators']['ma_5']:.2f if stock_data['technical_indicators']['ma_5'] else 'N/A'}
  * MA20: {stock_data['technical_indicators']['ma_20']:.2f if stock_data['technical_indicators']['ma_20'] else 'N/A'}
  * MA50: {stock_data['technical_indicators']['ma_50']:.2f if stock_data['technical_indicators']['ma_50'] else 'N/A'}
- RSI (14): {stock_data['technical_indicators']['rsi']:.2f if stock_data['technical_indicators']['rsi'] else 'N/A'}
- Volatility (20D): {stock_data['volatility']:.2f}

VOLUME ANALYSIS:
- Current Volume vs Average: {stock_data['volume_analysis']['volume_ratio']:.2f}x
- Volume Trend: {'High' if stock_data['volume_analysis']['volume_ratio'] > 1.5 else 'Normal' if stock_data['volume_analysis']['volume_ratio'] > 0.8 else 'Low'}

MARKET CONTEXT:
- Trend Direction: {'Bullish' if stock_data['technical_indicators']['ma_5'] and stock_data['technical_indicators']['ma_20'] and stock_data['technical_indicators']['ma_5'] > stock_data['technical_indicators']['ma_20'] else 'Bearish' if stock_data['technical_indicators']['ma_5'] and stock_data['technical_indicators']['ma_20'] and stock_data['technical_indicators']['ma_5'] < stock_data['technical_indicators']['ma_20'] else 'Sideways'}
- Momentum: {'Strong' if abs(stock_data['price_changes']['5_day']) > 3 else 'Moderate' if abs(stock_data['price_changes']['5_day']) > 1 else 'Weak'}
- Volatility Level: {'High' if stock_data['volatility'] > stock_data['current_price'] * 0.03 else 'Normal'}

STRATEGY REQUIREMENTS:
1. DAILY SIGNAL GENERATION: Must produce at least 1-3 actionable signals per day
2. CLEAR ENTRY CONDITIONS: Specific, measurable criteria for opening positions
3. PRECISE EXIT CONDITIONS: Both profit-taking and stop-loss rules
4. RISK MANAGEMENT: Position sizing, maximum exposure, daily loss limits
5. BACKTESTABLE LOGIC: All conditions must be programmable and testable

MANDATORY STRATEGY COMPONENTS:

ENTRY SIGNAL TYPES (Include at least 3):
- Momentum Breakouts: Price breaking above/below key levels with volume confirmation
- Mean Reversion: RSI oversold/overbought with price divergence
- Trend Following: Moving average crossovers with momentum confirmation
- Volume Spikes: Unusual volume with price movement
- Support/Resistance: Bounces or breaks at key technical levels

EXIT SIGNAL TYPES (Include all):
- Profit Targets: Specific percentage or technical level targets
- Stop Losses: Maximum acceptable loss per trade
- Time-based Exits: Maximum holding period
- Technical Exits: Indicator-based exit signals

RISK MANAGEMENT RULES (Mandatory):
- Position Size: Based on volatility and account risk
- Daily Loss Limit: Maximum loss per day
- Maximum Concurrent Positions: Risk diversification
- Correlation Limits: Avoid over-concentration

PERFORMANCE TARGETS:
- Win Rate: 55-70%
- Risk-Reward Ratio: Minimum 1:1.5
- Maximum Drawdown: <5%
- Daily Trading Frequency: 1-3 signals

Please respond with a detailed JSON object:
{{
    "strategy_name": "Descriptive strategy name",
    "description": "Comprehensive strategy explanation (2-3 sentences)",
    "market_analysis": "Current market condition assessment",
    "entry_conditions": [
        {{
            "condition": "Detailed entry condition with specific parameters",
            "parameters": {{
                "indicator": "specific_indicator",
                "threshold": numeric_value,
                "confirmation": "additional_confirmation_required"
            }},
            "confidence": 0.75,
            "expected_frequency": "signals_per_day"
        }}
    ],
    "exit_conditions": [
        {{
            "condition": "Specific exit condition",
            "parameters": {{
                "target_pct": 0.03,
                "stop_pct": 0.015
            }},
            "type": "profit_target",
            "priority": 1
        }}
    ],
    "risk_management": {{
        "max_position_size": 0.05,
        "stop_loss_pct": 0.02,
        "take_profit_pct": 0.04,
        "max_daily_trades": 3,
        "max_daily_loss": 0.01,
        "position_sizing_method": "volatility_based"
    }},
    "signal_frequency": "1-3 per day",
    "expected_win_rate": 0.65,
    "risk_reward_ratio": 1.8,
    "holding_period": "intraday to 2 days",
    "market_conditions": "optimal market conditions for this strategy"
}}

CRITICAL: Ensure all numerical parameters are realistic and backtestable. The strategy must be specific enough to generate consistent daily signals while maintaining proper risk management.
"""
        return prompt
    
    def generate_strategy_with_google_ai(self, stock_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Generate trading strategy using Google AI
        
        Args:
            stock_data: Stock data summary
            
        Returns:
            Generated strategy as dictionary
        """
        if not self.google_model:
            return None
            
        try:
            prompt = self.create_strategy_prompt(stock_data)
            response = self.google_model.generate_content(prompt)
            
            # Extract JSON from response
            response_text = response.text
            
            # Find JSON in the response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx]
                strategy = json.loads(json_str)
                strategy['model_used'] = 'Google AI'
                strategy['generated_at'] = datetime.now().isoformat()
                return strategy
            else:
                print("Could not extract JSON from Google AI response")
                return None
                
        except Exception as e:
            print(f"Error generating strategy with Google AI: {e}")
            return None
    
    def generate_strategy_with_ollama(self, stock_data: Dict[str, Any], model_name: str = "llama3.1:8b", 
                                    previous_results: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Generate trading strategy using Ollama local AI model with optimization feedback
        
        Args:
            stock_data: Stock analysis data
            model_name: Ollama model name
            previous_results: Previous backtest results for optimization
            
        Returns:
            Generated strategy dictionary or None if failed
        """
        try:
            # Use enhanced prompt with previous results
            prompt = self.create_enhanced_strategy_prompt(stock_data, previous_results)
            
            # Prepare request for Ollama API
            ollama_request = {
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,  # Lower temperature for more consistent results
                    "top_p": 0.9,
                    "num_predict": 2000
                }
            }
            
            # Make request to Ollama
            response = requests.post(
                "http://localhost:11434/api/generate",
                json=ollama_request,
                timeout=120
            )
            
            if response.status_code == 200:
                response_data = response.json()
                strategy_text = response_data.get("response", "")
                
                strategy = self._extract_json_from_response(strategy_text)
                if strategy:
                    # Add metadata
                    strategy["model_used"] = f"ollama:{model_name}"
                    strategy["generation_timestamp"] = datetime.now().isoformat()
                    if previous_results:
                        strategy["optimization_feedback"] = True
                        strategy["previous_return"] = previous_results.get("total_return", 0)
                    
                    self.logger.info(f"Successfully generated optimized strategy using Ollama {model_name}")
                    return strategy
                else:
                    self.logger.error("Failed to extract valid JSON from Ollama response")
                    return None
            else:
                self.logger.error(f"Ollama API request failed: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Error generating strategy with Ollama: {e}")
            return None
    
    def _extract_json_from_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """
        Extract JSON from AI model response text
        
        Args:
            response_text: Raw response text from AI model
            
        Returns:
            Parsed JSON dictionary or None if extraction failed
        """
        try:
            # Find JSON in the response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx]
                strategy = json.loads(json_str)
                return strategy
            else:
                # Try to find JSON array format
                start_idx = response_text.find('[')
                end_idx = response_text.rfind(']') + 1
                
                if start_idx != -1 and end_idx != -1:
                    json_str = response_text[start_idx:end_idx]
                    data = json.loads(json_str)
                    if isinstance(data, list) and len(data) > 0:
                        return data[0]  # Return first strategy if array
                
                return None
                
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error extracting JSON: {e}")
            return None
    
    def generate_strategy(self, symbol: str, model_type: str = "google", ollama_model: str = "llama3.1:8b") -> Optional[Dict[str, Any]]:
        """
        Generate trading strategy for a given symbol
        
        Args:
            symbol: Stock symbol
            model_type: "google" or "ollama"
            ollama_model: Ollama model name if using Ollama
            
        Returns:
            Generated strategy dictionary
        """
        try:
            # Get stock data
            stock_data = self.get_stock_data_summary(symbol)
            if not stock_data:
                print(f"Failed to get stock data for {symbol}")
                return self._generate_mock_strategy(symbol, model_type)
            
            # Generate strategy based on model type
            strategy = None
            if model_type.lower() == "google" and self.google_model:
                strategy = self.generate_strategy_with_google_ai(stock_data)
            elif model_type.lower() == "ollama":
                strategy = self.generate_strategy_with_ollama(stock_data, ollama_model)
            else:
                print(f"Invalid model type or model not available: {model_type}")
                return self._generate_mock_strategy(symbol, model_type)
            
            # If AI generation fails, return mock strategy
            if strategy is None:
                print(f"AI generation failed for {model_type}, returning mock strategy")
                return self._generate_mock_strategy(symbol, model_type)
                
            # Add metadata
            strategy['symbol'] = symbol
            strategy['generated_at'] = datetime.now().isoformat()
            
            return strategy
            
        except Exception as e:
            print(f"Error generating strategy: {str(e)}")
            return self._generate_mock_strategy(symbol, model_type)
    
    def _generate_mock_strategy(self, symbol: str, model_type: str) -> Dict[str, Any]:
        """Generate a mock strategy for testing purposes"""
        return {
            "strategy_name": f"AI Mock Strategy for {symbol}",
            "description": f"Mock trading strategy generated for {symbol} using {model_type} model (fallback)",
            "symbol": symbol,
            "model_used": f"{model_type} (mock)",
            "entry_conditions": [
                {
                    "condition": "RSI below 30 (oversold)",
                    "parameters": {
                        "indicator": "RSI",
                        "threshold": 30,
                        "confirmation": "volume_spike"
                    },
                    "confidence": 0.75,
                    "expected_frequency": "2-3 signals per week"
                },
                {
                    "condition": "Price above 20-day MA with momentum",
                    "parameters": {
                        "indicator": "MA_20",
                        "threshold": 1.02,
                        "confirmation": "increasing_volume"
                    },
                    "confidence": 0.68,
                    "expected_frequency": "1-2 signals per day"
                }
            ],
            "exit_conditions": [
                {
                    "condition": "Take profit at 3% gain",
                    "parameters": {
                        "target_pct": 0.03,
                        "stop_pct": 0.015
                    },
                    "type": "profit_target",
                    "priority": 1
                },
                {
                    "condition": "Stop loss at 1.5% loss",
                    "parameters": {
                        "stop_pct": 0.015
                    },
                    "type": "stop_loss",
                    "priority": 2
                }
            ],
            "risk_management": {
                "max_position_size": 0.05,
                "stop_loss_pct": 0.02,
                "take_profit_pct": 0.04,
                "max_daily_trades": 3,
                "max_daily_loss": 0.01,
                "position_sizing_method": "volatility_based"
            },
            "signal_frequency": "1-3 per day",
            "expected_win_rate": 0.65,
            "risk_reward_ratio": 2.0,
            "holding_period": "intraday to 2 days",
            "market_conditions": "suitable for trending and volatile markets",
            "generated_at": datetime.now().isoformat()
        }
    
    def validate_strategy(self, strategy: Dict[str, Any]) -> bool:
        """
        Validate that the generated strategy has required fields
        
        Args:
            strategy: Strategy dictionary
            
        Returns:
            True if valid, False otherwise
        """
        required_fields = [
            'strategy_name', 'description', 'entry_conditions', 
            'exit_conditions', 'risk_management'
        ]
        
        for field in required_fields:
            if field not in strategy:
                return False
        
        # Validate entry conditions
        if not isinstance(strategy['entry_conditions'], list) or len(strategy['entry_conditions']) == 0:
            return False
            
        # Validate exit conditions
        if not isinstance(strategy['exit_conditions'], list) or len(strategy['exit_conditions']) == 0:
            return False
            
        return True

# Global AI strategy generator instance
ai_strategy_generator = None

def initialize_ai_generator(google_api_key: str = None, ollama_url: str = "http://localhost:11434"):
    """Initialize the global AI strategy generator"""
    global ai_strategy_generator
    ai_strategy_generator = AIStrategyGenerator(google_api_key, ollama_url)
    return ai_strategy_generator

# Export for easy importing
__all__ = ['AIStrategyGenerator', 'ai_strategy_generator', 'initialize_ai_generator']