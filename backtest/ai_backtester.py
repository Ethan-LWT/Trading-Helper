import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from backtest.backtester import Backtester
import sys
import os

# Add strategy directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'strategy'))
from multi_timeframe_strategy import MultiTimeframeStrategy, TimeFrame, StrategySignal

class AIBacktester(Backtester):
    """
    Enhanced backtester specifically designed for AI-generated trading strategies
    """
    
    def __init__(self, initial_capital=100000):
        super().__init__(initial_capital)
        self.ai_strategy = None
        self.daily_signals = []
        self.daily_trades = []
        self.daily_pnl = []
        self.portfolio_value_history = []
        self.trade_history = []
        
        # Initialize multi-timeframe strategy
        self.multi_timeframe_strategy = MultiTimeframeStrategy(initial_capital)
        self.use_multi_timeframe = False  # Flag to enable multi-timeframe mode
        
    def load_ai_strategy(self, ai_strategy: Dict[str, Any]):
        """
        Load AI strategy configuration
        
        Args:
            ai_strategy: Dictionary containing strategy configuration
        """
        self.ai_strategy = ai_strategy
        
        # Check if multi-timeframe mode should be enabled
        if ai_strategy.get('use_multi_timeframe', False):
            self.use_multi_timeframe = True
            print(f"Loaded AI Strategy: {ai_strategy.get('strategy_name', 'Unknown')} (Multi-Timeframe Mode)")
        else:
            print(f"Loaded AI Strategy: {ai_strategy.get('strategy_name', 'Unknown')}")
    
    def enable_multi_timeframe_mode(self):
        """Enable multi-timeframe trading mode"""
        self.use_multi_timeframe = True
        print("Multi-timeframe trading mode enabled")
    
    def disable_multi_timeframe_mode(self):
        """Disable multi-timeframe trading mode"""
        self.use_multi_timeframe = False
        print("Multi-timeframe trading mode disabled")
        
    def execute_trade(self, symbol: str, action: str, shares: int, price: float, timestamp: str = None) -> bool:
        """
        Execute a trade (buy or sell)
        
        Args:
            symbol: Stock symbol
            action: 'BUY' or 'SELL'
            shares: Number of shares
            price: Price per share
            timestamp: Timestamp of the trade
            
        Returns:
            bool: True if trade was successful, False otherwise
        """
        try:
            if action.upper() == 'BUY':
                return self.buy(symbol, shares, price, timestamp)
            elif action.upper() == 'SELL':
                return self.sell(symbol, shares, price, timestamp)
            else:
                print(f"Invalid trade action: {action}")
                return False
        except Exception as e:
            print(f"Error executing trade: {e}")
            return False
    
    def get_mock_data(self, symbol: str, days: int = 252) -> pd.DataFrame:
        """
        Generate mock stock data for testing when real data is not available
        
        Args:
            symbol: Stock symbol
            days: Number of days of data to generate
            
        Returns:
            DataFrame with mock stock data
        """
        np.random.seed(42)  # For reproducible results
        
        # Generate dates
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        dates = dates[dates.weekday < 5][:days]  # Only weekdays
        
        # Generate realistic stock price data
        initial_price = 150.0  # Starting price
        returns = np.random.normal(0.0005, 0.02, len(dates))  # Daily returns
        prices = [initial_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # Generate volumes
        base_volume = 50000000
        volumes = np.random.lognormal(np.log(base_volume), 0.3, len(dates))
        
        # Create DataFrame
        data = pd.DataFrame({
            'Date': dates,
            'Open': [p * np.random.uniform(0.99, 1.01) for p in prices],
            'High': [p * np.random.uniform(1.00, 1.03) for p in prices],
            'Low': [p * np.random.uniform(0.97, 1.00) for p in prices],
            'Close': prices,
            'Volume': volumes.astype(int)
        })
        
        data.set_index('Date', inplace=True)
        return data
        
    def calculate_technical_indicators(self, price_data: List[float], volumes: List[int] = None) -> Dict[str, float]:
        """
        Calculate technical indicators needed for AI strategy evaluation
        
        Args:
            price_data: List of closing prices (most recent first)
            volumes: List of volumes (most recent first)
            
        Returns:
            Dictionary of calculated indicators
        """
        if len(price_data) < 2:
            return {}
            
        prices = np.array(price_data[::-1])  # Reverse for chronological order
        
        indicators = {}
        
        # Current price
        indicators['current_price'] = prices[-1]
        
        # Moving averages
        if len(prices) >= 5:
            indicators['ma_5'] = np.mean(prices[-5:])
        if len(prices) >= 10:
            indicators['ma_10'] = np.mean(prices[-10:])
        if len(prices) >= 20:
            indicators['ma_20'] = np.mean(prices[-20:])
        if len(prices) >= 50:
            indicators['ma_50'] = np.mean(prices[-50:])
            
        # RSI calculation
        if len(prices) >= 15:
            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gain = np.mean(gains[-14:])
            avg_loss = np.mean(losses[-14:])
            
            if avg_loss != 0:
                rs = avg_gain / avg_loss
                indicators['rsi'] = 100 - (100 / (1 + rs))
            else:
                indicators['rsi'] = 100
                
        # Price momentum
        if len(prices) >= 2:
            indicators['price_change_1d'] = (prices[-1] - prices[-2]) / prices[-2] * 100
        if len(prices) >= 6:
            indicators['price_change_5d'] = (prices[-1] - prices[-6]) / prices[-6] * 100
            
        # Price momentum (20-day)
        if len(prices) >= 21:
            indicators['price_change_20d'] = (prices[-1] - prices[-21]) / prices[-21] * 100
            
        # Volatility
        if len(prices) >= 20:
            indicators['volatility'] = np.std(prices[-20:])
        else:
            indicators['volatility'] = np.std(prices)
            
        # Volume indicators
        if volumes and len(volumes) >= 2:
            volumes_array = np.array(volumes[::-1])
            indicators['volume_change_1d'] = (volumes_array[-1] - volumes_array[-2]) / volumes_array[-2] * 100
            if len(volumes_array) >= 5:
                indicators['avg_volume_5d'] = np.mean(volumes_array[-5:])
            
            # Volume ratio
            if len(volumes_array) >= 20:
                indicators['volume_ratio'] = volumes_array[-1] / np.mean(volumes_array[-20:])
            else:
                indicators['volume_ratio'] = volumes_array[-1] / np.mean(volumes_array)
        else:
            indicators['volume_ratio'] = 1.0
            
        # Support and resistance levels
        if len(prices) >= 20:
            indicators['resistance_20d'] = np.max(prices[-20:])
            indicators['support_20d'] = np.min(prices[-20:])
            indicators['price_position'] = (prices[-1] - indicators['support_20d']) / (indicators['resistance_20d'] - indicators['support_20d'])
            
        return indicators
        
    def evaluate_entry_conditions(self, indicators: Dict[str, float], current_price: float, current_date: str = None) -> Dict[str, Any]:
        """
        Evaluate AI strategy entry conditions with time-based and volatility-adaptive logic
        
        Args:
            indicators: Technical indicators
            current_price: Current stock price
            current_date: Current date for time-based strategy
            
        Returns:
            Signal information if conditions are met
        """
        if not self.ai_strategy or 'entry_conditions' not in self.ai_strategy:
            return None
            
        # Get time-based strategy parameters
        time_strategy = self._get_time_based_strategy(current_date)
        volatility_factor = self._calculate_volatility_factor(indicators)
        
        signals = []
        entry_conditions = self.ai_strategy['entry_conditions']
        
        # Handle both dict and list formats
        if isinstance(entry_conditions, dict):
            conditions_list = []
            for key, value in entry_conditions.items():
                conditions_list.append({
                    'condition': key,
                    'parameters': {'threshold': value}
                })
        else:
            conditions_list = entry_conditions
        
        # Apply time-based and volatility adjustments to conditions
        adjusted_conditions = self._adjust_conditions_for_time_and_volatility(
            conditions_list, time_strategy, volatility_factor
        )
        
        for condition in adjusted_conditions:
            signal_strength = 0
            condition_met = False
            
            if isinstance(condition, dict):
                condition_text = condition.get('condition', '').lower()
                params = condition.get('parameters', {})
            else:
                condition_text = str(condition).lower()
                params = {}
            
            # RSI oversold condition with dynamic thresholds
            if 'rsi_oversold' in condition_text or condition_text == 'rsi_oversold':
                if 'rsi' in indicators:
                    threshold = params.get('threshold', 50)
                    if indicators['rsi'] < threshold:
                        condition_met = True
                        signal_strength = 0.9
                else:
                    # Use price momentum as proxy with time-adjusted sensitivity
                    price_threshold = params.get('price_threshold', -2.0)
                    if 'price_change_1d' in indicators and indicators['price_change_1d'] < price_threshold:
                        condition_met = True
                        signal_strength = 0.6
                        
            # Volume spike condition with time-based adjustments
            elif 'volume_spike' in condition_text or condition_text == 'volume_spike':
                if 'volume_ratio' in indicators:
                    threshold = params.get('threshold', 1.2)
                    if indicators['volume_ratio'] > threshold:
                        condition_met = True
                        signal_strength = 0.6
                        
            # Moving average conditions
            elif 'moving average' in condition_text or 'ma' in condition_text:
                if 'ma_5' in indicators and 'ma_20' in indicators:
                    if 'crossover' in condition_text and indicators['ma_5'] > indicators['ma_20']:
                        condition_met = True
                        signal_strength = 0.8
                    elif 'above' in condition_text and current_price > indicators.get('ma_20', 0):
                        condition_met = True
                        signal_strength = 0.7
                        
            # Breakout conditions
            elif 'breakout' in condition_text:
                if 'resistance_20d' in indicators and current_price > indicators['resistance_20d']:
                    condition_met = True
                    signal_strength = 0.8
                    
            # Time-based opportunity conditions
            elif 'time_opportunity' in condition_text:
                # This condition is always met but with varying strength based on time
                condition_met = True
                signal_strength = params.get('strength', 0.4)
                    
            # Price drop condition with dynamic thresholds
            elif 'price_dip_opportunity' in condition_text or 'price_drop' in condition_text:
                price_threshold = params.get('threshold', -1.0)
                if 'price_change_1d' in indicators and indicators['price_change_1d'] < price_threshold:
                    condition_met = True
                    signal_strength = params.get('strength', 0.5)
                    
            if condition_met:
                signals.append({
                    'condition': condition_text,
                    'strength': signal_strength,
                    'confidence': condition.get('confidence', 0.7),
                    'time_factor': time_strategy['aggressiveness'],
                    'volatility_factor': volatility_factor
                })
        
        # Always add time-based signals to ensure regular trading
        time_signals = self._generate_time_based_signals(current_date, indicators, current_price)
        signals.extend(time_signals)
                
        if signals:
            # Calculate overall signal strength with time and volatility factors
            # Ensure all values are numeric to prevent sequence multiplication errors
            total_strength = 0
            for s in signals:
                strength = float(s.get('strength', 0))
                confidence = float(s.get('confidence', 0))
                time_factor = float(s.get('time_factor', 1.0))
                total_strength += strength * confidence * time_factor
            
            avg_confidence = sum(float(s.get('confidence', 0)) for s in signals) / len(signals)
            
            # Apply volatility boost to confidence
            final_confidence = min(avg_confidence * (1 + float(volatility_factor) * 0.3), 1.0)
            
            return {
                'action': 'BUY',
                'strength': total_strength / len(signals),
                'confidence': final_confidence,
                'signals': signals,
                'time_strategy': time_strategy,
                'volatility_factor': volatility_factor,
                'timestamp': datetime.now().isoformat()
            }
            
        return None
    
    def _get_time_based_strategy(self, current_date: str = None) -> Dict[str, Any]:
        """
        Get time-based strategy parameters based on current date
        """
        if not current_date:
            current_date = datetime.now().strftime('%Y-%m-%d')
        
        try:
            date_obj = datetime.strptime(current_date, '%Y-%m-%d')
        except:
            date_obj = datetime.now()
        
        # Get day of week (0=Monday, 6=Sunday)
        day_of_week = date_obj.weekday()
        # Get day of month
        day_of_month = date_obj.day
        # Get week of month
        week_of_month = (day_of_month - 1) // 7 + 1
        
        strategy = {
            'day_of_week': day_of_week,
            'day_of_month': day_of_month,
            'week_of_month': week_of_month,
            'aggressiveness': 1.0,
            'frequency_multiplier': 1.0,
            'risk_tolerance': 1.0
        }
        
        # Monday (0) - Start of week, more conservative
        if day_of_week == 0:
            strategy.update({
                'aggressiveness': 0.8,
                'frequency_multiplier': 1.2,
                'risk_tolerance': 0.9,
                'strategy_type': 'monday_conservative'
            })
        # Tuesday-Wednesday (1-2) - Mid-week, more aggressive
        elif day_of_week in [1, 2]:
            strategy.update({
                'aggressiveness': 1.3,
                'frequency_multiplier': 1.5,
                'risk_tolerance': 1.2,
                'strategy_type': 'midweek_aggressive'
            })
        # Thursday (3) - Pre-weekend positioning
        elif day_of_week == 3:
            strategy.update({
                'aggressiveness': 1.1,
                'frequency_multiplier': 1.3,
                'risk_tolerance': 1.0,
                'strategy_type': 'thursday_positioning'
            })
        # Friday (4) - End of week, profit taking
        elif day_of_week == 4:
            strategy.update({
                'aggressiveness': 0.9,
                'frequency_multiplier': 1.1,
                'risk_tolerance': 0.8,
                'strategy_type': 'friday_profit_taking'
            })
        
        # Weekly adjustments
        if week_of_month == 1:  # First week of month
            strategy['frequency_multiplier'] *= 1.2
            strategy['aggressiveness'] *= 1.1
        elif week_of_month == 4:  # Last week of month
            strategy['frequency_multiplier'] *= 1.3
            strategy['aggressiveness'] *= 1.2
        
        # Monthly adjustments
        if day_of_month <= 5:  # Beginning of month
            strategy['frequency_multiplier'] *= 1.15
        elif day_of_month >= 25:  # End of month
            strategy['frequency_multiplier'] *= 1.25
        
        return strategy
    
    def _calculate_volatility_factor(self, indicators: Dict[str, float]) -> float:
        """
        Calculate market volatility factor to adjust strategy aggressiveness
        """
        volatility_factor = 0.0
        
        # Price volatility
        if 'volatility' in indicators:
            # Normalize volatility (assuming typical range 0-10)
            normalized_vol = min(indicators['volatility'] / 10.0, 1.0)
            volatility_factor += normalized_vol * 0.4
        
        # Price momentum volatility
        if 'price_change_1d' in indicators:
            abs_change = abs(indicators['price_change_1d'])
            # High daily changes indicate volatility
            if abs_change > 3:
                volatility_factor += 0.3
            elif abs_change > 1.5:
                volatility_factor += 0.2
        
        # Volume volatility
        if 'volume_ratio' in indicators:
            if indicators['volume_ratio'] > 2.0:
                volatility_factor += 0.3
            elif indicators['volume_ratio'] > 1.5:
                volatility_factor += 0.2
        
        return min(volatility_factor, 1.0)
    
    def _adjust_conditions_for_time_and_volatility(self, conditions: List[Dict], 
                                                 time_strategy: Dict, volatility_factor: float) -> List[Dict]:
        """
        Adjust trading conditions based on time and volatility factors
        """
        adjusted_conditions = []
        
        for condition in conditions:
            adjusted_condition = condition.copy()
            params = adjusted_condition.get('parameters', {})
            
            # Adjust thresholds based on time strategy
            if 'threshold' in params:
                original_threshold = params['threshold']
                # Make conditions more lenient during high-frequency periods
                time_adjustment = 1.0 / time_strategy['frequency_multiplier']
                params['threshold'] = original_threshold * time_adjustment
            
            # Adjust confidence based on volatility
            if 'confidence' in adjusted_condition:
                vol_boost = volatility_factor * 0.2
                adjusted_condition['confidence'] = min(adjusted_condition['confidence'] + vol_boost, 1.0)
            
            adjusted_conditions.append(adjusted_condition)
        
        return adjusted_conditions
    
    def _generate_time_based_signals(self, current_date: str, indicators: Dict[str, float], 
                                   current_price: float) -> List[Dict]:
        """
        Generate additional signals based on time patterns to ensure regular trading
        """
        signals = []
        
        if not current_date:
            return signals
        
        try:
            date_obj = datetime.strptime(current_date, '%Y-%m-%d')
        except:
            return signals
        
        day_of_week = date_obj.weekday()
        day_of_month = date_obj.day
        
        # Monday opportunity signal
        if day_of_week == 0:
            signals.append({
                'condition': 'monday_opportunity',
                'strength': 0.4,
                'confidence': 0.6,
                'time_factor': 1.2,
                'volatility_factor': 0.0
            })
        
        # Mid-week momentum signal
        if day_of_week in [1, 2]:
            if 'price_change_1d' in indicators:
                # Any price movement triggers signal
                if abs(indicators['price_change_1d']) > 0.5:
                    signals.append({
                        'condition': 'midweek_momentum',
                        'strength': 0.5,
                        'confidence': 0.7,
                        'time_factor': 1.5,
                        'volatility_factor': 0.0
                    })
        
        # End of week positioning
        if day_of_week == 3:
            signals.append({
                'condition': 'thursday_positioning',
                'strength': 0.45,
                'confidence': 0.65,
                'time_factor': 1.3,
                'volatility_factor': 0.0
            })
        
        # Monthly cycle signals
        if day_of_month <= 5 or day_of_month >= 25:
            signals.append({
                'condition': 'monthly_cycle_opportunity',
                'strength': 0.4,
                'confidence': 0.6,
                'time_factor': 1.25,
                'volatility_factor': 0.0
            })
        
        # Daily opportunity signal (ensures at least some trading activity)
        if len(signals) == 0:
            # Create a weak signal to ensure some trading activity
            base_strength = 0.3
            if 'price_change_1d' in indicators and indicators['price_change_1d'] < 0:
                base_strength = 0.4  # Slightly stronger for price drops
            
            signals.append({
                'condition': 'daily_opportunity',
                'strength': base_strength,
                'confidence': 0.5,
                'time_factor': 1.0,
                'volatility_factor': 0.0
            })
        
        return signals
            
        # Get time-based strategy parameters
        time_strategy = self._get_time_based_strategy(current_date)
        volatility_factor = self._calculate_volatility_factor(indicators)
        
        signals = []
        entry_conditions = self.ai_strategy['entry_conditions']
        
        # Handle both dict and list formats
        if isinstance(entry_conditions, dict):
            conditions_list = []
            for key, value in entry_conditions.items():
                conditions_list.append({
                    'condition': key,
                    'parameters': {'threshold': value}
                })
        else:
            conditions_list = entry_conditions
        
        # Apply time-based and volatility adjustments to conditions
        adjusted_conditions = self._adjust_conditions_for_time_and_volatility(
            conditions_list, time_strategy, volatility_factor
        )
        
        for condition in adjusted_conditions:
            signal_strength = 0
            condition_met = False
            
            if isinstance(condition, dict):
                condition_text = condition.get('condition', '').lower()
                params = condition.get('parameters', {})
            else:
                condition_text = str(condition).lower()
                params = {}
            
            # RSI oversold condition with dynamic thresholds
            if 'rsi_oversold' in condition_text or condition_text == 'rsi_oversold':
                if 'rsi' in indicators:
                    threshold = params.get('threshold', 50)
                    if indicators['rsi'] < threshold:
                        condition_met = True
                        signal_strength = 0.9
                else:
                    # Use price momentum as proxy with time-adjusted sensitivity
                    price_threshold = params.get('price_threshold', -2.0)
                    if 'price_change_1d' in indicators and indicators['price_change_1d'] < price_threshold:
                        condition_met = True
                        signal_strength = 0.6
                        
            # Volume spike condition with time-based adjustments
            elif 'volume_spike' in condition_text or condition_text == 'volume_spike':
                if 'volume_ratio' in indicators:
                    threshold = params.get('threshold', 1.2)
                    if indicators['volume_ratio'] > threshold:
                        condition_met = True
                        signal_strength = 0.6
                        
            # Moving average conditions
            elif 'moving average' in condition_text or 'ma' in condition_text:
                if 'ma_5' in indicators and 'ma_20' in indicators:
                    if 'crossover' in condition_text and indicators['ma_5'] > indicators['ma_20']:
                        condition_met = True
                        signal_strength = 0.8
                    elif 'above' in condition_text and current_price > indicators.get('ma_20', 0):
                        condition_met = True
                        signal_strength = 0.7
                        
            # Breakout conditions
            elif 'breakout' in condition_text:
                if 'resistance_20d' in indicators and current_price > indicators['resistance_20d']:
                    condition_met = True
                    signal_strength = 0.8
                    
            # Time-based opportunity conditions
            elif 'time_opportunity' in condition_text:
                # This condition is always met but with varying strength based on time
                condition_met = True
                signal_strength = params.get('strength', 0.4)
                    
            # Price drop condition with dynamic thresholds
            elif 'price_dip_opportunity' in condition_text or 'price_drop' in condition_text:
                price_threshold = params.get('threshold', -1.0)
                if 'price_change_1d' in indicators and indicators['price_change_1d'] < price_threshold:
                    condition_met = True
                    signal_strength = params.get('strength', 0.5)
                    
            if condition_met:
                signals.append({
                    'condition': condition_text,
                    'strength': signal_strength,
                    'confidence': condition.get('confidence', 0.7),
                    'time_factor': time_strategy['aggressiveness'],
                    'volatility_factor': volatility_factor
                })
        
        # Always add time-based signals to ensure regular trading
        time_signals = self._generate_time_based_signals(current_date, indicators, current_price)
        signals.extend(time_signals)
                
        if signals:
            # Calculate overall signal strength with time and volatility factors
            # Ensure all values are numeric to prevent sequence multiplication errors
            total_strength = 0
            for s in signals:
                strength = float(s.get('strength', 0))
                confidence = float(s.get('confidence', 0))
                time_factor = float(s.get('time_factor', 1.0))
                total_strength += strength * confidence * time_factor
            
            avg_confidence = sum(float(s.get('confidence', 0)) for s in signals) / len(signals)
            
            # Apply volatility boost to confidence
            final_confidence = min(avg_confidence * (1 + float(volatility_factor) * 0.3), 1.0)
            
            return {
                'action': 'BUY',
                'strength': total_strength / len(signals),
                'confidence': final_confidence,
                'signals': signals,
                'time_strategy': time_strategy,
                'volatility_factor': volatility_factor,
                'timestamp': datetime.now().isoformat()
            }
            
        return None
        
    def evaluate_exit_conditions(self, entry_price: float, current_price: float, 
                               days_held: int, indicators: Dict[str, float]) -> Dict[str, Any]:
        """
        Evaluate AI strategy exit conditions
        
        Args:
            entry_price: Price at which position was entered
            current_price: Current stock price
            days_held: Number of days position has been held
            indicators: Technical indicators
            
        Returns:
            Exit signal information if conditions are met
        """
        if not self.ai_strategy or 'exit_conditions' not in self.ai_strategy:
            return None
            
        # Calculate current P&L
        pnl_pct = (current_price - entry_price) / entry_price
        
        exit_conditions = self.ai_strategy['exit_conditions']
        
        # Handle both dict and list formats
        if isinstance(exit_conditions, dict):
            # Check for specific exit conditions
            for key, value in exit_conditions.items():
                if key == 'rsi_overbought':
                    if 'rsi' in indicators and indicators['rsi'] > value:
                        return {
                            'action': 'SELL',
                            'reason': 'rsi_overbought',
                            'pnl_pct': pnl_pct,
                            'confidence': 0.9
                        }
                elif key == 'stop_loss':
                    if pnl_pct <= -value:
                        return {
                            'action': 'SELL',
                            'reason': 'stop_loss',
                            'pnl_pct': pnl_pct,
                            'confidence': 1.0
                        }
                elif key == 'take_profit':
                    if pnl_pct >= value:
                        return {
                            'action': 'SELL',
                            'reason': 'take_profit',
                            'pnl_pct': pnl_pct,
                            'confidence': 1.0
                        }
        
        # Check risk management rules
        risk_mgmt = self.ai_strategy.get('risk_management', {})
        
        # Stop loss check
        stop_loss_pct = risk_mgmt.get('stop_loss_pct', 0.02)
        if pnl_pct <= -stop_loss_pct:
            return {
                'action': 'SELL',
                'reason': 'stop_loss',
                'pnl_pct': pnl_pct,
                'confidence': 1.0
            }
            
        # Take profit check
        take_profit_pct = risk_mgmt.get('take_profit_pct', 0.04)
        if pnl_pct >= take_profit_pct:
            return {
                'action': 'SELL',
                'reason': 'take_profit',
                'pnl_pct': pnl_pct,
                'confidence': 1.0
            }
            
        # Time-based exit (max holding period)
        max_days = risk_mgmt.get('max_holding_days', 5)
        if days_held >= max_days:
            return {
                'action': 'SELL',
                'reason': 'time_limit',
                'pnl_pct': pnl_pct,
                'confidence': 0.7
            }
            
        # Technical exit conditions (for list format)
        if isinstance(exit_conditions, list):
            for condition in exit_conditions:
                condition_text = condition.get('condition', '').lower()
            
            # RSI overbought exit
            if 'rsi' in condition_text and 'overbought' in condition_text:
                if indicators.get('rsi', 50) > 75:
                    return {
                        'action': 'SELL',
                        'reason': 'rsi_overbought',
                        'pnl_pct': pnl_pct,
                        'confidence': 0.8
                    }
                    
        return None
        
    def run_ai_backtest(self, symbol: str, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """
        Run backtest using AI-generated strategy
        
        Args:
            symbol: Stock symbol to test
            start_date: Start date for backtest
            end_date: End date for backtest
            
        Returns:
            Comprehensive backtest results
        """
        if not self.ai_strategy:
            raise ValueError("No AI strategy loaded. Call load_ai_strategy() first.")
            
        self.reset()
        self.daily_signals = []
        self.daily_trades = []
        self.daily_pnl = []
        
        # Get historical data - use mock data as fallback
        try:
            from data.data_fetcher import get_daily_adjusted
            data = get_daily_adjusted(symbol)
            if 'Time Series (Daily)' not in data:
                print(f"No real data available for {symbol}, using mock data")
                mock_data = self.get_mock_data(symbol, days=(end_date - start_date).days + 50)
                time_series = {}
                for date, row in mock_data.iterrows():
                    date_str = date.strftime('%Y-%m-%d')
                    time_series[date_str] = {
                         '1. open': str(row['Open']),
                         '2. high': str(row['High']),
                         '3. low': str(row['Low']),
                         '4. close': str(row['Close']),
                         '5. adjusted close': str(row['Close']),
                         '6. volume': str(int(row['Volume']))
                     }
            else:
                time_series = data['Time Series (Daily)']
        except Exception as e:
            print(f"Error fetching real data for {symbol}: {e}, using mock data")
            mock_data = self.get_mock_data(symbol, days=(end_date - start_date).days + 50)
            time_series = {}
            for date, row in mock_data.iterrows():
                date_str = date.strftime('%Y-%m-%d')
                time_series[date_str] = {
                     '1. open': str(row['Open']),
                     '2. high': str(row['High']),
                     '3. low': str(row['Low']),
                     '4. close': str(row['Close']),
                     '5. adjusted close': str(row['Close']),
                     '6. volume': str(int(row['Volume']))
                 }
            
        # Filter dates and prepare data
        all_dates = []
        for date_str in sorted(time_series.keys()):
            # Skip invalid date strings
            if not date_str or len(date_str) < 8 or date_str in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                continue
            
            try:
                date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                if start_date <= date_obj <= end_date:
                    all_dates.append(date_str)
            except ValueError:
                # Skip invalid date formats
                print(f"Skipping invalid date format: {date_str}")
                continue
                
        if len(all_dates) < 20:
            raise ValueError("Insufficient data for backtesting (need at least 20 days)")
            
        # Track positions
        current_position = None
        position_entry_date = None
        position_entry_price = None
        
        # Run day-by-day simulation
        for i, date_str in enumerate(all_dates):
            day_data = time_series[date_str]
            current_price = float(day_data['4. close'])
            current_volume = int(day_data['6. volume'])
            
            # Prepare price history for indicators
            price_history = []
            volume_history = []
            
            # Get up to 50 days of history for indicators
            for j in range(max(0, i-49), i+1):
                if j < len(all_dates):
                    hist_date = all_dates[j]
                    hist_data = time_series[hist_date]
                    price_history.append(float(hist_data['4. close']))
                    volume_history.append(int(hist_data['6. volume']))
                    
            # Calculate indicators
            indicators = self.calculate_technical_indicators(price_history[::-1], volume_history[::-1])
            
            # Track daily metrics
            daily_info = {
                'date': date_str,
                'price': current_price,
                'volume': current_volume,
                'indicators': indicators.copy(),
                'signals': [],
                'trades': [],
                'position': current_position is not None
            }
            
            # Check exit conditions if we have a position
            if current_position:
                days_held = i - all_dates.index(position_entry_date)
                exit_signal = self.evaluate_exit_conditions(
                    position_entry_price, current_price, days_held, indicators
                )
                
                if exit_signal:
                    # Execute sell
                    shares = current_position['shares']
                    if self.execute_trade(symbol, 'SELL', shares, current_price, date_str):
                        daily_info['trades'].append({
                            'action': 'SELL',
                            'shares': shares,
                            'price': current_price,
                            'reason': exit_signal['reason'],
                            'pnl_pct': exit_signal['pnl_pct']
                        })
                        
                        # Record trade result
                        self.daily_trades.append({
                            'entry_date': position_entry_date,
                            'exit_date': date_str,
                            'entry_price': position_entry_price,
                            'exit_price': current_price,
                            'shares': shares,
                            'pnl_pct': exit_signal['pnl_pct'],
                            'days_held': days_held,
                            'exit_reason': exit_signal['reason']
                        })
                        
                        current_position = None
                        position_entry_date = None
                        position_entry_price = None
                        
            # Check entry conditions if we don't have a position
            else:
                if self.use_multi_timeframe:
                    # Use multi-timeframe strategy
                    entry_signals = self._evaluate_multi_timeframe_entry(
                        symbol, indicators, current_price, date_str, price_history, volume_history
                    )
                else:
                    # Use original single-timeframe strategy
                    entry_signals = [self.evaluate_entry_conditions(indicators, current_price, date_str)]
                
                # Process all entry signals
                for entry_signal in entry_signals:
                    if entry_signal and entry_signal['confidence'] > 0.4:
                        # Calculate position size based on timeframe and signal
                        timeframe = entry_signal.get('timeframe', '1d')
                        position_size = self._calculate_multi_timeframe_position_size(
                            entry_signal, timeframe, current_price
                        )
                        
                        shares = int(position_size / current_price) if position_size > 0 else 0
                        
                        if shares > 0 and self.execute_trade(symbol, 'BUY', shares, current_price, date_str):
                            current_position = {
                                'shares': shares, 
                                'entry_price': current_price,
                                'timeframe': timeframe,
                                'entry_signal': entry_signal
                            }
                            position_entry_date = date_str
                            position_entry_price = current_price
                            
                            daily_info['trades'].append({
                                'action': 'BUY',
                                'shares': shares,
                                'price': current_price,
                                'confidence': entry_signal['confidence'],
                                'timeframe': timeframe,
                                'strategy_type': 'multi_timeframe' if self.use_multi_timeframe else 'single',
                                'time_strategy': entry_signal.get('time_strategy', {}),
                                'volatility_factor': entry_signal.get('volatility_factor', 0.0)
                            })
                            
                            daily_info['signals'].append(entry_signal)
                            break  # Only take one position per day
                        
            # Calculate portfolio value
            portfolio_value = self.calculate_portfolio_value({symbol: current_price})
            self.portfolio_value_history.append({
                'date': date_str,
                'value': portfolio_value,
                'cash': self.current_capital,
                'positions_value': portfolio_value - self.current_capital
            })
            
            # Calculate daily P&L
            if i > 0:
                prev_value = self.portfolio_value_history[i-1]['value']
                daily_pnl_pct = (portfolio_value - prev_value) / prev_value * 100
                self.daily_pnl.append(daily_pnl_pct)
                daily_info['daily_pnl_pct'] = daily_pnl_pct
            else:
                daily_info['daily_pnl_pct'] = 0
                
            self.daily_signals.append(daily_info)
            
        # Calculate performance metrics
        return self.calculate_performance_metrics(symbol, start_date, end_date)
    
    def _evaluate_multi_timeframe_entry(self, symbol: str, indicators: Dict[str, float], 
                                      current_price: float, date_str: str, 
                                      price_history: List[float], volume_history: List[int]) -> List[Dict[str, Any]]:
        """
        Evaluate entry conditions using multi-timeframe strategy
        """
        try:
            # Prepare market data for multi-timeframe analysis
            market_data = {
                'current_price': current_price,
                'price_history': price_history,
                'volume_history': volume_history,
                'indicators': indicators
            }
            
            current_date = datetime.strptime(date_str, '%Y-%m-%d')
            
            # Generate coordinated signals across all timeframes
            timeframe_signals = self.multi_timeframe_strategy.generate_coordinated_signals(
                symbol, market_data, current_date
            )
            
            # Convert multi-timeframe signals to entry signals
            entry_signals = []
            for timeframe, signal_data in timeframe_signals.items():
                if signal_data['signal'] in [StrategySignal.BUY, StrategySignal.STRONG_BUY]:
                    entry_signal = {
                        'action': 'BUY',
                        'confidence': signal_data['confidence'],
                        'timeframe': timeframe.value,
                        'reason': signal_data['reason'],
                        'target_profit': signal_data.get('target_profit', 0.03),
                        'stop_loss': signal_data.get('stop_loss', 0.015),
                        'max_holding_period': signal_data.get('max_holding_period', '3d'),
                        'signal_strength': 1.0 if signal_data['signal'] == StrategySignal.STRONG_BUY else 0.7,
                        'coordination_note': signal_data.get('coordination_note', ''),
                        'timestamp': current_date.isoformat()
                    }
                    entry_signals.append(entry_signal)
            
            return entry_signals
            
        except Exception as e:
            print(f"Error in multi-timeframe evaluation: {e}")
            # Fallback to single timeframe
            single_signal = self.evaluate_entry_conditions(indicators, current_price, date_str)
            return [single_signal] if single_signal else []
    
    def _calculate_multi_timeframe_position_size(self, entry_signal: Dict[str, Any], 
                                               timeframe: str, current_price: float) -> float:
        """
        Calculate position size based on timeframe and signal strength
        """
        # Base allocation per timeframe
        timeframe_allocations = {
            '1d': 0.15,   # 15% for daily
            '3d': 0.20,   # 20% for 3-day
            '1w': 0.25,   # 25% for weekly
            '2w': 0.20,   # 20% for bi-weekly
            '1M': 0.20    # 20% for monthly
        }
        
        base_allocation = timeframe_allocations.get(timeframe, 0.1)
        signal_strength = entry_signal.get('signal_strength', 0.7)
        confidence = entry_signal.get('confidence', 0.5)
        
        # Calculate position size
        position_ratio = base_allocation * signal_strength * confidence
        position_size = self.current_capital * position_ratio
        
        return max(0, min(position_size, self.current_capital * 0.3))  # Cap at 30%
        
    def get_price_data_for_chart(self, symbol: str, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """
        Extract price data for chart visualization
        
        Returns:
            List of price data points with date and price
        """
        price_data = []
        
        # Extract price data from daily signals
        for daily_info in self.daily_signals:
            price_data.append({
                'date': daily_info['date'],
                'price': daily_info['price'],
                'volume': daily_info.get('volume', 0)
            })
            
        return price_data
        
    def calculate_performance_metrics(self, symbol: str, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """
        Calculate comprehensive performance metrics for the AI strategy backtest
        """
        if not self.portfolio_value_history or not self.daily_signals:
            return {'error': 'No backtest data available'}
            
        # Basic performance metrics
        initial_value = self.initial_capital
        final_value = self.portfolio_value_history[-1]['value']
        total_return = (final_value - initial_value) / initial_value * 100
        
        # Daily returns
        daily_returns = []
        for i in range(1, len(self.portfolio_value_history)):
            prev_val = self.portfolio_value_history[i-1]['value']
            curr_val = self.portfolio_value_history[i]['value']
            daily_ret = (curr_val - prev_val) / prev_val
            daily_returns.append(daily_ret)
            
        # Risk metrics
        volatility = np.std(daily_returns) * np.sqrt(252) * 100 if daily_returns else 0
        
        # Drawdown calculation
        peak = initial_value
        max_drawdown = 0
        for record in self.portfolio_value_history:
            if record['value'] > peak:
                peak = record['value']
            drawdown = (peak - record['value']) / peak * 100
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                
        # Trading statistics
        total_trades = len(self.daily_trades)
        winning_trades = len([t for t in self.daily_trades if t['pnl_pct'] > 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Trading date range
        start_trading_date = None
        end_trading_date = None
        if self.daily_trades:
            start_trading_date = min(trade['entry_date'] for trade in self.daily_trades)
            end_trading_date = max(trade['exit_date'] for trade in self.daily_trades if trade.get('exit_date'))
        
        # Daily trading frequency
        trading_days = len([d for d in self.daily_signals if d['trades']])
        total_days = len(self.daily_signals)
        daily_trading_frequency = (trading_days / total_days * 100) if total_days > 0 else 0
        
        # Daily profit frequency
        profitable_days = len([d for d in self.daily_signals if d.get('daily_pnl_pct', 0) > 0])
        daily_profit_frequency = (profitable_days / total_days * 100) if total_days > 0 else 0
        
        # Average trade metrics
        if self.daily_trades:
            avg_trade_return = np.mean([t['pnl_pct'] for t in self.daily_trades]) * 100
            avg_holding_period = np.mean([t['days_held'] for t in self.daily_trades])
        else:
            avg_trade_return = 0
            avg_holding_period = 0
            
        return {
            'strategy_name': self.ai_strategy.get('strategy_name', 'AI Strategy'),
            'symbol': symbol,
            'backtest_period': {
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d'),
                'total_days': total_days
            },
            'performance_metrics': {
                'total_return_pct': round(total_return, 2),
                'annualized_volatility_pct': round(volatility, 2),
                'max_drawdown_pct': round(max_drawdown, 2),
                'sharpe_ratio': round(total_return / volatility, 2) if volatility > 0 else 0
            },
            'trading_statistics': {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'win_rate_pct': round(win_rate, 2),
                'avg_trade_return_pct': round(avg_trade_return, 2),
                'avg_holding_period_days': round(avg_holding_period, 1),
                'start_trading_date': start_trading_date,
                'end_trading_date': end_trading_date
            },
            'daily_analysis': {
                'daily_trading_frequency_pct': round(daily_trading_frequency, 2),
                'daily_profit_frequency_pct': round(daily_profit_frequency, 2),
                'days_with_trades': trading_days,
                'days_with_profits': profitable_days,
                'avg_daily_return_pct': round(np.mean(self.daily_pnl), 2) if self.daily_pnl else 0
            },
            'final_portfolio': {
                'initial_capital': initial_value,
                'final_value': round(final_value, 2),
                'cash_remaining': round(self.current_capital, 2),
                'positions_value': round(final_value - self.current_capital, 2)
            },
            'trade_history': self.trade_history,  # All individual trades
            'detailed_trades': self.daily_trades,  # Complete trade records with entry/exit
            'portfolio_history': self.portfolio_value_history,  # Portfolio value over time
            'daily_signals_summary': {
                'total_signals_generated': sum(len(d['signals']) for d in self.daily_signals),
                'signals_per_day': round(sum(len(d['signals']) for d in self.daily_signals) / total_days, 2) if total_days > 0 else 0
            },
            'price_data': self.get_price_data_for_chart(symbol, start_date, end_date)  # Add price data for chart
        }

# Global instance
ai_backtester = AIBacktester()

def initialize_ai_backtester():
    """Initialize the AI backtester"""
    global ai_backtester
    ai_backtester = AIBacktester()
    return ai_backtester