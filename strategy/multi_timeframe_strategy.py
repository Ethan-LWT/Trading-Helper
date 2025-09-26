"""
Multi-Timeframe Trading Strategy Framework
Implements independent but logically connected trading strategies across different timeframes
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import json

class TimeFrame(Enum):
    DAILY = "1d"
    THREE_DAY = "3d"
    WEEKLY = "1w"
    BI_WEEKLY = "2w"
    MONTHLY = "1M"

class StrategySignal(Enum):
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"

class MultiTimeframeStrategy:
    """
    Multi-timeframe trading strategy that coordinates signals across different time horizons
    """
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.strategies = {}
        self.signal_history = {}
        self.position_allocations = {
            TimeFrame.DAILY: 0.15,      # 15% for daily trading
            TimeFrame.THREE_DAY: 0.20,  # 20% for 3-day swings
            TimeFrame.WEEKLY: 0.25,     # 25% for weekly trends
            TimeFrame.BI_WEEKLY: 0.20,  # 20% for bi-weekly momentum
            TimeFrame.MONTHLY: 0.20     # 20% for monthly positions
        }
        
        # Initialize individual strategies
        self._initialize_strategies()
    
    def _initialize_strategies(self):
        """Initialize all timeframe-specific strategies"""
        self.strategies = {
            TimeFrame.DAILY: DailyStrategy(),
            TimeFrame.THREE_DAY: ThreeDayStrategy(),
            TimeFrame.WEEKLY: WeeklyStrategy(),
            TimeFrame.BI_WEEKLY: BiWeeklyStrategy(),
            TimeFrame.MONTHLY: MonthlyStrategy()
        }
    
    def generate_coordinated_signals(self, symbol: str, market_data: Dict[str, Any], 
                                   current_date: datetime) -> Dict[TimeFrame, Dict[str, Any]]:
        """
        Generate coordinated signals across all timeframes with time-based enhancements
        
        Args:
            symbol: Stock symbol
            market_data: Market data including price, volume, indicators
            current_date: Current trading date
            
        Returns:
            Dictionary of signals for each timeframe
        """
        signals = {}
        
        # Get time-based strategy parameters for enhanced daily operations
        time_strategy = self._get_time_based_strategy(current_date)
        volatility_factor = self._calculate_volatility_factor(market_data.get('indicators', {}))
        
        # Generate individual timeframe signals with time-based enhancements
        for timeframe, strategy in self.strategies.items():
            signal = strategy.generate_signal(symbol, market_data, current_date)
            
            # Apply time-based enhancements to daily and 3-day strategies
            if timeframe in [TimeFrame.DAILY, TimeFrame.THREE_DAY]:
                signal = self._enhance_signal_with_time_factors(signal, time_strategy, volatility_factor)
            
            signals[timeframe] = signal
        
        # Add time-based signals to ensure daily activity
        daily_time_signals = self._generate_time_based_signals(current_date, market_data)
        if daily_time_signals:
            # Merge time-based signals with daily strategy
            if TimeFrame.DAILY in signals:
                signals[TimeFrame.DAILY] = self._merge_daily_signals(signals[TimeFrame.DAILY], daily_time_signals)
        
        # Apply coordination logic
        coordinated_signals = self._coordinate_signals(signals, market_data, current_date)
        
        # Store signal history
        self._update_signal_history(symbol, coordinated_signals, current_date)
        
        return coordinated_signals
    
    def _get_time_based_strategy(self, current_date: datetime) -> Dict[str, Any]:
        """
        Get time-based strategy parameters based on current date
        Enhanced version with detailed daily, weekly, and monthly adjustments
        """
        # Get day of week (0=Monday, 6=Sunday)
        day_of_week = current_date.weekday()
        # Get day of month
        day_of_month = current_date.day
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
        
        # A. 每日策略变化
        # Monday (0) - 保守开始
        if day_of_week == 0:
            strategy.update({
                'aggressiveness': 0.8,
                'frequency_multiplier': 1.2,
                'risk_tolerance': 0.9,
                'strategy_type': 'monday_conservative',
                'daily_signal_boost': 0.1
            })
        # Tuesday-Wednesday (1-2) - 激进中段
        elif day_of_week in [1, 2]:
            strategy.update({
                'aggressiveness': 1.3,
                'frequency_multiplier': 1.5,
                'risk_tolerance': 1.2,
                'strategy_type': 'midweek_aggressive',
                'daily_signal_boost': 0.2
            })
        # Thursday (3) - 预周末定位
        elif day_of_week == 3:
            strategy.update({
                'aggressiveness': 1.1,
                'frequency_multiplier': 1.3,
                'risk_tolerance': 1.0,
                'strategy_type': 'thursday_positioning',
                'daily_signal_boost': 0.15
            })
        # Friday (4) - 获利了结
        elif day_of_week == 4:
            strategy.update({
                'aggressiveness': 0.9,
                'frequency_multiplier': 1.1,
                'risk_tolerance': 0.8,
                'strategy_type': 'friday_profit_taking',
                'daily_signal_boost': 0.05
            })
        
        # B. 每周策略调整
        if week_of_month == 1:  # 第1周: 频率 +20%, 激进度 +10%
            strategy['frequency_multiplier'] *= 1.2
            strategy['aggressiveness'] *= 1.1
            strategy['weekly_boost'] = 'first_week'
        elif week_of_month == 4:  # 第4周: 频率 +30%, 激进度 +20%
            strategy['frequency_multiplier'] *= 1.3
            strategy['aggressiveness'] *= 1.2
            strategy['weekly_boost'] = 'fourth_week'
        
        # C. 每月策略周期
        if day_of_month <= 5:  # 月初 (1-5日): 频率 +15%
            strategy['frequency_multiplier'] *= 1.15
            strategy['monthly_phase'] = 'month_start'
        elif day_of_month >= 25:  # 月末 (25-31日): 频率 +25%
            strategy['frequency_multiplier'] *= 1.25
            strategy['monthly_phase'] = 'month_end'
        
        return strategy
    
    def _calculate_volatility_factor(self, indicators: Dict[str, float]) -> float:
        """
        Calculate market volatility factor to adjust strategy aggressiveness
        Enhanced version with detailed volatility components
        """
        volatility_factor = 0.0
        
        # 价格波动性 (0-40%)
        if 'volatility' in indicators:
            # Ensure volatility is a number
            volatility = indicators.get('volatility', 0)
            try:
                volatility = float(volatility) if volatility is not None else 0.0
            except (TypeError, ValueError):
                volatility = 0.0
            
            # Normalize volatility (assuming typical range 0-10)
            normalized_vol = min(volatility / 10.0, 1.0)
            volatility_factor += normalized_vol * 0.4
        
        # 价格动量波动性 (0-30%)
        if 'price_change_1d' in indicators:
            price_change = indicators.get('price_change_1d', 0)
            try:
                price_change = float(price_change) if price_change is not None else 0.0
            except (TypeError, ValueError):
                price_change = 0.0
                
            abs_change = abs(price_change)
            # High daily changes indicate volatility
            if abs_change > 3:
                volatility_factor += 0.3
            elif abs_change > 1.5:
                volatility_factor += 0.2
            elif abs_change > 0.8:
                volatility_factor += 0.1
        
        # 成交量波动性 (0-30%)
        if 'volume_ratio' in indicators:
            volume_ratio = indicators.get('volume_ratio', 1.0)
            try:
                volume_ratio = float(volume_ratio) if volume_ratio is not None else 1.0
            except (TypeError, ValueError):
                volume_ratio = 1.0
                
            if volume_ratio > 2.5:
                volatility_factor += 0.3
            elif volume_ratio > 1.8:
                volatility_factor += 0.2
            elif volume_ratio > 1.3:
                volatility_factor += 0.1
        
        return min(volatility_factor, 1.0)
    
    def _generate_time_based_signals(self, current_date: datetime, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate time-based signals to ensure daily trading activity
        Enhanced version with comprehensive daily signal generation
        """
        signals = []
        indicators = market_data.get('indicators', {})
        current_price = market_data.get('current_price', 0)
        
        day_of_week = current_date.weekday()
        day_of_month = current_date.day
        
        # 周一机会信号
        if day_of_week == 0:
            signals.append({
                'condition': 'monday_opportunity',
                'strength': 0.4,
                'confidence': 0.6,
                'time_factor': 1.2,
                'volatility_factor': 0.0,
                'reason': 'Monday conservative start opportunity'
            })
        
        # 中周动量信号
        if day_of_week in [1, 2]:
            if 'price_change_1d' in indicators:
                # Any price movement triggers signal with enhanced strength
                price_change = indicators.get('price_change_1d', 0)
                # Ensure price_change is a number, not a string or list
                try:
                    price_change = float(price_change) if price_change is not None else 0.0
                except (TypeError, ValueError):
                    price_change = 0.0
                
                if abs(price_change) > 0.3:  # Lower threshold
                    signals.append({
                        'condition': 'midweek_momentum',
                        'strength': 0.5 + min(abs(price_change) * 0.1, 0.3),
                        'confidence': 0.7,
                        'time_factor': 1.5,
                        'volatility_factor': 0.0,
                        'reason': 'Midweek aggressive momentum'
                    })
        
        # 周四定位信号
        if day_of_week == 3:
            signals.append({
                'condition': 'thursday_positioning',
                'strength': 0.45,
                'confidence': 0.65,
                'time_factor': 1.3,
                'volatility_factor': 0.0,
                'reason': 'Thursday pre-weekend positioning'
            })
        
        # 周五获利信号
        if day_of_week == 4:
            # Enhanced profit-taking signal
            if 'price_change_1d' in indicators:
                price_change = indicators.get('price_change_1d', 0)
                # Ensure price_change is a number, not a string or list
                try:
                    price_change = float(price_change) if price_change is not None else 0.0
                except (TypeError, ValueError):
                    price_change = 0.0
                
                if price_change > 0:
                    signals.append({
                        'condition': 'friday_profit_taking',
                        'strength': 0.3 + price_change * 0.1,
                        'confidence': 0.6,
                        'time_factor': 1.1,
                        'volatility_factor': 0.0,
                        'reason': 'Friday profit taking opportunity'
                    })
        
        # 月度周期信号
        if day_of_month <= 5 or day_of_month >= 25:
            cycle_strength = 0.5 if day_of_month >= 25 else 0.4
            signals.append({
                'condition': 'monthly_cycle_opportunity',
                'strength': cycle_strength,
                'confidence': 0.6,
                'time_factor': 1.25 if day_of_month >= 25 else 1.15,
                'volatility_factor': 0.0,
                'reason': f'Monthly cycle {"end" if day_of_month >= 25 else "start"} opportunity'
            })
        
        # 每日机会信号 (确保最低活动) - Enhanced to guarantee daily activity
        if len(signals) == 0 or day_of_week in [5, 6]:  # Weekend or no signals
            base_strength = 0.35
            
            # Enhance based on price movement
            if 'price_change_1d' in indicators:
                if indicators['price_change_1d'] < -0.5:  # Price drop
                    base_strength = 0.45
                elif indicators['price_change_1d'] > 0.5:  # Price rise
                    base_strength = 0.4
            
            # Enhance based on volume
            if 'volume_ratio' in indicators and indicators['volume_ratio'] > 1.1:
                base_strength += 0.1
            
            signals.append({
                'condition': 'daily_opportunity',
                'strength': min(base_strength, 0.6),
                'confidence': 0.5,
                'time_factor': 1.0,
                'volatility_factor': 0.0,
                'reason': 'Daily guaranteed trading opportunity'
            })
        
        return signals
    
    def _enhance_signal_with_time_factors(self, signal: Dict[str, Any], time_strategy: Dict[str, Any], 
                                        volatility_factor: float) -> Dict[str, Any]:
        """
        Enhance signals with time-based factors for better daily operations
        """
        enhanced_signal = signal.copy()
        
        try:
            # Apply time-based confidence boost with safe type conversion
            daily_boost = float(time_strategy.get('daily_signal_boost', 0.0))
            original_confidence = float(signal.get('confidence', 0.5))
            enhanced_signal['confidence'] = min(original_confidence + daily_boost, 1.0)
            
            # Apply frequency multiplier to signal strength with safe type conversion
            freq_multiplier = float(time_strategy.get('frequency_multiplier', 1.0))
            original_strength = float(signal.get('signal_strength', 0.7))
            enhanced_signal['signal_strength'] = original_strength * min(freq_multiplier, 1.5)
            
            # Apply volatility factor with safe type conversion
            volatility_factor = float(volatility_factor)
            current_confidence = float(enhanced_signal.get('confidence', 0.5))
            
            # 高波动期: 降低入场阈值，提高交易频率
            if volatility_factor > 0.5:
                enhanced_signal['confidence'] = min(current_confidence * 1.2, 1.0)
                enhanced_signal['volatility_adjustment'] = 'high_volatility_boost'
            # 低波动期: 保持标准阈值，确保质量
            elif volatility_factor < 0.2:
                enhanced_signal['confidence'] = max(current_confidence * 0.9, 0.3)
                enhanced_signal['volatility_adjustment'] = 'low_volatility_conservative'
            
        except (TypeError, ValueError):
            # If type conversion fails, keep original values
            pass
        
        # Add time strategy context
        enhanced_signal['time_strategy'] = time_strategy
        enhanced_signal['volatility_factor'] = volatility_factor
        
        return enhanced_signal
    
    def _merge_daily_signals(self, daily_signal: Dict[str, Any], time_signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge daily strategy signal with time-based signals for comprehensive daily operations
        """
        merged_signal = daily_signal.copy()
        
        if time_signals:
            # Find the strongest time-based signal with safe type conversion
            strongest_time_signal = None
            max_strength = 0
            
            for signal in time_signals:
                try:
                    strength = float(signal.get('strength', 0))
                    confidence = float(signal.get('confidence', 0))
                    combined_strength = strength * confidence
                    
                    if combined_strength > max_strength:
                        max_strength = combined_strength
                        strongest_time_signal = signal
                except (TypeError, ValueError):
                    # Skip signals with invalid strength/confidence values
                    continue
            
            # Enhance daily signal with time-based signal
            if strongest_time_signal and max_strength > 0.3:
                try:
                    # Safe type conversion for confidence calculation
                    daily_confidence = float(daily_signal.get('confidence', 0.5))
                    time_confidence = float(strongest_time_signal.get('confidence', 0))
                    
                    # Boost confidence and add time context
                    merged_signal['confidence'] = min(
                        daily_confidence + time_confidence * 0.3, 1.0
                    )
                    merged_signal['time_enhancement'] = strongest_time_signal
                    
                    # Safe string concatenation
                    daily_reason = str(daily_signal.get('reason', ''))
                    time_reason = str(strongest_time_signal.get('reason', ''))
                    merged_signal['reason'] = f"{daily_reason} + {time_reason}"
                    
                except (TypeError, ValueError):
                    # If type conversion fails, keep original signal
                    pass
        
        return merged_signal
    
    def _coordinate_signals(self, raw_signals: Dict[TimeFrame, Dict[str, Any]], 
                          market_data: Dict[str, Any], current_date: datetime) -> Dict[TimeFrame, Dict[str, Any]]:
        """
        Coordinate signals across timeframes to ensure logical consistency
        """
        coordinated = raw_signals.copy()
        
        # Get signal strengths
        monthly_signal = raw_signals[TimeFrame.MONTHLY]['signal']
        biweekly_signal = raw_signals[TimeFrame.BI_WEEKLY]['signal']
        weekly_signal = raw_signals[TimeFrame.WEEKLY]['signal']
        three_day_signal = raw_signals[TimeFrame.THREE_DAY]['signal']
        daily_signal = raw_signals[TimeFrame.DAILY]['signal']
        
        # Coordination Rules:
        
        # 1. Monthly strategy dominates long-term direction
        if monthly_signal in [StrategySignal.STRONG_SELL, StrategySignal.SELL]:
            # If monthly is bearish, limit bullish signals in shorter timeframes
            if biweekly_signal == StrategySignal.STRONG_BUY:
                coordinated[TimeFrame.BI_WEEKLY]['signal'] = StrategySignal.BUY
                coordinated[TimeFrame.BI_WEEKLY]['coordination_note'] = "Reduced from monthly bearish trend"
            
            if weekly_signal == StrategySignal.STRONG_BUY:
                coordinated[TimeFrame.WEEKLY]['signal'] = StrategySignal.BUY
                coordinated[TimeFrame.WEEKLY]['coordination_note'] = "Reduced from monthly bearish trend"
        
        # 2. Weekly strategy influences medium-term trades
        if weekly_signal in [StrategySignal.STRONG_SELL, StrategySignal.SELL]:
            if three_day_signal == StrategySignal.STRONG_BUY:
                coordinated[TimeFrame.THREE_DAY]['signal'] = StrategySignal.BUY
                coordinated[TimeFrame.THREE_DAY]['coordination_note'] = "Reduced from weekly bearish trend"
        
        # 3. Enhance signals when timeframes align
        if (monthly_signal in [StrategySignal.BUY, StrategySignal.STRONG_BUY] and
            weekly_signal in [StrategySignal.BUY, StrategySignal.STRONG_BUY]):
            
            # Boost shorter timeframe signals
            if three_day_signal == StrategySignal.BUY:
                coordinated[TimeFrame.THREE_DAY]['signal'] = StrategySignal.STRONG_BUY
                coordinated[TimeFrame.THREE_DAY]['coordination_note'] = "Enhanced by aligned longer timeframes"
            
            if daily_signal == StrategySignal.BUY:
                coordinated[TimeFrame.DAILY]['signal'] = StrategySignal.STRONG_BUY
                coordinated[TimeFrame.DAILY]['coordination_note'] = "Enhanced by aligned longer timeframes"
        
        # 4. Risk management - prevent conflicting positions
        active_long_timeframes = sum(1 for tf, signal in coordinated.items() 
                                   if signal['signal'] in [StrategySignal.BUY, StrategySignal.STRONG_BUY])
        
        if active_long_timeframes > 3:  # Too many long positions
            # Reduce daily position size
            coordinated[TimeFrame.DAILY]['position_multiplier'] = 0.7
            coordinated[TimeFrame.DAILY]['coordination_note'] = "Reduced size due to multiple long positions"
        
        return coordinated
    
    def _update_signal_history(self, symbol: str, signals: Dict[TimeFrame, Dict[str, Any]], 
                             current_date: datetime):
        """Update signal history for trend analysis"""
        if symbol not in self.signal_history:
            self.signal_history[symbol] = {}
        
        date_str = current_date.strftime('%Y-%m-%d')
        self.signal_history[symbol][date_str] = signals
    
    def calculate_position_sizes(self, signals: Dict[TimeFrame, Dict[str, Any]], 
                               available_capital: float) -> Dict[TimeFrame, float]:
        """
        Calculate position sizes for each timeframe based on signals and allocations
        """
        position_sizes = {}
        
        for timeframe, signal_data in signals.items():
            try:
                base_allocation = float(self.position_allocations[timeframe])
                signal_strength = float(self._get_signal_strength(signal_data['signal']))
                position_multiplier = float(signal_data.get('position_multiplier', 1.0))
                available_capital_float = float(available_capital)
                
                # Calculate position size with safe type conversion
                position_size = available_capital_float * base_allocation * signal_strength * position_multiplier
                position_sizes[timeframe] = max(0, position_size)
                
            except (TypeError, ValueError, KeyError):
                # If calculation fails, set position size to 0
                position_sizes[timeframe] = 0.0
        
        return position_sizes
    
    def _get_signal_strength(self, signal: StrategySignal) -> float:
        """Convert signal to position strength multiplier"""
        strength_map = {
            StrategySignal.STRONG_SELL: 0.0,
            StrategySignal.SELL: 0.0,
            StrategySignal.HOLD: 0.0,
            StrategySignal.BUY: 0.7,
            StrategySignal.STRONG_BUY: 1.0
        }
        return strength_map.get(signal, 0.0)


class BaseTimeframeStrategy:
    """Base class for timeframe-specific strategies"""
    
    def __init__(self, timeframe: TimeFrame):
        self.timeframe = timeframe
        self.name = f"{timeframe.value}_strategy"
    
    def generate_signal(self, symbol: str, market_data: Dict[str, Any], 
                       current_date: datetime) -> Dict[str, Any]:
        """Generate trading signal for this timeframe"""
        raise NotImplementedError("Subclasses must implement generate_signal")
    
    def _calculate_technical_indicators(self, price_data: List[float], 
                                      volume_data: List[int] = None) -> Dict[str, float]:
        """Calculate technical indicators for the timeframe"""
        if len(price_data) < 2:
            return {}
        
        prices = np.array(price_data)
        indicators = {}
        
        # Moving averages
        if len(prices) >= 5:
            indicators['ma_5'] = np.mean(prices[-5:])
        if len(prices) >= 10:
            indicators['ma_10'] = np.mean(prices[-10:])
        if len(prices) >= 20:
            indicators['ma_20'] = np.mean(prices[-20:])
        
        # RSI
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
            indicators['momentum_1'] = (prices[-1] - prices[-2]) / prices[-2] * 100
        if len(prices) >= 5:
            indicators['momentum_5'] = (prices[-1] - prices[-5]) / prices[-5] * 100
        
        # Volatility
        if len(prices) >= 10:
            indicators['volatility'] = np.std(prices[-10:])
        
        return indicators


class DailyStrategy(BaseTimeframeStrategy):
    """Daily trading strategy - focuses on intraday momentum and quick profits with enhanced daily operations"""
    
    def __init__(self):
        super().__init__(TimeFrame.DAILY)
        self.target_profit = 0.012   # 1.2% target (reduced for more frequent trades)
        self.stop_loss = 0.006       # 0.6% stop loss (tighter control)
        self.max_holding_hours = 18  # Hold max 18 hours (allow overnight)
    
    def generate_signal(self, symbol: str, market_data: Dict[str, Any], 
                       current_date: datetime) -> Dict[str, Any]:
        """Generate daily trading signal with enhanced daily activity focus"""
        
        price_data = market_data.get('price_history', [])
        volume_data = market_data.get('volume_history', [])
        current_price = market_data.get('current_price', 0)
        
        indicators = self._calculate_technical_indicators(price_data, volume_data)
        
        signal = StrategySignal.HOLD
        confidence = 0.5
        reason = "Daily analysis pending"
        
        # Enhanced daily strategy conditions for more frequent trading
        if len(indicators) > 0:
            rsi = indicators.get('rsi', 50)
            momentum_1 = indicators.get('momentum_1', 0)
            volatility = indicators.get('volatility', 0)
            ma_5 = indicators.get('ma_5', 0)
            
            # More aggressive buy conditions for daily activity
            # Strong buy conditions (lowered thresholds)
            if (rsi < 40 and momentum_1 > 0.8 and volatility > 0.8):
                signal = StrategySignal.STRONG_BUY
                confidence = 0.85
                reason = "Daily oversold bounce with momentum"
            
            # Regular buy conditions (more lenient)
            elif (rsi < 50 and momentum_1 > 0.3) or (current_price < ma_5 * 0.995 and rsi < 55):
                signal = StrategySignal.BUY
                confidence = 0.7
                reason = "Daily short-term opportunity"
            
            # Micro-dip buying (new condition for daily activity)
            elif (momentum_1 < -0.3 and momentum_1 > -1.5 and rsi > 35):
                signal = StrategySignal.BUY
                confidence = 0.6
                reason = "Daily micro-dip buying opportunity"
            
            # Volume-based buying (new condition)
            elif ('volume_ratio' in indicators and indicators['volume_ratio'] > 1.2 and rsi < 60):
                signal = StrategySignal.BUY
                confidence = 0.65
                reason = "Daily volume spike opportunity"
            
            # Sell conditions (profit taking and risk management)
            elif (rsi > 65 or momentum_1 < -1.2):
                signal = StrategySignal.SELL
                confidence = 0.75
                reason = "Daily profit taking or risk management"
        
        return {
            'signal': signal,
            'confidence': confidence,
            'reason': reason,
            'timeframe': self.timeframe.value,
            'target_profit': self.target_profit,
            'stop_loss': self.stop_loss,
            'max_holding_period': f"{self.max_holding_hours}h",
            'indicators': indicators,
            'signal_strength': 1.0 if signal == StrategySignal.STRONG_BUY else 0.7,
            'timestamp': current_date.isoformat()
        }


class ThreeDayStrategy(BaseTimeframeStrategy):
    """3-day swing trading strategy - captures short-term price swings with enhanced coordination"""
    
    def __init__(self):
        super().__init__(TimeFrame.THREE_DAY)
        self.target_profit = 0.025   # 2.5% target (slightly reduced for more activity)
        self.stop_loss = 0.012       # 1.2% stop loss
        self.max_holding_days = 3
    
    def generate_signal(self, symbol: str, market_data: Dict[str, Any], 
                       current_date: datetime) -> Dict[str, Any]:
        """Generate 3-day swing trading signal with enhanced swing detection"""
        
        price_data = market_data.get('price_history', [])
        indicators = self._calculate_technical_indicators(price_data)
        
        signal = StrategySignal.HOLD
        confidence = 0.5
        reason = "Swing analysis pending"
        
        if len(indicators) > 0:
            rsi = indicators.get('rsi', 50)
            momentum_5 = indicators.get('momentum_5', 0)
            ma_5 = indicators.get('ma_5', 0)
            ma_10 = indicators.get('ma_10', 0)
            ma_20 = indicators.get('ma_20', 0)
            
            current_price = market_data.get('current_price', 0)
            
            # Enhanced swing trading conditions
            # Strong buy - oversold with MA support (more lenient)
            if (rsi < 45 and momentum_5 > 1.5 and ma_5 > ma_20 and current_price > ma_5 * 0.98):
                signal = StrategySignal.STRONG_BUY
                confidence = 0.8
                reason = "3-day oversold swing with strong trend support"
            
            # Buy - pullback in uptrend (enhanced detection)
            elif (rsi < 55 and ma_5 > ma_20 and current_price < ma_5 * 0.99):
                signal = StrategySignal.BUY
                confidence = 0.7
                reason = "3-day pullback in uptrend"
            
            # Buy - momentum reversal (new condition)
            elif (momentum_5 < -1.0 and momentum_5 > -3.0 and rsi > 30 and rsi < 50):
                signal = StrategySignal.BUY
                confidence = 0.65
                reason = "3-day momentum reversal opportunity"
            
            # Buy - MA crossover support (new condition)
            elif (ma_5 > ma_10 and current_price > ma_10 and rsi < 60):
                signal = StrategySignal.BUY
                confidence = 0.6
                reason = "3-day MA crossover support"
            
            # Sell - overbought or trend break (adjusted)
            elif (rsi > 70 or (ma_5 < ma_20 and momentum_5 < -2.5)):
                signal = StrategySignal.SELL
                confidence = 0.75
                reason = "3-day overbought or trend reversal"
        
        return {
            'signal': signal,
            'confidence': confidence,
            'reason': reason,
            'timeframe': self.timeframe.value,
            'target_profit': self.target_profit,
            'stop_loss': self.stop_loss,
            'max_holding_period': f"{self.max_holding_days}d",
            'indicators': indicators,
            'signal_strength': 1.0 if signal == StrategySignal.STRONG_BUY else 0.7,
            'timestamp': current_date.isoformat()
        }


class WeeklyStrategy(BaseTimeframeStrategy):
    """Weekly trend-following strategy - captures medium-term trends"""
    
    def __init__(self):
        super().__init__(TimeFrame.WEEKLY)
        self.target_profit = 0.08   # 8% target
        self.stop_loss = 0.04       # 4% stop loss
        self.max_holding_weeks = 4
    
    def generate_signal(self, symbol: str, market_data: Dict[str, Any], 
                       current_date: datetime) -> Dict[str, Any]:
        """Generate weekly trend-following signal"""
        
        price_data = market_data.get('price_history', [])
        indicators = self._calculate_technical_indicators(price_data)
        
        signal = StrategySignal.HOLD
        confidence = 0.5
        reason = "Trend analysis pending"
        
        if len(indicators) > 0:
            ma_10 = indicators.get('ma_10', 0)
            ma_20 = indicators.get('ma_20', 0)
            momentum_5 = indicators.get('momentum_5', 0)
            rsi = indicators.get('rsi', 50)
            
            current_price = market_data.get('current_price', 0)
            
            # Strong uptrend
            if (ma_10 > ma_20 and current_price > ma_10 and momentum_5 > 3.0 and rsi > 50):
                signal = StrategySignal.STRONG_BUY
                confidence = 0.85
                reason = "Strong uptrend confirmed"
            
            # Emerging uptrend
            elif (ma_10 > ma_20 * 1.02 and current_price > ma_20 and rsi > 45):
                signal = StrategySignal.BUY
                confidence = 0.7
                reason = "Emerging uptrend"
            
            # Downtrend
            elif (ma_10 < ma_20 and current_price < ma_10 and momentum_5 < -3.0):
                signal = StrategySignal.SELL
                confidence = 0.8
                reason = "Downtrend confirmed"
        
        return {
            'signal': signal,
            'confidence': confidence,
            'reason': reason,
            'timeframe': self.timeframe.value,
            'target_profit': self.target_profit,
            'stop_loss': self.stop_loss,
            'max_holding_period': f"{self.max_holding_weeks}w",
            'indicators': indicators,
            'timestamp': current_date.isoformat()
        }


class BiWeeklyStrategy(BaseTimeframeStrategy):
    """Bi-weekly momentum strategy - captures momentum shifts"""
    
    def __init__(self):
        super().__init__(TimeFrame.BI_WEEKLY)
        self.target_profit = 0.12   # 12% target
        self.stop_loss = 0.06       # 6% stop loss
        self.max_holding_weeks = 8
    
    def generate_signal(self, symbol: str, market_data: Dict[str, Any], 
                       current_date: datetime) -> Dict[str, Any]:
        """Generate bi-weekly momentum signal"""
        
        price_data = market_data.get('price_history', [])
        indicators = self._calculate_technical_indicators(price_data)
        
        signal = StrategySignal.HOLD
        confidence = 0.5
        reason = "Momentum analysis"
        
        if len(indicators) > 0:
            momentum_5 = indicators.get('momentum_5', 0)
            volatility = indicators.get('volatility', 0)
            rsi = indicators.get('rsi', 50)
            ma_20 = indicators.get('ma_20', 0)
            
            current_price = market_data.get('current_price', 0)
            
            # Strong momentum with low volatility
            if (momentum_5 > 5.0 and volatility < 2.0 and rsi < 70 and current_price > ma_20):
                signal = StrategySignal.STRONG_BUY
                confidence = 0.8
                reason = "Strong momentum with stability"
            
            # Building momentum
            elif (momentum_5 > 2.0 and rsi > 40 and rsi < 65):
                signal = StrategySignal.BUY
                confidence = 0.65
                reason = "Building momentum"
            
            # Momentum breakdown
            elif (momentum_5 < -4.0 or rsi > 75):
                signal = StrategySignal.SELL
                confidence = 0.75
                reason = "Momentum breakdown"
        
        return {
            'signal': signal,
            'confidence': confidence,
            'reason': reason,
            'timeframe': self.timeframe.value,
            'target_profit': self.target_profit,
            'stop_loss': self.stop_loss,
            'max_holding_period': f"{self.max_holding_weeks}w",
            'indicators': indicators,
            'timestamp': current_date.isoformat()
        }


class MonthlyStrategy(BaseTimeframeStrategy):
    """Monthly position strategy - long-term trend and value investing"""
    
    def __init__(self):
        super().__init__(TimeFrame.MONTHLY)
        self.target_profit = 0.25   # 25% target
        self.stop_loss = 0.12       # 12% stop loss
        self.max_holding_months = 6
    
    def generate_signal(self, symbol: str, market_data: Dict[str, Any], 
                       current_date: datetime) -> Dict[str, Any]:
        """Generate monthly long-term position signal"""
        
        price_data = market_data.get('price_history', [])
        indicators = self._calculate_technical_indicators(price_data)
        
        signal = StrategySignal.HOLD
        confidence = 0.5
        reason = "Long-term analysis"
        
        if len(indicators) > 0:
            ma_20 = indicators.get('ma_20', 0)
            momentum_5 = indicators.get('momentum_5', 0)
            rsi = indicators.get('rsi', 50)
            volatility = indicators.get('volatility', 0)
            
            current_price = market_data.get('current_price', 0)
            
            # Long-term value opportunity
            if (current_price < ma_20 * 0.9 and rsi < 45 and momentum_5 > -5.0):
                signal = StrategySignal.STRONG_BUY
                confidence = 0.75
                reason = "Long-term value opportunity"
            
            # Gradual accumulation
            elif (current_price < ma_20 and rsi < 55):
                signal = StrategySignal.BUY
                confidence = 0.6
                reason = "Gradual accumulation zone"
            
            # Overvalued or major trend change
            elif (current_price > ma_20 * 1.3 or momentum_5 < -10.0):
                signal = StrategySignal.SELL
                confidence = 0.7
                reason = "Overvalued or major trend change"
        
        return {
            'signal': signal,
            'confidence': confidence,
            'reason': reason,
            'timeframe': self.timeframe.value,
            'target_profit': self.target_profit,
            'stop_loss': self.stop_loss,
            'max_holding_period': f"{self.max_holding_months}m",
            'indicators': indicators,
            'timestamp': current_date.isoformat()
        }