import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from data.data_fetcher import get_daily_adjusted

class TradingStrategy:
    def __init__(self):
        self.signals = []
        self.indicators = {}
        
    def calculate_moving_average(self, prices, window):
        """Calculate simple moving average"""
        return pd.Series(prices).rolling(window=window).mean().tolist()
        
    def calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index"""
        prices = pd.Series(prices)
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.tolist()
        
    def detect_golden_cross(self, symbol):
        """Detect golden cross pattern (50-day MA crosses above 200-day MA)"""
        try:
            data = get_daily_adjusted(symbol)
            if 'Time Series (Daily)' not in data:
                return None
                
            time_series = data['Time Series (Daily)']
            dates = list(time_series.keys())[:250]  # Get last 250 days
            prices = [float(time_series[date]['4. close']) for date in dates]
            prices.reverse()  # Reverse to chronological order
            
            if len(prices) < 200:
                return None
                
            ma_50 = self.calculate_moving_average(prices, 50)
            ma_200 = self.calculate_moving_average(prices, 200)
            
            # Check for golden cross in recent days
            for i in range(len(ma_50) - 5, len(ma_50)):
                if (ma_50[i] > ma_200[i] and 
                    ma_50[i-1] <= ma_200[i-1]):
                    signal = {
                        'symbol': symbol,
                        'signal_type': 'Golden Cross',
                        'action': 'BUY',
                        'timestamp': datetime.now().isoformat(),
                        'price': prices[-1],
                        'confidence': 0.75,
                        'reason': f'50-day MA ({ma_50[i]:.2f}) crossed above 200-day MA ({ma_200[i]:.2f})'
                    }
                    return signal
            return None
            
        except Exception as e:
            print(f"Error detecting golden cross for {symbol}: {e}")
            return None
            
    def detect_oversold_condition(self, symbol):
        """Detect oversold condition using RSI"""
        try:
            data = get_daily_adjusted(symbol)
            if 'Time Series (Daily)' not in data:
                return None
                
            time_series = data['Time Series (Daily)']
            dates = list(time_series.keys())[:30]  # Get last 30 days
            prices = [float(time_series[date]['4. close']) for date in dates]
            prices.reverse()  # Reverse to chronological order
            
            rsi_values = self.calculate_rsi(prices)
            current_rsi = rsi_values[-1] if rsi_values else None
            
            if current_rsi and current_rsi < 30:
                signal = {
                    'symbol': symbol,
                    'signal_type': 'Oversold RSI',
                    'action': 'BUY',
                    'timestamp': datetime.now().isoformat(),
                    'price': prices[-1],
                    'confidence': 0.65,
                    'reason': f'RSI is oversold at {current_rsi:.2f}'
                }
                return signal
            return None
            
        except Exception as e:
            print(f"Error detecting oversold condition for {symbol}: {e}")
            return None
            
    def generate_signals_for_stock(self, symbol):
        """Generate all possible signals for a given stock"""
        signals = []
        
        # Check for different signal types
        golden_cross = self.detect_golden_cross(symbol)
        if golden_cross:
            signals.append(golden_cross)
            
        oversold = self.detect_oversold_condition(symbol)
        if oversold:
            signals.append(oversold)
            
        return signals
        
    def scan_market_for_signals(self, symbols):
        """Scan multiple stocks for trading signals"""
        all_signals = []
        
        for symbol in symbols:
            try:
                signals = self.generate_signals_for_stock(symbol)
                all_signals.extend(signals)
            except Exception as e:
                print(f"Error scanning {symbol}: {e}")
                
        # Sort by confidence
        all_signals.sort(key=lambda x: x['confidence'], reverse=True)
        return all_signals

# Global strategy instance
trading_strategy = TradingStrategy()

# Export for easy importing
__all__ = ['trading_strategy', 'TradingStrategy']