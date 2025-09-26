"""
Enhanced Intraday Data Fetcher
Fetches minute-level stock data (1m, 5m, 15m, 30m, 1h) from multiple sources
including price, volume, and chip distribution data
"""

import os
import json
import pandas as pd
import yfinance as yf
import requests
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

class EnhancedIntradayFetcher:
    """
    Enhanced intraday data fetcher with multiple data sources
    Supports 1m, 5m, 15m, 30m, 1h timeframes with price, volume, and chip data
    """
    
    def __init__(self, cache_dir: str = None):
        """
        Initialize the enhanced intraday data fetcher
        
        Args:
            cache_dir: Directory to store cached data
        """
        self.cache_dir = cache_dir or os.path.join(os.path.dirname(__file__), 'enhanced_intraday_cache')
        self.logger = logging.getLogger(__name__)
        
        # Create cache directory if it doesn't exist
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        
        # Supported timeframes
        self.timeframes = ['1m', '5m', '15m', '30m', '1h']
        
        # Data source configurations
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY', '51JCHRJ1QK3TUA1R')
        self.polygon_key = os.getenv('POLYGON_API_KEY')
        self.finnhub_key = os.getenv('FINNHUB_API_KEY')
        
        # Rate limits for different sources
        self.rate_limits = {
            'yfinance': {'calls_per_minute': 60, 'last_call': 0},
            'alpha_vantage': {'calls_per_minute': 5, 'last_call': 0},
            'polygon': {'calls_per_minute': 5, 'last_call': 0},
            'finnhub': {'calls_per_minute': 60, 'last_call': 0},
            'twelve_data': {'calls_per_minute': 8, 'last_call': 0}
        }
        
        # Data source priority
        self.data_sources = ['yfinance', 'alpha_vantage', 'polygon', 'finnhub', 'twelve_data']
        
    def get_cache_path(self, symbol: str, timeframe: str, data_type: str = 'ohlcv') -> str:
        """Get cache file path for symbol, timeframe, and data type"""
        symbol_dir = os.path.join(self.cache_dir, symbol.upper())
        Path(symbol_dir).mkdir(parents=True, exist_ok=True)
        return os.path.join(symbol_dir, f"{timeframe}_{data_type}.json")
    
    def is_cache_valid(self, cache_path: str, max_age_minutes: int = 5) -> bool:
        """Check if cache file is valid (not expired)"""
        if not os.path.exists(cache_path):
            return False
        
        file_age = time.time() - os.path.getmtime(cache_path)
        return file_age < (max_age_minutes * 60)
    
    def load_from_cache(self, symbol: str, timeframe: str, data_type: str = 'ohlcv') -> Optional[pd.DataFrame]:
        """Load data from cache if valid"""
        cache_path = self.get_cache_path(symbol, timeframe, data_type)
        
        if not self.is_cache_valid(cache_path):
            return None
        
        try:
            with open(cache_path, 'r') as f:
                data = json.load(f)
            
            df = pd.DataFrame(data['data'])
            if not df.empty and 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            
            return df
        except Exception as e:
            self.logger.error(f"Error loading cache for {symbol} {timeframe}: {e}")
            return None
    
    def save_to_cache(self, symbol: str, timeframe: str, data: pd.DataFrame, data_type: str = 'ohlcv'):
        """Save data to cache"""
        cache_path = self.get_cache_path(symbol, timeframe, data_type)
        
        try:
            # Prepare data for JSON serialization
            cache_data = {
                'symbol': symbol,
                'timeframe': timeframe,
                'data_type': data_type,
                'timestamp': datetime.now().isoformat(),
                'data': data.reset_index().to_dict('records')
            }
            
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f, default=str)
                
        except Exception as e:
            self.logger.error(f"Error saving cache for {symbol} {timeframe}: {e}")
    
    def respect_rate_limit(self, source: str):
        """Respect rate limits for different data sources"""
        if source not in self.rate_limits:
            return
        
        rate_limit = self.rate_limits[source]
        current_time = time.time()
        
        if current_time - rate_limit['last_call'] < (60 / rate_limit['calls_per_minute']):
            sleep_time = (60 / rate_limit['calls_per_minute']) - (current_time - rate_limit['last_call'])
            time.sleep(sleep_time)
        
        self.rate_limits[source]['last_call'] = time.time()
    
    def fetch_from_yfinance(self, symbol: str, timeframe: str, period: str = "1d") -> Optional[pd.DataFrame]:
        """Fetch intraday data from Yahoo Finance"""
        try:
            self.respect_rate_limit('yfinance')
            
            # Map timeframes to yfinance intervals
            interval_map = {
                '1m': '1m',
                '5m': '5m',
                '15m': '15m',
                '30m': '30m',
                '1h': '1h'
            }
            
            if timeframe not in interval_map:
                return None
            
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval_map[timeframe])
            
            if data.empty:
                return None
            
            # Standardize column names
            data.columns = [col.lower().replace(' ', '_') for col in data.columns]
            
            # Add volume ratio and other indicators
            if 'volume' in data.columns:
                data['volume_sma_20'] = data['volume'].rolling(20).mean()
                data['volume_ratio'] = data['volume'] / data['volume_sma_20']
                data['volume_ratio'] = data['volume_ratio'].fillna(1.0)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching from yfinance for {symbol} {timeframe}: {e}")
            return None
    
    def fetch_from_alpha_vantage(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Fetch intraday data from Alpha Vantage"""
        if not self.alpha_vantage_key:
            return None
        
        try:
            self.respect_rate_limit('alpha_vantage')
            
            # Map timeframes to Alpha Vantage intervals
            interval_map = {
                '1m': '1min',
                '5m': '5min',
                '15m': '15min',
                '30m': '30min',
                '1h': '60min'
            }
            
            if timeframe not in interval_map:
                return None
            
            url = f"https://www.alphavantage.co/query"
            params = {
                'function': 'TIME_SERIES_INTRADAY',
                'symbol': symbol,
                'interval': interval_map[timeframe],
                'apikey': self.alpha_vantage_key,
                'outputsize': 'full'
            }
            
            response = requests.get(url, params=params, timeout=30)
            data = response.json()
            
            # Check for API errors
            if 'Error Message' in data or 'Note' in data:
                return None
            
            # Extract time series data
            time_series_key = f"Time Series ({interval_map[timeframe]})"
            if time_series_key not in data:
                return None
            
            time_series = data[time_series_key]
            
            # Convert to DataFrame
            df_data = []
            for timestamp, values in time_series.items():
                row = {
                    'timestamp': pd.to_datetime(timestamp),
                    'open': float(values['1. open']),
                    'high': float(values['2. high']),
                    'low': float(values['3. low']),
                    'close': float(values['4. close']),
                    'volume': int(values['5. volume'])
                }
                df_data.append(row)
            
            df = pd.DataFrame(df_data)
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            # Add volume indicators
            if 'volume' in df.columns:
                df['volume_sma_20'] = df['volume'].rolling(20).mean()
                df['volume_ratio'] = df['volume'] / df['volume_sma_20']
                df['volume_ratio'] = df['volume_ratio'].fillna(1.0)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching from Alpha Vantage for {symbol} {timeframe}: {e}")
            return None
    
    def fetch_from_polygon(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Fetch intraday data from Polygon.io"""
        if not self.polygon_key:
            return None
        
        try:
            self.respect_rate_limit('polygon')
            
            # Map timeframes to Polygon multipliers
            multiplier_map = {
                '1m': (1, 'minute'),
                '5m': (5, 'minute'),
                '15m': (15, 'minute'),
                '30m': (30, 'minute'),
                '1h': (1, 'hour')
            }
            
            if timeframe not in multiplier_map:
                return None
            
            multiplier, timespan = multiplier_map[timeframe]
            
            # Get data for the last trading day
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            
            url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{start_date}/{end_date}"
            params = {'apikey': self.polygon_key}
            
            response = requests.get(url, params=params, timeout=30)
            data = response.json()
            
            if 'results' not in data or not data['results']:
                return None
            
            # Convert to DataFrame
            df_data = []
            for result in data['results']:
                row = {
                    'timestamp': pd.to_datetime(result['t'], unit='ms'),
                    'open': result['o'],
                    'high': result['h'],
                    'low': result['l'],
                    'close': result['c'],
                    'volume': result['v']
                }
                df_data.append(row)
            
            df = pd.DataFrame(df_data)
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching from Polygon for {symbol} {timeframe}: {e}")
            return None
    
    def fetch_from_twelve_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Fetch intraday data from Twelve Data (free tier)"""
        try:
            self.respect_rate_limit('twelve_data')
            
            url = "https://api.twelvedata.com/time_series"
            params = {
                'symbol': symbol,
                'interval': timeframe,
                'outputsize': '100',  # Free tier limit
                'format': 'JSON'
            }
            
            response = requests.get(url, params=params, timeout=30)
            data = response.json()
            
            if 'values' not in data:
                return None
            
            # Convert to DataFrame
            df_data = []
            for item in data['values']:
                row = {
                    'timestamp': pd.to_datetime(item['datetime']),
                    'open': float(item['open']),
                    'high': float(item['high']),
                    'low': float(item['low']),
                    'close': float(item['close']),
                    'volume': int(item['volume'])
                }
                df_data.append(row)
            
            df = pd.DataFrame(df_data)
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching from Twelve Data for {symbol} {timeframe}: {e}")
            return None
    
    def calculate_chip_distribution(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate chip distribution data from OHLCV data"""
        if data.empty:
            return data
        
        try:
            # Calculate price levels
            data['price_range'] = data['high'] - data['low']
            data['price_center'] = (data['high'] + data['low']) / 2
            
            # Calculate volume-weighted price levels
            data['vwap'] = (data['volume'] * data['price_center']).cumsum() / data['volume'].cumsum()
            
            # Calculate chip concentration levels
            data['chip_concentration'] = data['volume'] / data['price_range']
            data['chip_concentration'] = data['chip_concentration'].fillna(0)
            
            # Calculate support and resistance levels
            window = min(20, len(data))
            data['support_level'] = data['low'].rolling(window).min()
            data['resistance_level'] = data['high'].rolling(window).max()
            
            # Calculate chip distribution percentiles
            data['chip_25'] = data['close'].rolling(window).quantile(0.25)
            data['chip_50'] = data['close'].rolling(window).quantile(0.50)
            data['chip_75'] = data['close'].rolling(window).quantile(0.75)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error calculating chip distribution: {e}")
            return data
    
    def get_intraday_data(self, symbol: str, timeframe: str, force_refresh: bool = False, 
                         include_chips: bool = True) -> Optional[pd.DataFrame]:
        """
        Get comprehensive intraday data with price, volume, and chip distribution
        
        Args:
            symbol: Stock symbol
            timeframe: Time interval (1m, 5m, 15m, 30m, 1h)
            force_refresh: Force refresh from API
            include_chips: Include chip distribution calculations
            
        Returns:
            DataFrame with OHLCV data and additional indicators
        """
        if timeframe not in self.timeframes:
            self.logger.error(f"Unsupported timeframe: {timeframe}")
            return None
        
        # Try to load from cache first
        if not force_refresh:
            cached_data = self.load_from_cache(symbol, timeframe)
            if cached_data is not None:
                return cached_data
        
        # Try each data source in order
        data = None
        for source in self.data_sources:
            try:
                if source == 'yfinance':
                    data = self.fetch_from_yfinance(symbol, timeframe)
                elif source == 'alpha_vantage':
                    data = self.fetch_from_alpha_vantage(symbol, timeframe)
                elif source == 'polygon':
                    data = self.fetch_from_polygon(symbol, timeframe)
                elif source == 'twelve_data':
                    data = self.fetch_from_twelve_data(symbol, timeframe)
                
                if data is not None and not data.empty:
                    self.logger.info(f"Successfully fetched {symbol} {timeframe} data from {source}")
                    break
                    
            except Exception as e:
                self.logger.error(f"Error fetching from {source}: {e}")
                continue
        
        if data is None or data.empty:
            self.logger.warning(f"No data available for {symbol} {timeframe}")
            return None
        
        # Add technical indicators
        try:
            # Price indicators
            data['sma_5'] = data['close'].rolling(5).mean()
            data['sma_20'] = data['close'].rolling(20).mean()
            data['ema_12'] = data['close'].ewm(span=12).mean()
            data['ema_26'] = data['close'].ewm(span=26).mean()
            
            # RSI
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            data['macd'] = data['ema_12'] - data['ema_26']
            data['macd_signal'] = data['macd'].ewm(span=9).mean()
            data['macd_histogram'] = data['macd'] - data['macd_signal']
            
            # Bollinger Bands
            data['bb_middle'] = data['close'].rolling(20).mean()
            bb_std = data['close'].rolling(20).std()
            data['bb_upper'] = data['bb_middle'] + (bb_std * 2)
            data['bb_lower'] = data['bb_middle'] - (bb_std * 2)
            
            # Volume indicators
            if 'volume' in data.columns:
                data['volume_sma'] = data['volume'].rolling(20).mean()
                data['volume_ratio'] = data['volume'] / data['volume_sma']
                data['volume_ratio'] = data['volume_ratio'].fillna(1.0)
            
            # Add chip distribution if requested
            if include_chips:
                data = self.calculate_chip_distribution(data)
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {e}")
        
        # Save to cache
        self.save_to_cache(symbol, timeframe, data)
        
        return data
    
    def get_multiple_timeframes(self, symbol: str, timeframes: List[str] = None, 
                              force_refresh: bool = False) -> Dict[str, pd.DataFrame]:
        """Get data for multiple timeframes"""
        if timeframes is None:
            timeframes = self.timeframes
        
        results = {}
        for tf in timeframes:
            data = self.get_intraday_data(symbol, tf, force_refresh)
            if data is not None:
                results[tf] = data
        
        return results
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cached data"""
        cache_info = {
            'cache_dir': str(self.cache_dir),
            'symbols': [],
            'total_files': 0,
            'total_size_mb': 0
        }
        
        try:
            for symbol_dir in self.cache_dir.iterdir():
                if symbol_dir.is_dir():
                    symbol_info = {
                        'symbol': symbol_dir.name,
                        'timeframes': [],
                        'files': 0,
                        'size_mb': 0
                    }
                    
                    for cache_file in symbol_dir.iterdir():
                        if cache_file.is_file() and cache_file.suffix == '.json':
                            symbol_info['files'] += 1
                            file_size = cache_file.stat().st_size / (1024 * 1024)
                            symbol_info['size_mb'] += file_size
                            
                            # Extract timeframe from filename
                            timeframe = cache_file.stem.split('_')[0]
                            if timeframe not in symbol_info['timeframes']:
                                symbol_info['timeframes'].append(timeframe)
                    
                    cache_info['symbols'].append(symbol_info)
                    cache_info['total_files'] += symbol_info['files']
                    cache_info['total_size_mb'] += symbol_info['size_mb']
        
        except Exception as e:
            self.logger.error(f"Error getting cache info: {e}")
        
        return cache_info
    
    def clear_cache(self, symbol: str = None, timeframe: str = None):
        """Clear cache data"""
        try:
            if symbol and timeframe:
                # Clear specific symbol and timeframe
                cache_path = self.get_cache_path(symbol, timeframe)
                if os.path.exists(cache_path):
                    os.remove(cache_path)
            elif symbol:
                # Clear all data for a symbol
                symbol_dir = os.path.join(self.cache_dir, symbol.upper())
                if os.path.exists(symbol_dir):
                    import shutil
                    shutil.rmtree(symbol_dir)
            else:
                # Clear all cache
                import shutil
                shutil.rmtree(self.cache_dir)
                Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
                
        except Exception as e:
            self.logger.error(f"Error clearing cache: {e}")

# Global instance
enhanced_intraday_fetcher = EnhancedIntradayFetcher()