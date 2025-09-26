"""
Intraday Data Fetcher
Fetches intraday stock data (1m, 5m, 15m, 30m, 1h) from multiple sources
and manages local caching to reduce API calls
"""

import os
import json
import pandas as pd
import yfinance as yf
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import time

class IntradayDataFetcher:
    """
    Fetches and caches intraday stock data from multiple sources
    """
    
    def __init__(self, cache_dir: str = None):
        """
        Initialize the intraday data fetcher
        
        Args:
            cache_dir: Directory to store cached data
        """
        self.cache_dir = cache_dir or os.path.join(os.path.dirname(__file__), 'intraday_cache')
        self.logger = logging.getLogger(__name__)
        
        # Create cache directory if it doesn't exist
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        
        # Supported timeframes
        self.timeframes = ['1m', '5m', '15m', '30m', '1h']
        
        # Data source configurations
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        self.rate_limits = {
            'yfinance': {'calls_per_minute': 60, 'last_call': 0},
            'alpha_vantage': {'calls_per_minute': 5, 'last_call': 0}
        }
        
    def get_cache_path(self, symbol: str, timeframe: str) -> str:
        """Get cache file path for symbol and timeframe"""
        symbol_dir = os.path.join(self.cache_dir, symbol.upper())
        Path(symbol_dir).mkdir(parents=True, exist_ok=True)
        return os.path.join(symbol_dir, f"{timeframe}.json")
    
    def is_cache_valid(self, cache_path: str, max_age_minutes: int = 5) -> bool:
        """Check if cached data is still valid"""
        if not os.path.exists(cache_path):
            return False
        
        file_age = time.time() - os.path.getmtime(cache_path)
        return file_age < (max_age_minutes * 60)
    
    def load_from_cache(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Load data from cache if available and valid"""
        cache_path = self.get_cache_path(symbol, timeframe)
        
        if self.is_cache_valid(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                
                df = pd.DataFrame(data['data'])
                if not df.empty:
                    df['datetime'] = pd.to_datetime(df['datetime'])
                    df.set_index('datetime', inplace=True)
                    self.logger.info(f"Loaded {len(df)} records from cache for {symbol} {timeframe}")
                    return df
            except Exception as e:
                self.logger.error(f"Error loading cache for {symbol} {timeframe}: {e}")
        
        return None
    
    def save_to_cache(self, symbol: str, timeframe: str, data: pd.DataFrame):
        """Save data to cache"""
        cache_path = self.get_cache_path(symbol, timeframe)
        
        try:
            # Prepare data for JSON serialization
            cache_data = {
                'symbol': symbol,
                'timeframe': timeframe,
                'last_updated': datetime.now().isoformat(),
                'data': []
            }
            
            # Convert DataFrame to list of dictionaries
            for idx, row in data.iterrows():
                cache_data['data'].append({
                    'datetime': idx.isoformat(),
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': float(row['Close']),
                    'volume': int(row['Volume'])
                })
            
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f, indent=2)
            
            self.logger.info(f"Cached {len(data)} records for {symbol} {timeframe}")
            
        except Exception as e:
            self.logger.error(f"Error saving cache for {symbol} {timeframe}: {e}")
    
    def respect_rate_limit(self, source: str):
        """Respect API rate limits"""
        if source not in self.rate_limits:
            return
        
        rate_limit = self.rate_limits[source]
        time_since_last = time.time() - rate_limit['last_call']
        min_interval = 60 / rate_limit['calls_per_minute']
        
        if time_since_last < min_interval:
            sleep_time = min_interval - time_since_last
            self.logger.info(f"Rate limiting {source}: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)
        
        self.rate_limits[source]['last_call'] = time.time()
    
    def fetch_from_yfinance(self, symbol: str, timeframe: str, period: str = "1d") -> Optional[pd.DataFrame]:
        """Fetch intraday data from yfinance"""
        try:
            self.respect_rate_limit('yfinance')
            
            # Map our timeframes to yfinance intervals
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
            
            if not data.empty:
                self.logger.info(f"Fetched {len(data)} records from yfinance for {symbol} {timeframe}")
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
            
            # Map our timeframes to Alpha Vantage intervals
            interval_map = {
                '1m': '1min',
                '5m': '5min',
                '15m': '15min',
                '30m': '30min',
                '1h': '60min'
            }
            
            if timeframe not in interval_map:
                return None
            
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'TIME_SERIES_INTRADAY',
                'symbol': symbol,
                'interval': interval_map[timeframe],
                'apikey': self.alpha_vantage_key,
                'outputsize': 'compact'
            }
            
            response = requests.get(url, params=params)
            data = response.json()
            
            # Check for API errors
            if 'Error Message' in data:
                self.logger.error(f"Alpha Vantage error: {data['Error Message']}")
                return None
            
            if 'Note' in data:
                self.logger.warning(f"Alpha Vantage rate limit: {data['Note']}")
                return None
            
            # Extract time series data
            time_series_key = f"Time Series ({interval_map[timeframe]})"
            if time_series_key not in data:
                return None
            
            time_series = data[time_series_key]
            
            # Convert to DataFrame
            df_data = []
            for timestamp, values in time_series.items():
                df_data.append({
                    'datetime': pd.to_datetime(timestamp),
                    'Open': float(values['1. open']),
                    'High': float(values['2. high']),
                    'Low': float(values['3. low']),
                    'Close': float(values['4. close']),
                    'Volume': int(values['5. volume'])
                })
            
            df = pd.DataFrame(df_data)
            df.set_index('datetime', inplace=True)
            df.sort_index(inplace=True)
            
            if not df.empty:
                self.logger.info(f"Fetched {len(df)} records from Alpha Vantage for {symbol} {timeframe}")
                return df
            
        except Exception as e:
            self.logger.error(f"Error fetching from Alpha Vantage for {symbol} {timeframe}: {e}")
        
        return None
    
    def generate_intraday_from_daily(self, daily_data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Generate synthetic intraday data from daily data
        This is a fallback when no intraday data is available
        """
        try:
            if daily_data.empty:
                return pd.DataFrame()
            
            # Get the latest day's data
            latest_day = daily_data.iloc[-1]
            
            # Calculate number of intervals per day
            intervals_per_day = {
                '1m': 390,   # 6.5 hours * 60 minutes
                '5m': 78,    # 6.5 hours * 12 intervals
                '15m': 26,   # 6.5 hours * 4 intervals
                '30m': 13,   # 6.5 hours * 2 intervals
                '1h': 7      # 6.5 hours
            }
            
            if timeframe not in intervals_per_day:
                return pd.DataFrame()
            
            num_intervals = intervals_per_day[timeframe]
            
            # Generate synthetic intraday data
            intraday_data = []
            current_price = latest_day['Open']
            
            # Create timestamps for the trading day (9:30 AM to 4:00 PM EST)
            base_date = daily_data.index[-1].date()
            start_time = pd.Timestamp.combine(base_date, pd.Timestamp('09:30:00').time())
            
            interval_minutes = {
                '1m': 1, '5m': 5, '15m': 15, '30m': 30, '1h': 60
            }[timeframe]
            
            for i in range(num_intervals):
                timestamp = start_time + pd.Timedelta(minutes=i * interval_minutes)
                
                # Calculate price movement (simple random walk towards close)
                progress = i / (num_intervals - 1) if num_intervals > 1 else 1
                target_price = latest_day['Close']
                
                # Add some randomness
                volatility = (latest_day['High'] - latest_day['Low']) / latest_day['Close']
                random_factor = (pd.np.random.random() - 0.5) * volatility * 0.5
                
                # Calculate OHLC for this interval
                open_price = current_price
                close_price = current_price + (target_price - current_price) * 0.1 + random_factor * current_price
                
                high_price = max(open_price, close_price) * (1 + abs(random_factor) * 0.5)
                low_price = min(open_price, close_price) * (1 - abs(random_factor) * 0.5)
                
                # Ensure high >= max(open, close) and low <= min(open, close)
                high_price = max(high_price, open_price, close_price)
                low_price = min(low_price, open_price, close_price)
                
                # Generate volume (distribute daily volume)
                volume = int(latest_day['Volume'] / num_intervals * (0.5 + pd.np.random.random()))
                
                intraday_data.append({
                    'datetime': timestamp,
                    'Open': round(open_price, 2),
                    'High': round(high_price, 2),
                    'Low': round(low_price, 2),
                    'Close': round(close_price, 2),
                    'Volume': volume
                })
                
                current_price = close_price
            
            df = pd.DataFrame(intraday_data)
            df.set_index('datetime', inplace=True)
            
            self.logger.info(f"Generated {len(df)} synthetic intraday records for {timeframe}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error generating synthetic intraday data: {e}")
            return pd.DataFrame()
    
    def get_intraday_data(self, symbol: str, timeframe: str, force_refresh: bool = False) -> pd.DataFrame:
        """
        Get intraday data for a symbol and timeframe
        
        Args:
            symbol: Stock symbol
            timeframe: Time interval (1m, 5m, 15m, 30m, 1h)
            force_refresh: Force fetch from API instead of using cache
            
        Returns:
            DataFrame with intraday data
        """
        if timeframe not in self.timeframes:
            raise ValueError(f"Unsupported timeframe: {timeframe}. Supported: {self.timeframes}")
        
        # Try cache first (unless force refresh)
        if not force_refresh:
            cached_data = self.load_from_cache(symbol, timeframe)
            if cached_data is not None:
                return cached_data
        
        # Try to fetch from data sources
        data = None
        
        # Try yfinance first (usually has good intraday data)
        data = self.fetch_from_yfinance(symbol, timeframe)
        
        # If yfinance fails, try Alpha Vantage
        if data is None or data.empty:
            data = self.fetch_from_alpha_vantage(symbol, timeframe)
        
        # If all APIs fail, try to generate from daily data
        if data is None or data.empty:
            self.logger.warning(f"No intraday data available for {symbol} {timeframe}, generating from daily data")
            
            # Try to get daily data
            try:
                from data.data_fetcher import get_stock_data_with_failover
                daily_data = get_stock_data_with_failover(symbol, period='5d', market='us')
                if daily_data is not None and not daily_data.empty:
                    data = self.generate_intraday_from_daily(daily_data, timeframe)
            except Exception as e:
                self.logger.error(f"Error generating from daily data: {e}")
        
        # Cache the data if we got something
        if data is not None and not data.empty:
            self.save_to_cache(symbol, timeframe, data)
            return data
        
        # Return empty DataFrame if all else fails
        self.logger.error(f"Failed to get any data for {symbol} {timeframe}")
        return pd.DataFrame()
    
    def get_cache_info(self, symbol: str = None) -> Dict:
        """Get information about cached data"""
        cache_info = {}
        
        if symbol:
            # Info for specific symbol
            symbol_dir = os.path.join(self.cache_dir, symbol.upper())
            if os.path.exists(symbol_dir):
                cache_info[symbol] = {}
                for timeframe in self.timeframes:
                    cache_path = os.path.join(symbol_dir, f"{timeframe}.json")
                    if os.path.exists(cache_path):
                        stat = os.stat(cache_path)
                        cache_info[symbol][timeframe] = {
                            'size': stat.st_size,
                            'last_modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                            'is_valid': self.is_cache_valid(cache_path)
                        }
        else:
            # Info for all symbols
            if os.path.exists(self.cache_dir):
                for symbol_dir in os.listdir(self.cache_dir):
                    symbol_path = os.path.join(self.cache_dir, symbol_dir)
                    if os.path.isdir(symbol_path):
                        cache_info[symbol_dir] = {}
                        for timeframe in self.timeframes:
                            cache_path = os.path.join(symbol_path, f"{timeframe}.json")
                            if os.path.exists(cache_path):
                                stat = os.stat(cache_path)
                                cache_info[symbol_dir][timeframe] = {
                                    'size': stat.st_size,
                                    'last_modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                                    'is_valid': self.is_cache_valid(cache_path)
                                }
        
        return cache_info
    
    def clear_cache(self, symbol: str = None, timeframe: str = None):
        """Clear cached data"""
        if symbol and timeframe:
            # Clear specific symbol and timeframe
            cache_path = self.get_cache_path(symbol, timeframe)
            if os.path.exists(cache_path):
                os.remove(cache_path)
                self.logger.info(f"Cleared cache for {symbol} {timeframe}")
        elif symbol:
            # Clear all timeframes for symbol
            symbol_dir = os.path.join(self.cache_dir, symbol.upper())
            if os.path.exists(symbol_dir):
                for file in os.listdir(symbol_dir):
                    os.remove(os.path.join(symbol_dir, file))
                os.rmdir(symbol_dir)
                self.logger.info(f"Cleared all cache for {symbol}")
        else:
            # Clear all cache
            import shutil
            if os.path.exists(self.cache_dir):
                shutil.rmtree(self.cache_dir)
                Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
                self.logger.info("Cleared all cache")

# Global instance
intraday_fetcher = IntradayDataFetcher()