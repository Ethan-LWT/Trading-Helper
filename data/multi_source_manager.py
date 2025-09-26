#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-source data manager with automatic failover and local caching
"""

import os
import json
import time
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
import yfinance as yf
import requests
from pathlib import Path

# Try to import scraper, create fallback if not available
try:
    from scrapers.stock_scraper import StockDataScraper
except ImportError:
    # Create a simple fallback scraper
    class StockDataScraper:
        def get_stock_data_with_fallback(self, symbol: str, period: str = "1y"):
            try:
                stock = yf.Ticker(symbol)
                return stock.history(period=period)
            except:
                return None

class DataSourceManager:
    """Manages multiple data sources with automatic failover and caching"""
    
    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Data source priority order
        self.data_sources = [
            'yfinance',
            'alpha_vantage', 
            'tushare',
            'scraper'
        ]
        
        # Cache settings
        self.cache_expiry_hours = 1  # Cache expires after 1 hour for intraday
        self.daily_cache_expiry_hours = 24  # Daily data cache expires after 24 hours
        
        # Alpha Vantage API key (if available)
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY', '51JCHRJ1QK3TUA1R')
        
        # Rate limiting
        self.last_request_time = {}
        self.min_request_interval = 1.0  # Minimum seconds between requests
        
        # Initialize scraper
        self.stock_scraper = StockDataScraper()
    
    def _get_cache_path(self, symbol: str, data_type: str = 'daily') -> Path:
        """Get cache file path for a symbol"""
        symbol_dir = self.cache_dir / symbol.upper()
        symbol_dir.mkdir(exist_ok=True)
        return symbol_dir / f"{data_type}.json"
    
    def _is_cache_valid(self, cache_path: Path, data_type: str = 'daily') -> bool:
        """Check if cached data is still valid"""
        if not cache_path.exists():
            return False
        
        # Get file modification time
        mod_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
        now = datetime.now()
        
        # Determine cache expiry based on data type
        if data_type == 'daily':
            expiry_hours = self.daily_cache_expiry_hours
        else:
            expiry_hours = self.cache_expiry_hours
        
        return (now - mod_time).total_seconds() < expiry_hours * 3600
    
    def _load_from_cache(self, cache_path: Path) -> Optional[pd.DataFrame]:
        """Load data from cache file"""
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
            
            # If there's a 'date' column, set it as index
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            
            return df
        except Exception as e:
            print(f"Error loading cache: {e}")
            return None
    
    def _save_to_cache(self, data: pd.DataFrame, cache_path: Path) -> None:
        """Save data to cache file"""
        try:
            # Reset index to include date as a column for JSON serialization
            data_to_save = data.reset_index()
            
            # Convert DataFrame to JSON-serializable format
            data_dict = data_to_save.to_dict('records')
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(data_dict, f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            print(f"Error saving cache: {e}")
    
    def _rate_limit(self, source: str) -> None:
        """Apply rate limiting for API calls"""
        if source in self.last_request_time:
            elapsed = time.time() - self.last_request_time[source]
            if elapsed < self.min_request_interval:
                time.sleep(self.min_request_interval - elapsed)
        
        self.last_request_time[source] = time.time()
    
    def _fetch_yfinance_data(self, symbol: str, period: str = '1y', interval: str = '1d') -> Optional[pd.DataFrame]:
        """Fetch data from Yahoo Finance"""
        try:
            self._rate_limit('yfinance')
            
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                return None
            
            # Standardize column names
            data.columns = [col.lower().replace(' ', '_') for col in data.columns]
            
            # Ensure the index is properly named as 'date'
            data.index.name = 'date'
            
            # Reset index to make date a column for consistent handling
            data = data.reset_index()
            
            # Ensure date column is properly formatted
            data['date'] = pd.to_datetime(data['date']).dt.tz_localize(None)
            
            # Set date as index again
            data.set_index('date', inplace=True)
            
            # Add source information
            data['source'] = 'yfinance'
            data['symbol'] = symbol
            
            print(f"✓ Successfully retrieved {symbol} data from yfinance")
            return data
            
        except Exception as e:
            print(f"Yahoo Finance error for {symbol}: {e}")
            return None
    
    def _fetch_alpha_vantage_data(self, symbol: str, function: str = 'TIME_SERIES_DAILY') -> Optional[pd.DataFrame]:
        """Fetch data from Alpha Vantage"""
        try:
            self._rate_limit('alpha_vantage')
            
            url = 'https://www.alphavantage.co/query'
            params = {
                'function': function,
                'symbol': symbol,
                'apikey': self.alpha_vantage_key,
                'outputsize': 'full'
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Check for API limit or error
            if 'Error Message' in data or 'Note' in data:
                print(f"Alpha Vantage API limit or error: {data}")
                return None
            
            # Extract time series data
            time_series_key = None
            for key in data.keys():
                if 'Time Series' in key:
                    time_series_key = key
                    break
            
            if not time_series_key:
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(data[time_series_key], orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            # Standardize column names
            df.columns = [col.split('. ')[1].lower().replace(' ', '_') for col in df.columns]
            
            # Convert to numeric
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Reset index to make date a column, then set it back as index
            df = df.reset_index()
            df.rename(columns={'index': 'date'}, inplace=True)
            df.set_index('date', inplace=True)
            
            # Add source information
            df['source'] = 'alpha_vantage'
            df['symbol'] = symbol
            
            print(f"✓ Successfully retrieved {symbol} data from alpha_vantage")
            return df
            
        except Exception as e:
            print(f"Alpha Vantage error for {symbol}: {e}")
            return None
    
    def _fetch_tushare_data(self, symbol: str, start_date: str = None, end_date: str = None, market: str = 'us') -> Optional[pd.DataFrame]:
        """Fetch data from Tushare"""
        try:
            # Use existing Tushare functionality
            data = data_fetcher.get_stock_data_unified(symbol, 'tushare', start_date, end_date, market)
            
            if data is not None and not data.empty:
                # Add source information
                data['source'] = 'tushare'
                data['symbol'] = symbol
                return data
            
            return None
            
        except Exception as e:
            print(f"Tushare error for {symbol}: {e}")
            return None
    
    def _fetch_scraper_data(self, symbol: str, period: str = '1y') -> Optional[pd.DataFrame]:
        """Fetch data using web scraper as fallback"""
        try:
            data = self.stock_scraper.get_stock_data_with_fallback(symbol, period)
            
            if data is not None and not data.empty:
                # Standardize column names if necessary
                data.columns = [col.lower() for col in data.columns]
                
                # Ensure date index
                if 'date' in data.columns:
                    data.set_index('date', inplace=True)
                
                data['source'] = 'scraper'
                data['symbol'] = symbol
                
                print(f"✓ Successfully retrieved {symbol} data from scraper")
                return data
            
            return None
            
        except Exception as e:
            print(f"Scraper error for {symbol}: {e}")
            return None
    
    def get_stock_data(self, symbol: str, period: str = '1y', data_type: str = 'daily', 
                      force_refresh: bool = False, market: str = 'us') -> Optional[pd.DataFrame]:
        """Get stock data with automatic source failover and caching
        
        Args:
            symbol: Stock symbol
            period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            data_type: Type of data ('daily', 'intraday')
            force_refresh: Force refresh from API (ignore cache)
            market: Market type ('us', 'cn', 'hk')
        
        Returns:
            DataFrame with stock data or None if all sources fail
        """
        # Check cache first (unless force refresh)
        cache_path = self._get_cache_path(symbol, data_type)
        
        if not force_refresh and self._is_cache_valid(cache_path, data_type):
            cached_data = self._load_from_cache(cache_path)
            if cached_data is not None and not cached_data.empty:
                print(f"Loaded {symbol} data from cache")
                return cached_data
        
        # Try each data source in order
        for source in self.data_sources:
            print(f"Trying {source} for {symbol}...")
            
            try:
                data = None
                
                if source == 'yfinance':
                    # Yahoo Finance works best for US stocks
                    if market == 'us':
                        interval = '1d' if data_type == 'daily' else '1m'
                        data = self._fetch_yfinance_data(symbol, period, interval)
                
                elif source == 'alpha_vantage':
                    # Alpha Vantage as backup
                    if market == 'us':
                        function = 'TIME_SERIES_DAILY' if data_type == 'daily' else 'TIME_SERIES_INTRADAY'
                        data = self._fetch_alpha_vantage_data(symbol, function)
                
                elif source == 'tushare':
                    # Tushare for all markets
                    # Convert period to date range for Tushare
                    end_date = datetime.now().strftime('%Y%m%d')
                    if period == '1y':
                        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
                    elif period == '6mo':
                        start_date = (datetime.now() - timedelta(days=180)).strftime('%Y%m%d')
                    elif period == '3mo':
                        start_date = (datetime.now() - timedelta(days=90)).strftime('%Y%m%d')
                    else:
                        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
                    
                    data = self._fetch_tushare_data(symbol, start_date, end_date, market)
                
                elif source == 'scraper':
                    data = self._fetch_scraper_data(symbol, period)
                
                # If data retrieved successfully, cache it and return
                if data is not None and not data.empty:
                    print(f"✓ Successfully retrieved {symbol} data from {source}")
                    self._save_to_cache(data, cache_path)
                    return data
                
            except Exception as e:
                print(f"Error with {source} for {symbol}: {e}")
                continue
        
        print(f"✗ All data sources failed for {symbol}")
        return None
    
    def get_chip_distribution(self, symbol: str) -> Optional[Dict]:
        """Get chip distribution data using scraper"""
        try:
            return self.stock_scraper.get_chip_distribution(symbol)
        except Exception as e:
            print(f"Error getting chip distribution for {symbol}: {e}")
            return None
        
        for symbol in symbols:
            print(f"\nFetching data for {symbol}...")
            data = self.get_stock_data(symbol, **kwargs)
            if data is not None:
                results[symbol] = data
        
        return results
    
    def clear_cache(self, symbol: str = None) -> None:
        """Clear cache for a specific symbol or all symbols"""
        if symbol:
            symbol_dir = self.cache_dir / symbol.upper()
            if symbol_dir.exists():
                for file in symbol_dir.glob('*.json'):
                    file.unlink()
                print(f"Cleared cache for {symbol}")
        else:
            for symbol_dir in self.cache_dir.iterdir():
                if symbol_dir.is_dir():
                    for file in symbol_dir.glob('*.json'):
                        file.unlink()
            print("Cleared all cache")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cached data"""
        cache_info = {}
        
        for symbol_dir in self.cache_dir.iterdir():
            if symbol_dir.is_dir():
                symbol = symbol_dir.name
                cache_info[symbol] = {}
                
                for cache_file in symbol_dir.glob('*.json'):
                    data_type = cache_file.stem
                    mod_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
                    is_valid = self._is_cache_valid(cache_file, data_type)
                    
                    cache_info[symbol][data_type] = {
                        'last_updated': mod_time.isoformat(),
                        'is_valid': is_valid,
                        'file_size': cache_file.stat().st_size
                    }
        
        return cache_info

# Global instance
data_manager = DataSourceManager()

# Convenience functions
def get_stock_data_multi_source(symbol: str, **kwargs) -> Optional[pd.DataFrame]:
    """Convenience function to get stock data with multi-source support"""
    return data_manager.get_stock_data(symbol, **kwargs)

def get_multiple_stocks_data(symbols: List[str], **kwargs) -> Dict[str, pd.DataFrame]:
    """Convenience function to get multiple stocks data"""
    return data_manager.get_multiple_stocks(symbols, **kwargs)

def clear_stock_cache(symbol: str = None) -> None:
    """Convenience function to clear cache"""
    data_manager.clear_cache(symbol)

def get_cache_status() -> Dict[str, Any]:
    """Convenience function to get cache status"""
    return data_manager.get_cache_info()