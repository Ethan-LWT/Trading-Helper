#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data fetcher module for stock data retrieval from multiple sources
"""

import os
import json
import pandas as pd
import yfinance as yf
import requests
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

# Try to import multi_source_manager, create fallback if not available
try:
    from .multi_source_manager import DataSourceManager
except ImportError:
    # Create a simple fallback data manager
    class DataSourceManager:
        def get_stock_data(self, symbol: str, period: str = '1y'):
            try:
                stock = yf.Ticker(symbol)
                return stock.history(period=period)
            except:
                return None

# Initialize data source manager
data_manager = DataSourceManager()

def get_daily_adjusted(symbol: str, outputsize: str = 'compact') -> Optional[Dict[str, Any]]:
    """
    Get daily adjusted stock data in Alpha Vantage format
    
    Args:
        symbol: Stock symbol
        outputsize: 'compact' or 'full'
        
    Returns:
        Dictionary with Alpha Vantage format data or None if failed
    """
    try:
        # Try to get data from multi-source manager
        df = data_manager.get_stock_data(symbol, period='1y' if outputsize == 'compact' else '5y')
        
        if df is None or df.empty:
            return None
            
        # Convert DataFrame to Alpha Vantage format
        time_series = {}
        
        for date, row in df.iterrows():
            # Ensure proper date formatting
            if hasattr(date, 'strftime'):
                date_str = date.strftime('%Y-%m-%d')
            elif isinstance(date, str):
                # Try to parse and reformat if it's already a string
                try:
                    parsed_date = pd.to_datetime(date)
                    date_str = parsed_date.strftime('%Y-%m-%d')
                except:
                    date_str = str(date)[:10]
            else:
                date_str = str(date)[:10]
            
            # Skip invalid dates or rows with all zero values
            if date_str == '0' or date_str.startswith('0'):
                continue
                
            # Get price values with proper validation
            open_val = row.get('Open', row.get('open', None))
            high_val = row.get('High', row.get('high', None))
            low_val = row.get('Low', row.get('low', None))
            close_val = row.get('Close', row.get('close', None))
            adj_close_val = row.get('Adj Close', close_val)
            volume_val = row.get('Volume', row.get('volume', 0))
            
            # Skip rows with invalid price data
            if not all(val is not None and val != 0 for val in [open_val, high_val, low_val, close_val]):
                continue
                
            time_series[date_str] = {
                '1. open': str(float(open_val)),
                '2. high': str(float(high_val)),
                '3. low': str(float(low_val)),
                '4. close': str(float(close_val)),
                '5. adjusted close': str(float(adj_close_val if adj_close_val is not None else close_val)),
                '6. volume': str(int(float(volume_val)) if volume_val else 0)
            }
        
        return {
            'Meta Data': {
                '1. Information': 'Daily Prices (open, high, low, close) and Volumes',
                '2. Symbol': symbol.upper(),
                '3. Last Refreshed': datetime.now().strftime('%Y-%m-%d'),
                '4. Output Size': outputsize,
                '5. Time Zone': 'US/Eastern'
            },
            'Time Series (Daily)': time_series
        }
        
    except Exception as e:
        print(f"Error in get_daily_adjusted for {symbol}: {e}")
        return None

def get_stock_data_with_failover(symbol: str, period: str = '1y', market: str = 'us', force_refresh: bool = False) -> Optional[pd.DataFrame]:
    """
    Get stock data with automatic failover between sources
    
    Args:
        symbol: Stock symbol
        period: Time period
        market: Market type ('us', 'cn', 'hk')
        force_refresh: Force refresh from API
        
    Returns:
        DataFrame with stock data or None if failed
    """
    return data_manager.get_stock_data(symbol, period, force_refresh=force_refresh, market=market)

def get_stock_data_unified(symbol: str, source: str, start_date: str = None, end_date: str = None, market: str = 'us') -> Optional[pd.DataFrame]:
    """
    Get stock data from a specific source
    
    Args:
        symbol: Stock symbol
        source: Data source ('yfinance', 'alpha_vantage', 'tushare')
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        market: Market type
        
    Returns:
        DataFrame with stock data or None if failed
    """
    try:
        if source == 'yfinance':
            ticker = yf.Ticker(symbol)
            if start_date and end_date:
                data = ticker.history(start=start_date, end=end_date)
            else:
                data = ticker.history(period='1y')
            return data if not data.empty else None
            
        elif source == 'alpha_vantage':
            result = get_daily_adjusted(symbol)
            if result and 'Time Series (Daily)' in result:
                # Convert back to DataFrame
                time_series = result['Time Series (Daily)']
                df_data = []
                for date_str, values in time_series.items():
                    df_data.append({
                        'Date': pd.to_datetime(date_str),
                        'Open': float(values['1. open']),
                        'High': float(values['2. high']),
                        'Low': float(values['3. low']),
                        'Close': float(values['4. close']),
                        'Adj Close': float(values['5. adjusted close']),
                        'Volume': int(values['6. volume'])
                    })
                df = pd.DataFrame(df_data)
                df.set_index('Date', inplace=True)
                return df.sort_index()
            return None
            
        else:
            # Use multi-source manager as fallback
            return data_manager.get_stock_data(symbol, market=market)
            
    except Exception as e:
        print(f"Error in get_stock_data_unified for {symbol} from {source}: {e}")
        return None

def get_tushare_daily_data(symbol: str, start_date: str = None, end_date: str = None) -> Optional[pd.DataFrame]:
    """
    Get daily data from Tushare (placeholder - implement if Tushare is available)
    
    Args:
        symbol: Stock symbol
        start_date: Start date
        end_date: End date
        
    Returns:
        DataFrame with stock data or None if failed
    """
    try:
        # Use multi-source manager which may have Tushare support
        return data_manager.get_stock_data(symbol, market='cn')
    except Exception as e:
        print(f"Error in get_tushare_daily_data for {symbol}: {e}")
        return None

def get_tushare_us_daily_data(symbol: str, start_date: str = None, end_date: str = None) -> Optional[pd.DataFrame]:
    """
    Get US daily data from Tushare (placeholder)
    
    Args:
        symbol: Stock symbol
        start_date: Start date
        end_date: End date
        
    Returns:
        DataFrame with stock data or None if failed
    """
    try:
        return data_manager.get_stock_data(symbol, market='us')
    except Exception as e:
        print(f"Error in get_tushare_us_daily_data for {symbol}: {e}")
        return None

def get_tushare_hk_daily_data(symbol: str, start_date: str = None, end_date: str = None) -> Optional[pd.DataFrame]:
    """
    Get Hong Kong daily data from Tushare (placeholder)
    
    Args:
        symbol: Stock symbol
        start_date: Start date
        end_date: End date
        
    Returns:
        DataFrame with stock data or None if failed
    """
    try:
        return data_manager.get_stock_data(symbol, market='hk')
    except Exception as e:
        print(f"Error in get_tushare_hk_daily_data for {symbol}: {e}")
        return None